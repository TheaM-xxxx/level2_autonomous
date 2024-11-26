# -*- coding: utf-8 -*-
# @Author   : Xiyao Ma

import torch
import torch.nn as nn
import torch.nn.functional as F

from UCTransNet.CTrans_light import ChannelTransformer_light
from UCTransNet.CTrans_slim import ChannelTransformer_slim
from thop import profile


class Se_Module(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(Se_Module, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        xs = self.se(x)
        return x * xs

class DW_Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DW_se_Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,T=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*T, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels*T),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels*T, in_channels*T, kernel_size=3, stride=1, padding=1,groups=in_channels*T, bias=False),
            nn.BatchNorm2d(in_channels*T),
            nn.ReLU6(inplace=True)
        )
        self.se = Se_Module(in_channels*T)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels*T, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.se(out)
        out = self.conv2(out)
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,se_flag=False):
        super().__init__()
        if se_flag:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DW_se_Conv(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DW_Conv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super(ConvBatchNorm, self).__init__()
        self.conv = DW_Conv(in_channels, out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

def _make_nConv(in_channels, out_channels, nb_Conv):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels))
    return nn.Sequential(*layers)


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, cca_flag=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.cca_flag = cca_flag
        if cca_flag:
            self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv)

    def forward(self, x, skip_x):
        up = self.up(x)
        if self.cca_flag:
            skip_x_att = self.coatt(g=up, x=skip_x)
            x = torch.cat([skip_x_att, up], dim=1) # dim 1 is the channel dimension
        else:
            x = torch.cat([skip_x, up], dim=1)
        x = self.nConvs(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out



class Slim_UCTransNet(nn.Module):
    def __init__(self,in_ch=3, out_ch=1,img_size=512,cfg=None):
        super().__init__()

        self.n_channels = in_ch
        self.n_classes = out_ch
        in_channels = 64
        self.inc = ConvBatchNorm(in_ch, in_channels)

        self.down1 = Down(in_channels, in_channels * 2, se_flag=True)
        self.down2 = Down(in_channels * 2, in_channels * 4, se_flag=True)
        self.down3 = Down(in_channels * 4, in_channels * 8, se_flag=True)
        self.down4 = Down(in_channels * 8, in_channels * 8, se_flag=True)
        self.mtc = ChannelTransformer_slim(img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=[16,8,4,2],cfg=cfg)
        self.up4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2, cca_flag=True)
        self.up3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2, cca_flag=False)
        self.up2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2, cca_flag=False)
        self.up1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2, cca_flag=False)
        self.outc = nn.Conv2d(in_channels, out_ch, kernel_size=(1,1), stride=(1,1))

        # print("You are using UCTransNet!")

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1,x2,x3,x4,_ = self.mtc(x1,x2,x3,x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        logits = self.outc(x)

        return logits

class Light_UCTransNet(nn.Module):
    def __init__(self,in_ch=3, out_ch=1,img_size=512):
        super().__init__()

        self.n_channels = in_ch
        self.n_classes = out_ch
        in_channels = 64
        self.inc = ConvBatchNorm(in_ch, in_channels)

        self.down1 = Down(in_channels, in_channels*2, se_flag = True)
        self.down2 = Down(in_channels*2, in_channels*4, se_flag = True)
        self.down3 = Down(in_channels*4, in_channels*8, se_flag = True)
        self.down4 = Down(in_channels*8, in_channels*8, se_flag = True)

        self.mtc = ChannelTransformer_light(img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=[16,8,4,2])

        self.up4 = UpBlock_attention(in_channels*16, in_channels*4, nb_Conv=2,cca_flag=True)
        self.up3 = UpBlock_attention(in_channels*8, in_channels*2, nb_Conv=2,cca_flag=False)
        self.up2 = UpBlock_attention(in_channels*4, in_channels, nb_Conv=2,cca_flag=False)
        self.up1 = UpBlock_attention(in_channels*2, in_channels, nb_Conv=2,cca_flag=False)
        self.outc = nn.Conv2d(in_channels, out_ch, kernel_size=(1,1), stride=(1,1))

        # print("You are using UCTransNet!")

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1,x2,x3,x4,_ = self.mtc(x1,x2,x3,x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        logits = self.outc(x)

        return logits

if __name__ == '__main__':

    net = Light_UCTransNet(in_ch=3,out_ch=3)

    # n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print('number of params (M): %.2f' % (n_parameters / 1.e6))
    x = torch.rand([1, 3, 512, 512])
    flops, params = profile(net, inputs=(x,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))


    y = net(x)
    print(y.size())
