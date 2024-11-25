import argparse
import os
import time
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

from network import *
from datasets import build_dataloader
import torch.nn.functional as F
from metrics import multiclass_dice_coeff, multiclass_iou_coeff
from UCTransNet.CTrans_light import channel_selection



def get_args_parser():
    parser = argparse.ArgumentParser('Segmentation testing', add_help=False)

    # Dataset parameters
    parser.add_argument('--inference', default=True)
    parser.add_argument('--datapath', default='./dataset/', type=str, help='dataset path')
    parser.add_argument('--dataset', default='animal', type=str, help='dataset path')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--input_size', default=512, type=int, help='images input size')

    return parser


def pruning(net):
    total = 0
    for m in net.modules():
        if isinstance(m, channel_selection):
            total += m.indexes.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in net.modules():
        if isinstance(m, channel_selection):
            size = m.indexes.data.shape[0]
            bn[index:(index + size)] = m.indexes.data.abs().clone()
            index += size

    percent = 0.3

    y, i = torch.sort(bn)
    thre_index = int(total * percent)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    # for k, m in enumerate(net.modules()):
    for k, (name, m) in enumerate(net.named_modules()):
        if isinstance(m, channel_selection):

            weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.indexes.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())

            print('layer index: {:d} \t layer name: {:s} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, name, mask.shape[0], int(torch.sum(mask))))


    pruned_ratio = pruned / total
    print('Pre-processing Successful! ')
    print('Pruned ratio:', pruned_ratio.item())
    # print(cfg)
    print("Threshold:", thre.item())

    return cfg,cfg_mask

def apply_colors_to_mask(mask):
    category_to_color = {
        1: (255, 0, 0),  # red
        2: (0, 255, 0),  # green
    }

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for category, color in category_to_color.items():
        color_mask[mask == category] = color

    return color_mask

def inference(net, test_dataloader,args,vis=False):

    print(f"Start testing....")

    net.eval()
    dice_score = 0.
    iou_score = 0


    loop = tqdm((test_dataloader), total=len(test_dataloader))
    for (image, gt_mask, img_name) in loop:

        image = image.to(device)
        gt_mask = gt_mask.to(device)

        with torch.no_grad():
            with autocast():
                pred_mask = net(image)

        loop.set_description(f'Test')

        num_classes = pred_mask.shape[1]

        assert gt_mask.min() >= 0 and gt_mask.max() < num_classes, 'True mask indices should be in [0, n_classes['

        pred_mask_onehot = F.one_hot(pred_mask.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
        gt_mask_onehot = F.one_hot(gt_mask.squeeze(1).to(torch.int64), num_classes).permute(0, 3, 1, 2).float()

        dice = multiclass_dice_coeff(pred_mask_onehot[:, 1:], gt_mask_onehot[:, 1:])
        dice_score += dice
        iou = multiclass_iou_coeff(pred_mask_onehot[:, 1:], gt_mask_onehot[:, 1:])
        iou_score += iou

        # -----------------visualization----------------
        if vis:
            mask = pred_mask.argmax(dim=1).cpu().data.numpy()[0, :, :]
            # mask = gt_mask.cpu().data.numpy()[0, 0, :, :]

            colored_mask = apply_colors_to_mask(mask)

            plt.figure()
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image.cpu().data.numpy()[0, 0, :, :], cmap='gray')
            plt.imshow(colored_mask, alpha=.5)
            plt.show()
            # plt.savefig(os.path.join(args.output_dir, args.dataset, "masks/{}.png".format(img_name[0].split(".")[0])),
            #             bbox_inches='tight', pad_inches=0)
            plt.close()


    ave_dice = dice_score / len(test_dataloader)
    ave_IoU = iou_score / len(test_dataloader)

    print("\nAverage Dice:", str((ave_dice * 100).item()))
    print("Average IoU:", str((ave_IoU * 100).item()))


def measure_inference_time(net, test_dataloader, num_warmup_runs=10, num_runs=10):
    net.eval()

    fakeimage = torch.randn(1, 3, 512, 512).cuda()
    with torch.no_grad():
        with autocast():
            for _ in range(num_warmup_runs):
                _ = net(fakeimage)
                torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        times = []

        loop = tqdm((test_dataloader), total=len(test_dataloader))
        for (image, gt_mask, img_name) in loop:
            image = image.to(device)
            start_time = time.perf_counter()
            with torch.no_grad():
                with autocast():
                    _ = net(image)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_dev = np.std(times)

    print(f"Average inference time over {num_runs} runs: {avg_time:.6f} seconds")
    print(f"Standard deviation: {std_dev:.6f} seconds")
    print('FPS: {}'.format(1 / avg_time))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loader_test = build_dataloader(is_train=False,
                                       args=args,
                                       batch_size=1,
                                       drop_last=False,
                                       shuffle=False,
                                    num_workers=8)


    net = Light_UCTransNet(in_ch=3, out_ch=args.classes, img_size=args.input_size)
    net = nn.DataParallel(net)
    net = net.to(torch.float32)
    net.to(device)

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("params=", str(n_parameters / 1e6) + '{}'.format("M"))


    checkpoint_path = os.path.join('./checkpoints/{:s}_model.pth'.format(args.dataset))
    print("Loading", checkpoint_path)
    net.load_state_dict(torch.load(checkpoint_path),strict=False)

    inference(net, data_loader_test,args,vis=False)
    # measure_inference_time(net, data_loader_test, num_warmup_runs=20, num_runs=5)

    #####################
    Prune = True
    if Prune:
        cfg, cfg_mask = pruning(net)
        prune_num = 5 # kv+ffn*4=10
        cfg_prune = []
        temp = []
        for i in range(len(cfg)):
            if i % prune_num == 0 and i != 0:
                cfg_prune.append(temp)
                temp = []
            temp.append(cfg[i])
        cfg_prune.append(temp)

        newmodel = Slim_UCTransNet(in_ch=3, out_ch=args.classes, img_size=args.input_size,cfg=cfg_prune)
        newmodel = nn.DataParallel(newmodel)
        newmodel.to(device)

        n_parameters = sum(p.numel() for p in newmodel.parameters() if p.requires_grad)
        print("new params=", str(n_parameters / 1e6) + '{}'.format("M"))
        # count_parameters(newmodel)

        newmodel_dict = newmodel.state_dict().copy()

        newdict = {}
        for k, v in net.state_dict().items():
            if 'layer.0' in k:
                i = 0 * prune_num
            elif 'layer.1' in k:
                i = 1 * prune_num
            elif 'layer.2' in k:
                i = 2 * prune_num
            elif 'layer.3' in k:
                i = 3 * prune_num
            else:
                i = 0

            if 'key' in k or 'value' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
                newdict[k] = v[idx.tolist()].clone()
            elif 'ffn1.fc1.weight' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 1].cpu().numpy())))
                newdict[k] = v[idx.tolist()].clone()
            elif 'ffn1.fc1.bias' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 1].cpu().numpy())))
                newdict[k] = v[idx.tolist()].clone()
            elif 'ffn1.fc2.weight' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 1].cpu().numpy())))
                newdict[k] = v[:,idx.tolist()].clone()
            elif 'ffn2.fc1.weight' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 2].cpu().numpy())))
                newdict[k] = v[idx.tolist()].clone()
            elif 'ffn2.fc1.bias' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 2].cpu().numpy())))
                newdict[k] = v[idx.tolist()].clone()
            elif 'ffn2.fc2.weight' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 2].cpu().numpy())))
                newdict[k] = v[:,idx.tolist()].clone()
            elif 'ffn3.fc1.weight' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 3].cpu().numpy())))
                newdict[k] = v[idx.tolist()].clone()
            elif 'ffn3.fc1.bias' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 3].cpu().numpy())))
                newdict[k] = v[idx.tolist()].clone()
            elif 'ffn3.fc2.weight' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 3].cpu().numpy())))
                newdict[k] = v[:,idx.tolist()].clone()
            elif 'ffn4.fc1.weight' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 4].cpu().numpy())))
                newdict[k] = v[idx.tolist()].clone()
            elif 'ffn4.fc1.bias' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 4].cpu().numpy())))
                newdict[k] = v[idx.tolist()].clone()
            elif 'ffn4.fc2.weight' in k:
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i + 4].cpu().numpy())))
                newdict[k] = v[:,idx.tolist()].clone()
            elif k in newmodel.state_dict():
                newdict[k] = v

        newmodel_dict.update(newdict)
        newmodel.load_state_dict(newmodel_dict)

        print('after pruning: ', end=' ')

        inference(newmodel, data_loader_test, args,vis=True)
        # measure_inference_time(newmodel, data_loader_test, num_warmup_runs=20, num_runs=5)
