import os
# import PIL
import cv2
import json
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader, TensorDataset
import matplotlib.pyplot as plt


class CoroSeg(Dataset):

    def __init__(self,is_train,args):
        self.aug_rate = 4
        self.is_train = is_train
        self.reshape = args.input_size
        self.inference = args.inference
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
        ])
        self.dataset = args.dataset
        if self.inference:
            self.imgpath = os.path.join(args.datapath, args.dataset, 'test', 'imgs')
            self.maskpath = os.path.join(args.datapath, args.dataset, 'test', 'masks')
        else:
            self.imgpath = os.path.join(args.datapath, args.dataset, 'train' if is_train else 'val', 'imgs')
            self.maskpath = os.path.join(args.datapath, args.dataset, 'train' if is_train else 'val', 'masks')

        self.imagename = sorted(os.listdir(self.imgpath))


    def __getitem__(self, idx):

        if self.inference:
            original_idx = idx
        else:
            original_idx = idx // self.aug_rate

        img_name = self.imagename[original_idx]

        img = cv2.imread(os.path.join(os.path.abspath(self.imgpath), img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.reshape, self.reshape), interpolation=cv2.INTER_LINEAR)

        enhan_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        enhan_img = cv2.bilateralFilter(enhan_img, d=7, sigmaColor=20, sigmaSpace=20)
        enhan_img = enhan_img.astype(np.float32)
        enhan_img = enhan_img / 255.
        enhan_img = enhan_img.transpose(2, 0, 1)


        mask = cv2.imread(os.path.join(os.path.abspath(self.maskpath), img_name.replace('jpg','png')), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.reshape, self.reshape), interpolation=cv2.INTER_LINEAR)
        mask = mask.astype(np.float32)
        if mask.max() == 255:
            mask = mask / 255.
        mask = mask[np.newaxis, :,:]

        enhan_img = torch.from_numpy(np.array(enhan_img))

        mask = torch.from_numpy(np.array(mask))

        if not self.is_train:
            return enhan_img, mask, img_name
        else:
            if idx % self.aug_rate == 0:
                return enhan_img, mask
            else:
                seed = np.random.randint(0, 2 ** 31 - 1)
                random.seed(seed)
                torch.cuda.manual_seed(seed)
                torch.manual_seed(seed)
                enhan_img = self.transform(enhan_img)
                random.seed(seed)
                torch.cuda.manual_seed(seed)
                torch.manual_seed(seed)
                mask = self.transform(mask)
                #
                #
                # plt.imshow(enhan_img[0,:,:], cmap='gray')
                # plt.show()
                # plt.imshow(mask[0,:,:])
                # plt.show()
                return enhan_img, mask


    def __len__(self):
        if self.is_train and not self.inference:
            return len(self.imagename) * self.aug_rate
        else:
            return len(self.imagename)

class CoroSeg_animal(Dataset):

    def __init__(self,is_train,args):
        self.aug_rate = 8
        self.is_train = is_train
        self.reshape = args.input_size
        self.inference = args.inference
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.3),
        ])
        self.dataset = args.dataset
        if self.inference:
            self.imgpath = os.path.join(args.datapath, args.dataset, 'test', 'imgs')
            self.maskpath = os.path.join(args.datapath, args.dataset, 'test', 'masks')
        else:
            self.imgpath = os.path.join(args.datapath, args.dataset, 'train' if is_train else 'val', 'imgs')
            self.maskpath = os.path.join(args.datapath, args.dataset, 'train' if is_train else 'val', 'masks')

        self.imagename = sorted(os.listdir(self.imgpath))


    def __getitem__(self, idx):

        if self.inference:
            original_idx = idx
        else:
            original_idx = idx // self.aug_rate

        img_name = self.imagename[original_idx]

        img = cv2.imread(os.path.join(os.path.abspath(self.imgpath), img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.reshape, self.reshape), interpolation=cv2.INTER_LINEAR)

        enhan_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        enhan_img = cv2.bilateralFilter(enhan_img, d=7, sigmaColor=20, sigmaSpace=20)
        # enhan_img = clahe(enhan_img)
        enhan_img = enhan_img.astype(np.float32)
        enhan_img = enhan_img / 255.
        enhan_img = enhan_img.transpose(2, 0, 1)


        mask = cv2.imread(os.path.join(os.path.abspath(self.maskpath), img_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.reshape, self.reshape), interpolation=cv2.INTER_LINEAR) #[256,256]
        mask = mask.astype(np.float32)
        mask = mask[np.newaxis, :,:]

        enhan_img = torch.from_numpy(np.array(enhan_img))

        mask = torch.from_numpy(np.array(mask))

        if not self.is_train:
            return enhan_img, mask, img_name
        else:
            if idx % self.aug_rate == 0:
                return enhan_img, mask
            else:
                seed = np.random.randint(0, 2 ** 31 - 1)
                random.seed(seed)
                torch.cuda.manual_seed(seed)
                torch.manual_seed(seed)
                enhan_img = self.transform(enhan_img)
                random.seed(seed)
                torch.cuda.manual_seed(seed)
                torch.manual_seed(seed)
                mask = self.transform(mask)


                # plt.imshow(enhan_img[0,:,:], cmap='gray')
                # plt.imshow(mask[0,:,:], alpha=.5)
                # plt.show()
                return enhan_img, mask


    def __len__(self):
        if self.is_train and not self.inference:
            return len(self.imagename) * self.aug_rate
        else:
            return len(self.imagename)

def build_dataloader(is_train,args,batch_size,drop_last=False,shuffle=False,num_workers=8):

    dataset = CoroSeg_animal(is_train,args)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                             num_workers=num_workers, drop_last=drop_last)
    if is_train and not args.inference:
        print("number of train data:", len(dataset))
    elif not is_train and not args.inference:
        print("number of val data:", len(dataset))
    elif not is_train and args.inference:
        print("number of test data:", len(dataset))
    else:
        print("Wrong data mode!")
        exit(0)

    return data_loader
