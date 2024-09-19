from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset

from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize, Resize_3d, Resize_abs, Crop

class Train_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        print(args.dataset_path)
        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))
        print(len(self.filename_list))
        self.transforms = Compose([
                Resize_abs(256, 512, 512),
                Crop((127, 255), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                Resize_3d(1/2, 1/4, 1/8)
            ])
        self.transforms_a_128 = Compose([
                Resize_abs(256, 512, 512),
                Crop((127, 255), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                Resize_3d(1, 1/2, 1/4)
            ])
        self.transforms_f = Compose([
                Resize_abs(256, 512, 512),
                Crop((0, 128), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                Resize_3d(1/2, 1/4, 1/8)
            ])
        self.transforms_f_128 = Compose([
                Resize_abs(256, 512, 512),
                Crop((0, 128), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                Resize_3d(1, 1/2, 1/4)
            ])
        self.transforms_e_128 = Compose([
                Resize_abs(256, 512, 512),
                Crop((0, 128), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                Resize_3d(1, 1/2, 1/4)
            ])
        self.transforms_f_128 = Compose([
                Resize_abs(256, 512, 512),
                Crop((0, 128), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                Resize_3d(1, 1/2, 1/4)
            ])
        self.transforms_e_64 = Compose([
                Resize_abs(256, 512, 512),
                Crop((0, 128), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                Resize_3d(1/2, 1/4, 1/8)
            ])
        self.transforms_b_64 = Compose([
                Resize_abs(256, 512, 512),
                Crop((64, 192), (100, 356), (0, 512)),
                RandomFlip_LR(prob=0.5),
                Resize_3d(1/2, 1/4, 1/8)
            ])
        self.transforms_b_128 = Compose([
                Resize_abs(256, 512, 512),
                Crop((64, 192), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                Resize_3d(1, 1/2, 1/4)
            ])
        self.transforms_b_128_single_right = Compose([
                Resize_abs(256, 512, 512),
                Crop((64, 192), (0, 256), (0, 256)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                Resize_3d(1, 1/2, 1/2)
            ])
        self.transforms_b_128_single_left = Compose([
                Resize_abs(256, 512, 512),
                Crop((64, 192), (0, 256), (256, 512)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                Resize_3d(1, 1/2, 1/2)
            ])
        self.transforms_c_64 = Compose([
                Resize_abs(256, 512, 512),
                Crop((64, 192), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                Resize_3d(1/2, 1/4, 1/8)
            ])
        self.transforms_c_128 = Compose([
                Resize_abs(256, 512, 512),
                Crop((64, 192), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                Resize_3d(1, 1/2, 1/4)
            ])
        self.transforms_d_64 = Compose([
                Resize_abs(256, 512, 512),
                Crop((64, 192), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                Resize_3d(1/2, 1/4, 1/8)
            ])
        self.transforms_d_128 = Compose([
                Resize_abs(256, 512, 512),
                Crop((64, 192), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                Resize_3d(1, 1/2, 1/4)
            ])
        self.transforms_g = Compose([
                Resize_abs(256, 512, 512),
                Crop((0, 128), (100, 356), (0, 512)),
                RandomFlip_LR(prob=0.5),
                Resize_3d(1/2, 1/4, 1/8)
            ])
        self.transforms_g_128 = Compose([
                Resize_abs(256, 512, 512),
                Crop((0, 128), (0, 256), (0, 512)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                Resize_3d(1, 1/2, 1/4)
            ])
        self.transforms_skull = Compose([
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                Resize_3d(1/8, 1/8, 1/8)
            ])

    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkUInt8)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        seg_array = seg_array - ct_array
        seg_array = abs(seg_array)
        seg_array[seg_array > 1] = 0

        ct_array = ct_array.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        ct_array,seg_array = self.transforms_a_128(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
    from config import args
    train_ds = Train_Dataset(args)
    print(args.dataset_path)
    train_dl = DataLoader(train_ds, 1, False, num_workers=1)
    print(len(train_dl))
    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())
