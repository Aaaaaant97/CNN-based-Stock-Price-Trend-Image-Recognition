import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2


# class CRSP20(Dataset):
#     def __init__(self, data_path, split = "train"):
#         super().__init__()
#         self.data_path = data_path
#         self.split = split
#         self.IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
#         self.IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96} 

#         images = []
#         labels = []
#         for file in os.listdir(data_path):
#             year = int(file.split('_')[6])
#             file_img_path = os.path.join(data_path, file)
#             file_label_path = os.path.join(data_path, f'20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather')
#             if split == "train" and year <= 1999:
#                 if file.endswith('.dat'):
#                     print(f"loading dataset, year = {year}")
#                     images, labels = self.parse_data(file_img_path, file_label_path, images, labels)
#             if split == "test" and year > 1999:
#                 if file.endswith('.dat'):
#                     print(f"loading dataset, year = {year}")
#                     images, labels = self.parse_data(file_img_path, file_label_path, images, labels)

#         images = np.concatenate(images)

#         self.imgs = torch.Tensor(images.copy())
#         self.labels = torch.Tensor(labels)
#         self.len = len(self.imgs)
    
#     def parse_data(self, file_img_path, file_label_path, images, labels):
#         img = np.memmap(file_img_path, dtype=np.uint8, mode='r').reshape((-1, self.IMAGE_HEIGHT[20], self.IMAGE_WIDTH[20]))
#         label = pd.read_feather(file_label_path)
#         # drop Ret_20d = nan
#         valid_indices = label[pd.notna(label["Ret_20d"])].index 
#         label_valid = label.loc[valid_indices]
#         label_valid = label_valid.Ret_20d > 0
#         img_valid = img[valid_indices]
#         images.append(img_valid)
#         labels.extend(label_valid.tolist())
#         return images, labels
  
#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):
#         return self.imgs[idx], self.labels[idx]


class CRSP20(Dataset):
    def __init__(self, data_path, split="train"):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
        self.IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96} 

        self.labels = []
        self.img_offsets = []  # 记录每个文件在 memmap 中的起始索引

        # 首先统计总图片数
        total_images = 0
        for file in os.listdir(data_path):
            year = int(file.split('_')[6])
            if split == "train" and year <= 1999 or split == "test" and year > 1999:
                if file.endswith('.dat'):
                    print(f"loading dataset, year = {year}")
                    img_path = os.path.join(data_path, file)
                    img = np.memmap(img_path, dtype=np.uint8, mode='r').reshape((-1, self.IMAGE_HEIGHT[20], self.IMAGE_WIDTH[20]))
                    label_path = os.path.join(data_path, f'20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather')
                    label = pd.read_feather(label_path)
                    valid_indices = label[pd.notna(label["Ret_5d"])].index
                    total_images += len(valid_indices)
                    self.labels.extend((label.loc[valid_indices].Ret_5d > 0).tolist())
                    self.img_offsets.append((img_path, valid_indices.copy()))

        # 创建 memmap 文件
        H, W = self.IMAGE_HEIGHT[20], self.IMAGE_WIDTH[20]
        self.imgs = np.memmap('all_images.dat', dtype=np.uint8, mode='w+', shape=(total_images, H, W))

        # 把数据写入 memmap
        start = 0
        for img_path, valid_indices in tqdm(self.img_offsets):
            img = np.memmap(img_path, dtype=np.uint8, mode='r').reshape((-1, H, W))
            n = len(valid_indices)
            self.imgs[start:start+n] = img[valid_indices]
            start += n
        self.len = len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label