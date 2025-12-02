import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

class CRSP20(Dataset):
    def __init__(self, data_path, split="train", target_col="Ret_20d"):
        """
        Args:
            data_path (str): folder path containing the .dat and label files
            split (str): 'train' (<=1999) or 'test' (>1999)
            target_col (str): objective column name in label files, e.g., 'Ret_20d' or 'Ret_60d'
        """
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.target_col = target_col
        
        self.IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
        self.IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96} 

        # sample metadata lists
        self.labels = []      
        self.dates = []       
        self.stock_ids = []   
        self.raw_rets = []    
        self.market_caps = [] # Absolute MarketCap values
        
        self.img_offsets = [] 

        files = sorted(os.listdir(data_path))
        
        total_images = 0
        print(f"Initializing dataset ({split}) for target: {target_col}...")

        for file in files:
            if len(file.split('_')) < 7: continue
            
            try:
                year = int(file.split('_')[6])
            except ValueError: continue
            if (split == "train" and year <= 1999) or (split == "test" and year > 1999):
                if file.endswith('.dat'):
                    print(f"Loading metadata from year: {year}")
                    img_path = os.path.join(data_path, file)
                    label_path = os.path.join(data_path, f'20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather')
                    try:
                        label_df = pd.read_feather(label_path)
                    except Exception as e:
                        print(f"Error reading {label_path}: {e}")
                        continue
                    if self.target_col not in label_df.columns:
                        print(f"Warning: Column '{self.target_col}' not found in {label_path}, skipping...")
                        continue
                    valid_indices = label_df[pd.notna(label_df[self.target_col])].index
                    total_images += len(valid_indices)
                    
                    valid_data = label_df.loc[valid_indices]
                    

                    self.labels.extend((valid_data[self.target_col] > 0).astype(int).tolist())
                    
                    self.dates.extend(valid_data.Date.astype(str).tolist())

                    self.stock_ids.extend(valid_data.StockID.tolist())

                    self.raw_rets.extend(valid_data[self.target_col].tolist())

                    if "MarketCap" in valid_data.columns:
                        self.market_caps.extend(valid_data.MarketCap.abs().tolist())
                    else:
                        self.market_caps.extend([0.0] * len(valid_data))
                    
                    self.img_offsets.append((img_path, valid_indices.copy()))

        memmap_filename = f'all_images_{split}_{self.target_col}.dat'
        H, W = self.IMAGE_HEIGHT[20], self.IMAGE_WIDTH[20]
    
        if os.path.exists(memmap_filename):
            print(f"Loading existing memmap file: {memmap_filename}")
            temp_memmap = np.memmap(memmap_filename, dtype=np.uint8, mode='r')
            if temp_memmap.size != total_images * H * W:
                print("Warning: Memmap size mismatch! Recreating...")
                os.remove(memmap_filename)
            else:
                self.imgs = temp_memmap.reshape((total_images, H, W))

        if not os.path.exists(memmap_filename):
            print(f"Creating new memmap file: {memmap_filename}")
            self.imgs = np.memmap(memmap_filename, dtype=np.uint8, mode='w+', shape=(total_images, H, W))
            
            start = 0
            print("Writing images to memmap...")
            for img_path, valid_indices in tqdm(self.img_offsets):
                try:
                    temp_img = np.memmap(img_path, dtype=np.uint8, mode='r').reshape((-1, H, W))
                    n = len(valid_indices)
                    self.imgs[start:start+n] = temp_img[valid_indices]
                    start += n
                except Exception as e:
                    print(f"Error processing image file {img_path}: {e}")
            self.imgs.flush()
            self.imgs = np.memmap(memmap_filename, dtype=np.uint8, mode='r', shape=(total_images, H, W))

        self.len = len(self.labels)
        print(f"Dataset initialized. Total samples: {self.len}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx], dtype=torch.float32).unsqueeze(0) 
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        date = self.dates[idx]
        stock_id = self.stock_ids[idx]
        
        raw_ret = torch.tensor(self.raw_rets[idx], dtype=torch.float32)
        market_cap = torch.tensor(self.market_caps[idx], dtype=torch.float32)
        
        return img, label, date, stock_id, raw_ret, market_cap