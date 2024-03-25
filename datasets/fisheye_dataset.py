import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import glob

def load_simple_pcd(file_path):
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if line.startswith('DATA'):
                break
        points = np.loadtxt(f)
    return points


class FisheyeDataset(Dataset):
    def __init__(self, img_dir,pcd_dir, transform=None):
        self.img_dir = img_dir
        self.pcd_dir = pcd_dir
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        self.pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))


        if len(self.img_files) == 0:
            raise ValueError(f"No images found in {img_dir}.")
        if len(self.pcd_files) == 0:
            raise ValueError(f"No point clouds found in {pcd_dir}.")
        if len(self.img_files) != len(self.pcd_files):
            raise ValueError(f"Number of images and point clouds do not match.")

        self.len = len(self.img_files)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        pcd_path = self.pcd_files[idx]
        img = Image.open(img_path)
        pcd = load_simple_pcd(pcd_path)

        if self.transform:
            img = self.transform(img)

        return img, pcd