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
    def __init__(self, img_dir, pcd_dir, transform=None):
        self.img_dir = img_dir
        self.pcd_dir = pcd_dir
        self.transform = transform

        # Finding directories for camera0 to camera5
        self.img_dirs = [os.path.join(img_dir, f'camera{i}') for i in range(6)]
        for dir in self.img_dirs:
            if not os.path.exists(dir):
                raise ValueError(f"Directory {dir} does not exist.")
        
        self.pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
        if len(self.pcd_files) == 0:
            raise ValueError(f"No point clouds found in {pcd_dir}.")

        # Verify that each camera directory has the same number of images as there are PCD files
        self.img_files = [sorted(glob.glob(os.path.join(dir, '*.png'))) for dir in self.img_dirs]
        num_images = [len(files) for files in self.img_files]
        if any(num != len(self.pcd_files) for num in num_images):
            raise ValueError("The number of images in each camera directory must match the number of PCD files.")

        self.len = len(self.pcd_files)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Load images from each camera directory for the current index
        imgs = [Image.open(self.img_files[i][idx]) for i in range(6)]
        pcd_path = self.pcd_files[idx]
        pcd = load_simple_pcd(pcd_path)

        # Apply transform to all images if transform is not None
        if self.transform:
            imgs = [self.transform(img) for img in imgs]

        return imgs, pcd