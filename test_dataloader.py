import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import open3d as o3d

from datasets.fisheye_dataset import FisheyeDataset


transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = FisheyeDataset('data/camera1/', 'data/lidar/', transform=transform)


count = 2
for images, pcds in dataset:
    # Show image
    plt.imshow(images.permute(1, 2, 0))
    plt.show()

    # Visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcds)
    # o3d.visualization.draw_geometries([pcd])

    if count == 0:
        break
    count -= 1