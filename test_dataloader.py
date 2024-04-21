import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import open3d as o3d

from datasets.fisheye_dataset import FisheyeDataset


transform = transforms.Compose([
    transforms.Resize((514,612)),
    transforms.ToTensor()
])


dataset = FisheyeDataset('data/', 'data/lidar/', transform=transform)


count = 0
for images, pcds in dataset:
    # Show image
    fig, axs = plt.subplots(2, 3, figsize=(15, 10),facecolor='black')

    for i, img in enumerate(images):
        ax = axs[i//3, i%3] 
        img = img.permute(1, 2, 0)
        ax.imshow(img)
        ax.axis('off')
    # plt.show()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
    plt.savefig(f"dataloader_{count}.png", bbox_inches='tight',pad_inches=0)

    # # Visualize point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcds)
    # # o3d.visualization.draw_geometries([pcd])

    if count == 2:
        break
    count += 1