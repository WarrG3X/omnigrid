import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from pytorch3d.renderer.fisheyecameras import FishEyeCameras
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from tqdm import tqdm
import cv2

from datasets.fisheye_dataset import FisheyeDataset
from utils import config, train_utils
from models.cnn import CNN

def main():
    transform = transforms.Compose([
        # transforms.Resize((514,612)),
        transforms.ToTensor()
    ])

    dataset = FisheyeDataset('data/', 'data/lidar/', transform=transform)

    num_epochs = 10
    # initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    counter = 0
    for epoch in range(num_epochs):
        for images, pcds in tqdm(dataset, desc=f"Epoch : {epoch}"):
            # get pcds in camera frame
            pcds = train_utils.get_pcd_camera_frame(pcds)

            # get upper and lower triplets
            upper_cams = np.array(images)[config.upper_cams] # (3, 3, 1028, 1224)
            cv2.imwrite("feed.png", np.transpose(upper_cams[0, ...].astype(np.uint8) * 255, (1, 2, 0)))

            # get upper and lower pcds, and projected depths
            upper_pcd, lower_pcd = train_utils.split_pcd(pcds)

            H = torch.Tensor(train_utils.get_cam_frame())
            rvec = R.from_matrix(H[:3, :3]).as_euler('xyz')
            rvec = np.array([np.pi, 0, 0], dtype=np.float32)
            tvec = np.array([H[0,3], H[1,3], H[2,3]], dtype=np.float32)
            xi = config.xi
            K = np.array([  [config.fx, 0, config.cx],
                            [0, config.fy, config.cy],
                            [0, 0, 1]])
            D = np.array([config.k[0], config.k[1], 0, 0], dtype=np.float32)
            upper_depth = train_utils.pcd_to_depth(upper_pcd, rvec, tvec, K, D, xi) # (1028, 1224, 1)

            # reshape inputs to (1, 9, 1028, 1224)
            upper_input = upper_cams.reshape(1, -1, config.image_size[0], config.image_size[1])
            upper_input = torch.Tensor(upper_input).to(device)

            # reshape inputs to (1, 1, 1028, 1224)
            upper_target = np.transpose(upper_depth, (2, 0, 1))
            upper_target = torch.Tensor(upper_target).to(device).unsqueeze(0)

            # get output
            output = model(upper_input)
            loss = criterion(output, upper_target)
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


if __name__ == "__main__":
    main()