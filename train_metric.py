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
from models.depth2pcd_cnn import CNN as CNN2

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
    model2 = CNN2().to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Define the loss function and optimizer
    criterion2 = nn.MSELoss()
    optimizer2 = optim.Adam(model2.parameters(), lr=3e-4)

    model.train()
    model2.train()

    for epoch in range(num_epochs):
        counter = 0
        for images, depths, pcds in tqdm(dataset, desc=f"Epoch : {epoch}"):
            # images: (6, 3, 1028, 1224)
            # depths: (6, 1, 1028, 1224)
            counter += 1

            # get upper and lower triplets
            upper_cams = np.array(images)[config.upper_cams] # (3, 3, 1028, 1224)
            lower_cams = np.array(images)[config.lower_cams] # (3, 3, 1028, 1224)

            # get upper and lower pcds, and projected depths
            try:
                upper_depth = np.array(depths)[config.upper_cams] # (3, 1, 1028, 1224)
                lower_depth = np.array(depths)[config.lower_cams] # (3, 1, 1028, 1224)
            except:
                continue

            # [UPPER] reshape inputs to (1, 9, 1028, 1224)
            upper_input = upper_cams.reshape(1, -1, config.image_size[0], config.image_size[1])
            upper_input = torch.Tensor(upper_input).to(device)
            upper_input = upper_input.repeat(3, 1, 1, 1) # (3, 9, 1028, 1224)

            dino_outputs = torch.Tensor(np.random.uniform(0, 1, (3, 1, 1028, 1224))).to(device) # (3, 1, 1028, 1224)
            upper_input = torch.cat((upper_input, dino_outputs), 1) # (3, 10, 1028, 1224)

            # [LOWER] reshape inputs to (1, 9, 1028, 1224)
            lower_input = lower_cams.reshape(1, -1, config.image_size[0], config.image_size[1])
            lower_input = torch.Tensor(lower_input).to(device)
            lower_input = lower_input.repeat(3, 1, 1, 1) # (3, 9, 1028, 1224)

            dino_outputs = torch.Tensor(np.random.uniform(0, 1, (3, 1, 1028, 1224))).to(device) # (3, 1, 1028, 1224)
            lower_input = torch.cat((lower_input, dino_outputs), 1) # (3, 10, 1028, 1224)

            # reshape inputs to (3, 1, 1028, 1224)
            upper_target = torch.Tensor(upper_depth).to(device)
            lower_target = torch.Tensor(lower_depth).to(device)

            # get output
            upper_output = model(upper_input)
            lower_output = model(lower_input)
            loss = criterion(upper_output, upper_target) + criterion(lower_output, lower_target)
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

            if counter % 10 == 0:
                print("Saving first.path ; ", counter)
                torch.save(model.state_dict(), 'first.pth')
            

            depths = torch.cat((upper_output, lower_output), 0) # (6, 1, 1028, 1224)



if __name__ == "__main__":
    main()