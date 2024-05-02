"""
In the data folder, we have the following data:
1. lidar
2. depth images
    - for each frame in the camera folder, a corresponding depth image is present

Let the following be the configuration of the camera setup:
- camera 1, 2, 3 are the down cameras
- camera 0, 4, 5 are the up cameras

Training pipeline:
- for each frame, get the depth images from the camera folder
- for all the upper cameras (i.e. camera 0, 4, 5), get the depth images
    - concatenate these three depth images to form a 3 channel image
    - pass this 3 channel image through the CNN model 
    - the output of the model is a predicted point cloud
- for all the lower cameras (i.e. camera 1, 2, 3), get the depth images
    - concatenate these three depth images to form a 3 channel image
    - pass this 3 channel image through the CNN model 
    - the output of the model is a predicted point cloud
- concatenate the predicted point clouds from the upper and lower cameras
- get the ground truth point cloud from the lidar folder
- calculate the chamfer loss between the predicted point cloud and the ground truth point cloud
- backpropagate the loss and update the weights of the model
- repeat for all frames



shapes:

pcd: (29184, 3) ~ 14592 points in the upper half and 14592 points in the lower half
"""

import os
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch import nn, optim
from torchvision import transforms
from datasets.depth_dataset import DepthDataset
from datasets.fisheye_dataset import FisheyeDataset
from utils import config, train_utils
from models.depth2pcd_cnn import CNN
from pytorch3d.loss import chamfer

import train_metric

import ipdb
st = ipdb.set_trace


def chamfer_loss(pcd_pred, pcd_gt):
    # calculate the chamfer loss
    loss = chamfer.chamfer_distance(pcd_pred, pcd_gt)[0]
    return loss


def main():

    # st()

    # define the cameras
    upper_cameras = [0, 4, 5]
    lower_cameras = [1, 2, 3]

    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # set the dataset
    # dataset = DepthDataset('data_selected/', 'data/lidar/', transform=transform)
    dataset = FisheyeDataset('data/', 'data/lidar/', transform=transform)

    # set the model
    model = CNN().to(device)

    # might have change this to chamfer loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # set the number of epochs
    num_epochs = 10

    # set the counter
    counter = 0

    # iterate over the dataset

    for epoch in range(num_epochs):
        for images, pcd_gt in tqdm(dataset, desc=f"Epoch : {epoch}"):
            # counter += 1

            # if (counter % 10 != 0) or (counter < 3000):
            #     continue

            # get the ground truth point cloud
            # st()
            pcd_gt = torch.Tensor(pcd_gt).to(device).unsqueeze(0)
            
            # use the depth image from the upper cameras to predict the point cloud
            # upper_image = np.array(images)[upper_cameras]  # (3, 1, 1028, 1224)
            # upper_image = upper_image.reshape(1, -1, upper_image.shape[2], upper_image.shape[3]) # (1, 3, 224, 224)
            # upper_image = torch.Tensor(upper_image).to(device)

            #TODO: remove this later for actual depth images
            # upper_image = torch.randn(1, 3, 224, 224).to(device)
            upper_image = torch.randn(1, 3, 224, 224).to(device)

            # get the output
            upper_pcd_pred = model(upper_image)

            # lower_image = np.array(images)[lower_cameras]  # (3, 224, 224)
            # lower_image = lower_image.reshape(1, -1, upper_image.shape[2], upper_image.shape[3]) # (1, 3, 224, 224)
            # lower_image = torch.Tensor(lower_image).to(device)

            #TODO: remove this later for actual depth images
            lower_image = torch.randn(1, 3, 224, 224).to(device)

            lower_pcd_pred = model(lower_image)

            # concatenate the predicted point clouds
            pcd_pred = torch.cat((upper_pcd_pred, lower_pcd_pred), dim=1)

            # calculate the chamfer loss
            loss = chamfer_loss(pcd_pred, pcd_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            # print loss
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


    # save the model
    torch.save(model.state_dict(), 'models/depth2pcd_cnn.pth')
    print("Model saved successfully")



if __name__ == "__main__":
    main()