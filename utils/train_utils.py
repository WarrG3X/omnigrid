import math
import numpy as np
import torch

from . import config
import cv2

from pytorch3d.renderer.fisheyecameras import FishEyeCameras
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# H_A_wrt_B
H_FLU_Lidar = np.array([[0, 1, 0, 0],
                        [-1, 0, 0, 0.116655],
                        [0, 0, 1, -0.082],
                        [0., 0., 0., 1. ]])
H_Yao_FLU = np.array([  [0, -1, 0, 0],
                        [-1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
H_IMU_camera = np.array([[-0.00729687, -0.99951677, -0.03021566, -0.16152484],
                        [0.99990183, -0.00765447, 0.01173613, 0.26446004],
                        [-0.01196174, -0.03012706, 0.9994745, -0.05705898],
                        [0., 0., 0., 1. ]])

def split_pcd(pcd):
    upper_pcd = pcd[pcd[:, 2] > 0]
    lower_pcd = pcd[pcd[:, 2] < 0]
    return upper_pcd, lower_pcd

def get_cam_frame():
    H_camera_IMU = np.linalg.inv(H_IMU_camera)
    H_camera_Lidar = H_camera_IMU @ H_Yao_FLU @ H_FLU_Lidar
    return H_camera_Lidar

def get_pcd_camera_frame(pcds):
    H_camera_IMU = np.linalg.inv(H_IMU_camera)
    H_camera_Lidar = H_camera_IMU @ H_Yao_FLU @ H_FLU_Lidar

    ones = np.ones((pcds.shape[0], 1))
    pcds = np.concatenate((pcds, ones), axis=1)

    pcds = np.transpose(H_camera_Lidar @ np.transpose(pcds))[:, :3]

    pcds = np.random.uniform(low=-20, high=20, size=(10000,3))

    return pcds

def isValid(u, v):
    if (0 <= u <= config.image_size[0] - 1) and (0 <= v <= config.image_size[1] - 1):
        return True
    return False
    
def get_pix_in_range(pix, pcd):
    pcd = pcd[pix[:, 0] > 0]
    pix = pix[pix[:, 0] > 0]

    pcd = pcd[pix[:, 1] > 0]
    pix = pix[pix[:, 1] > 0]

    pcd = pcd[pix[:, 0] < config.image_size[0]]
    pix = pix[pix[:, 0] < config.image_size[0]]

    pcd = pcd[pix[:, 1] < config.image_size[1]]
    pix = pix[pix[:, 1] < config.image_size[1]]

    return pix, pcd

def pcd_to_depth(pcd, rvec, tvec, K, D, xi):
    """
    Input: pcd (N, 3)
    Output: 
    """
    pcd_in = pcd
    depth_image = np.zeros((config.image_size[0], config.image_size[1], 1))

    pcd = pcd.reshape(-1, 1, 3).astype(np.float32)
    pix, jac = cv2.fisheye.projectPoints(pcd,
                                        rvec=rvec,
                                        tvec=tvec,
                                        K=K.astype(np.float32),
                                        D=D.astype(np.float32),
                                        alpha=xi)
    pix = np.round(pix.reshape(-1,2))
    pix = pix[:, [1, 0]]
    pix, pcd = get_pix_in_range(pix, pcd.reshape(-1,3))
    pix = pix.astype(int)
    print(np.min(pcd[:, 0]), np.min(pcd[:, 1]), np.min(pcd[:, 2]))
    print(np.max(pcd[:, 0]), np.max(pcd[:, 1]), np.max(pcd[:, 2]))

    for i, p in enumerate(pix):
        depth_image[p[0], p[1], 0] = pcd[i, 2] * 10

    cv2.imwrite("image.png", depth_image)
    print(">> Image saved")
    exit()

    return depth_image