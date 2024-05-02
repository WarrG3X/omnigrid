import cv2
import numpy as np
import glob

from utils import config

def main():

    f = "/home/jaskaran/l3d_project/omnigrid/data/camera0/camera0_lp1_2190.png"

    K = np.array([  [config.fx, 0, config.cx],
                    [0, config.fy, config.cy],
                    [0, 0, 1]])
    D = np.array([config.k[0], config.k[1], config.tan[0], config.tan[1]], dtype=np.float32)

    distorted = cv2.imread(f)
    h, w = config.image_size
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

    img = cv2.fisheye.undistortImage(distorted, K, D, None, newcameramtx)
    print(roi)
    img = img[:-150, :, :]
    # img = img[roi[0]:roi[3], roi[1]:roi[2], :]

    cv2.imwrite("distorted.png", distorted[:-150, :, :])
    cv2.imwrite("undistorted.png", img)

if __name__ == "__main__":
    main()