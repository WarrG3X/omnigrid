import cv2
import os
import numpy as np

# Base directory for the images
base_dir = 'data_selected'

# Camera and depth indices
upper_cams = [0, 4, 5]
lower_cams = [1, 2, 3]
all_cams = upper_cams + lower_cams

# Get a list of files in the first camera directory to determine the available frames
sample_files = os.listdir(os.path.join(base_dir, 'scamera0'))
frame_indices = [f.replace('frame_', '').replace('.png', '') for f in sample_files if 'frame_' in f]

def load_image_pair(cam_index, frame_index):
    """ Load an image and its corresponding depth image based on index """
    img_path = os.path.join(base_dir, f'scamera{cam_index}', f'frame_{frame_index}.png')
    depth_path = os.path.join(base_dir, f'depth{cam_index}', f'depth_frame_{frame_index}.png')
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_COLOR)

    # Resize the images to 1/4th of the original size
    if img is not None:
        img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
    if depth is not None:
        depth = cv2.resize(depth, (depth.shape[1] // 4, depth.shape[0] // 4))
    
    return img, depth

def display_images(frame_index):
    """ Display images in a 4x3 grid for the given frame index """
    pairs = []
    for cam in all_cams:
        img, depth = load_image_pair(cam, frame_index)
        # Stack images vertically (camera on top, depth below)
        pair = np.vstack((img, depth))
        pairs.append(pair)
    
    # Combine pairs into a 4x3 grid (2 rows of 3 pairs)
    row_1 = np.hstack(pairs[:3])  # Upper cams and their depths
    row_2 = np.hstack(pairs[3:])  # Lower cams and their depths
    full_grid = np.vstack((row_1, row_2))
    
    cv2.imshow('Camera and Depth Viewer', full_grid)

current_frame = 0
display_images(frame_indices[current_frame])
print(len(frame_indices))

while True:
    print(f'Frame {frame_indices[current_frame]}')
    key = cv2.waitKey(0) & 0xFF
    print("key", key)
    if key == ord('q'):
        break
    elif key in [81, 37]:
        current_frame = (current_frame - 1) % len(frame_indices)
        print(current_frame)
        display_images(frame_indices[current_frame])
    elif key in [83, 39]:
        current_frame = (current_frame + 1) % len(frame_indices)
        print(current_frame)
        display_images(frame_indices[current_frame])
    else:
        print('Invalid key pressed.')

cv2.destroyAllWindows()
