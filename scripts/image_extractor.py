import cv2
import os
import sys
from pathlib import Path
import time

output_dir = 'data/camera5/'
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import config

# Path to .qt video file
video_path = config.feed_path5

cap = cv2.VideoCapture(video_path)

frame_count = 0
if cap.isOpened():
    while True:
        ret, _ = cap.read()
        if ret:
            frame_count += 1
        else:
            break
    cap.release()
else:
    print("Error: Could not open video.")
    sys.exit()

# Parameters for sampling
lidar_samples = 7963 # Number of LIDAR samples
subset = 7963  # Number of frames to save


sampling_rate = max(frame_count // lidar_samples, 1)

# Re-open the video file to save sampled frames
cap = cv2.VideoCapture(video_path)
saved_frames = 0
current_frame = 0

print(f"The video has {frame_count} frames. The LIDAR data has {lidar_samples} samples.")
print(f"Sampling every {sampling_rate} frames.")
print("Saving sampled frames...")

if cap.isOpened():
    while saved_frames < subset:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % sampling_rate == 0:
            frame_filename = os.path.join(output_dir, f'frame_{saved_frames:04d}.png')
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
        current_frame += 1
    cap.release()
else:
    print("Error: Could not re-open video.")
    sys.exit()

print(f"Scaled {frame_count} frames to {lidar_samples} frames.")
print(f"Saved first {saved_frames} sampled frames")
