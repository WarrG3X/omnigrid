import cv2
import os
import sys
from pathlib import Path
import time

# General setup
run_id = "lp7"
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from utils import config

# Camera setup
camera_settings = {
    "camera0": (f"{run_id}_feed_path0", 'data/camera0/'),
    "camera1": (f"{run_id}_feed_path1", 'data/camera1/'),
    "camera2": (f"{run_id}_feed_path2", 'data/camera2/'),
    "camera3": (f"{run_id}_feed_path3", 'data/camera3/'),
    "camera4": (f"{run_id}_feed_path4", 'data/camera4/'),
    "camera5": (f"{run_id}_feed_path5", 'data/camera5/')
}

# # lp1
# start_minutes = 1
# start_seconds = 0  
# end_minutes = 9 
# end_seconds = 30

# # lp6
# start_minutes = 3
# start_seconds = 0
# end_minutes = 12
# end_seconds = 00

# lp7
start_minutes = 1
start_seconds = 30
end_minutes = 7
end_seconds = 00


# Parameters for sampling
# lidar_samples = 6175  # Number of LIDAR samples

frame_rate = 24.5  # FPS
start_time = start_minutes * 60 + start_seconds  # Convert start time to seconds
end_time = end_minutes * 60 + end_seconds  # Convert end time to seconds
start_index = int(frame_rate * start_time)  # Calculate the starting frame index
end_index = int(frame_rate * end_time)  # Calculate the ending frame index
print(f"Start index: {start_index}, End index: {end_index}")



for camera_name, (feed_path_attr, output_dir) in camera_settings.items():
    video_path = getattr(config, feed_path_attr)
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
        print(f"Error: Could not open video for {camera_name}.")
        continue

    # Calculate the initial sampling rate, then multiply by 4
    initial_sampling_rate = 2#max(frame_count // lidar_samples, 1)
    sampling_rate = initial_sampling_rate * 4
    print(f"Initial sampling rate: {initial_sampling_rate}, Final sampling rate: {sampling_rate}")
    # exit()

    # Re-open the video file to save sampled frames
    cap = cv2.VideoCapture(video_path)
    saved_frames = 0
    current_frame = 0

    print(f"Processing {camera_name} - The video has {frame_count} frames, starting from frame index {start_index} and ending at frame index {end_index}.")
    print(f"Sampling every {sampling_rate} frames.")
    print("Saving sampled frames...")

    if cap.isOpened():
        while current_frame < end_index:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame >= start_index and (current_frame - start_index) % sampling_rate == 0:
                # scaled_id = start_index // initial_sampling_rate + saved_frames * 4
                frame_filename = os.path.join(output_dir, f"{camera_name}_{run_id}_{current_frame:04d}.png")
                cv2.imwrite(frame_filename, frame)
                saved_frames += 1
            current_frame += 1
        cap.release()
    else:
        print(f"Error: Could not re-open video for {camera_name}.")
        continue
    print(f"Finished processing {camera_name} with {frame_count} frames.")
    print(f"Saved {saved_frames} sampled frames starting from frame index {start_index} and ending at frame index {end_index}")
