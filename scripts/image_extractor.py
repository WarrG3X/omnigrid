import cv2
import os
import sys
from pathlib import Path
import time

output_dir = 'data/camera1/'
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import config

# Path to .qt video file
video_path = config.feed_path

cap = cv2.VideoCapture(video_path)

frame_count = 0
t0 = time.time()
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Extracting frames...")
    while True:
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Extracted {frame_count} frames.")
        else:
            break

cap.release()

print(f"Extraction completed in {time.time() - t0:.2f} seconds. {frame_count} frames were saved.")
