import os
from shutil import copy2

def list_files(directory, pattern):
    """List files in a directory filtered by a pattern."""
    return [f for f in os.listdir(directory) if pattern in f]

def find_closest_lidar_file(lidar_files, target_time):
    """Find the LIDAR file with the closest timestamp to the target time."""
    closest_file = None
    smallest_diff = float('inf')
    for file in lidar_files:
        parts = file.split('_')
        timestamp = int(parts[-1].split('.')[0])  # Get the last part and remove the file extension
        diff = abs(timestamp - target_time)
        if diff < smallest_diff:
            smallest_diff = diff
            closest_file = file
    return closest_file

# Parameters
# # lp7
# camera_dir = 'omnigrid/data/camera0'
# lidar_dir = 'lidar7/'
# output_dir = 'omnigrid/data/lidar'
# total_frames = 13350  # Total number of camera frames
# min_timestamp = 1679939737
# max_timestamp = 1679940281
# duration = max_timestamp - min_timestamp
# lidar_file_pattern = 'lp7'  # Adjust this if your lidar pattern changes
# camera_file_pattern = 'lp7'  # Adjust this if your camera pattern changes
# output_extension = 'pcd'  # Change according to your LIDAR file extensions

# # lp6
# camera_dir = 'omnigrid/data/camera0'
# lidar_dir = 'lidar6/'
# output_dir = 'omnigrid/data/lidar'
# total_frames = 19686   # Total number of camera frames
# min_timestamp = 1679939746
# max_timestamp = 1679940549
# duration = max_timestamp - min_timestamp
# lidar_file_pattern = 'lp6'  # Adjust this if your lidar pattern changes
# camera_file_pattern = 'lp6'  # Adjust this if your camera pattern changes
# output_extension = 'pcd'  # Change according to your LIDAR file extensions


# lp1
camera_dir = 'omnigrid/data/camera0'
lidar_dir = 'lidar1/'
output_dir = 'omnigrid/data/lidar'
total_frames = 15290   # Total number of camera frames
min_timestamp = 1679941159
max_timestamp = 1679941782
duration = max_timestamp - min_timestamp
lidar_file_pattern = 'lp1'  # Adjust this if your lidar pattern changes
camera_file_pattern = 'lp1'  # Adjust this if your camera pattern changes
output_extension = 'pcd'  # Change according to your LIDAR file extensions




# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# List LIDAR files
lidar_files = list_files(lidar_dir, lidar_file_pattern)

# Process each camera file
camera_files = list_files(camera_dir, camera_file_pattern)
count = 0
for camera_file in camera_files:
    # Extract frame number
    parts = camera_file.split('_')
    frame_id = int(parts[2].split('.')[0])

    # Scale frame number to LIDAR timestamp
    scaled_time = min_timestamp + int((frame_id / total_frames) * duration)

    # Find closest LIDAR file
    closest_lidar = find_closest_lidar_file(lidar_files, scaled_time)

    # Copy and rename LIDAR file to output directory
    original_name = camera_file.split('.')[0]  # Remove the file extension
    output_filename = f"lidar_{original_name}.{output_extension}"
    copy2(os.path.join(lidar_dir, closest_lidar), os.path.join(output_dir, output_filename))

    # Print progress
    count += 1
    print(f'Processed {count}/{len(camera_files)} frames.', end='\r')

print("Dataset preprocessing complete.")
