import os
import shutil

def copy_every_nth_image(source_dir, target_dir, n):
    # Create the target directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # Get all files in the source directory and sort them to maintain order
    files = sorted(os.listdir(source_dir))
    
    # Iterate through files and copy every nth file
    for i, file in enumerate(files):
        if (i + 1) % n == 0:  # +1 to start counting from 1 instead of 0
            source_file = os.path.join(source_dir, file)
            target_file = os.path.join(target_dir, file)
            shutil.copy2(source_file, target_file)
            print(f"Copied {source_file} to {target_file}")

# Usage
source_directory = 'data/camera5'
target_directory = 'data/scamera5'
nth_image = 100

copy_every_nth_image(source_directory, target_directory, nth_image)
