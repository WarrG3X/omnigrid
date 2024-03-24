# omnigrid
Course project for CMU 16-825 -  Occupancy Grid prediction using Multi-View Fisheye cameras

## Preprocessing
### ros2bag setup
These instructions are for processing the bag files in ROS2. If not needed, these steps can be skipped.

```bash
sudo apt-get install ros-$ROS_DISTRO-rosbag2
```

Next, get the `rosbags` python library by installing them from `requirements.txt` or by running the following command:

```bash
pip install rosbags
pip install rosbags-image
```

For future reference, the `rosbags` library can be found [here](https://pypi.org/project/rosbags/). This library is convenient as we don't need to setup a ROS2 environment to process the bag files.


### Usage
First convert the ROS1 bag files to ROS2 using the following command:

```bash
rosbags-convert lidar_points-007.bag
```

This will generate a folder with a sqlite database and metadata. Copy the folder to the `data` directory (create if it doesn't exist) and update the path in config.py.

Then run the following script to extract the lidar data.

```bash
# From the root directory
mkdir -p data/lidar
python scripts/lidar_extractor.py
```

This will take a couple of minutes and will extract the lidar data to the `data` directory.