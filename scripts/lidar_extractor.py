from rosbags.highlevel import AnyReader
from pathlib import Path
import struct
import time
import sys

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import config

bag_path = config.lidar_bag_path
output_dir = 'data/lidar'
count = 0


def read_points(msg):
    """
    Deserialize point data from a PointCloud2 message.
    Assumes each point consists of 'x', 'y', 'z' fields of type float32.
    """
    field_names = [field.name for field in msg.fields]
    fmt = '>' if msg.is_bigendian else '<'  # Big endian or little endian
    fmt += 'fff'  # Assuming 'x', 'y', 'z' fields are all float32

    point_step = msg.point_step
    points = []

    for i in range(msg.width * msg.height):
        offset = i * point_step
        point_data = msg.data[offset:offset+point_step]
        x, y, z = struct.unpack_from(fmt, point_data)
        points.append((x, y, z))

    return points

def save_pcd(file_name, points, fields=('x', 'y', 'z')):
    with open(file_name, 'w') as f:
        # PCD file header
        f.write("VERSION .7\n")
        f.write("FIELDS " + " ".join(fields) + "\n")
        f.write("SIZE " + " ".join(['4'] * len(fields)) + "\n")  # Assuming all fields are float32
        f.write("TYPE " + " ".join(['F'] * len(fields)) + "\n")  # F for float32
        f.write("COUNT " + " ".join(['1'] * len(fields)) + "\n")
        f.write("WIDTH {}\n".format(len(points)))
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")  # Default viewpoint
        f.write("POINTS {}\n".format(len(points)))
        f.write("DATA ascii\n")
        for point in points:
            f.write(" ".join(map(str, point)) + "\n")


with AnyReader([Path(bag_path)]) as reader:
    # topic and msgtype information is available on .connections list
    print("Connections:")
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)
    
    t0 = time.time()
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/velodyne_points': # topic Name of PointCloud2
            msg = reader.deserialize(rawdata, connection.msgtype)
            sec = msg.header.stamp.sec
            points = read_points(msg)
            pcd_file_name = f"{output_dir}/lidar_{count:04d}_{sec}.pcd"
            save_pcd(pcd_file_name, points)
            count += 1
            if count % 100 == 0:
                print(f"Extracted {count} point clouds from the bag file.")


print(f"Finished extracting in {time.time() - t0:.2f} seconds. Total {count} point clouds extracted.")