import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
from pathlib import Path

def load_reference_poses():
    reference_poses = []
    with open('/home/forge/Desktop/cs179-final/data/recording_2020-03-24_17-36-22_reference_poses/recording_2020-03-24_17-36-22/result.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # skip empty lines or comments
            parts = line.split()

            timestamp = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])

            # Parse quaternion to Rotation object
            rotation = R.from_quat([qx, qy, qz, qw])

            reference_poses.append((timestamp, np.array([tx,ty,tz]), rotation))

    return reference_poses

def read_imu() -> pd.DataFrame:
    """
    Reads IMU data from imu.txt and returns a DataFrame.
    Columns: timestamp, gx, gy, gz, ax, ay, az
    """
    cols = ['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']
    df = pd.read_csv('/home/forge/Desktop/cs179-final/data/recording_2020-03-24_17-36-22_imu_gnss/recording_2020-03-24_17-36-22/imu.txt', delim_whitespace=True, names=cols)
    df['timestamp'] = df['timestamp'].astype(float) / 1e9  # Convert microseconds to seconds
    return df.to_numpy()

def read_images() -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Yields (timestamp, img_left, img_right) tuples for each stereo pair.
    Images are loaded as PIL Images.
    """
    ret = []
    for i, f in enumerate(Path('/home/forge/Desktop/cs179-final/data/recording_2020-03-24_17-36-22_stereo_images_undistorted/recording_2020-03-24_17-36-22/undistorted_images/cam0').glob('*.png')):
        if i>10:
            break
        timestamp = int(f.stem) / 1000000  # Convert microseconds to seconds
        img_left = cv2.imread(str(f), cv2.IMREAD_COLOR)
        img_right = cv2.imread(str(f.parent.parent / 'cam1' / f.name), cv2.IMREAD_COLOR)

        if img_left is None or img_right is None:
            continue
        ret.append((timestamp, img_left, img_right))
    return ret
