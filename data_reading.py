import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
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
            rotation = Rotation.from_quat([qx, qy, qz, qw])

            reference_poses.append((timestamp, np.array([tx,ty,tz]), rotation))

    return reference_poses

def read_imu() -> np.ndarray:
    """
    Reads IMU data from imu.txt and returns a DataFrame.
    Columns: timestamp, gx, gy, gz, ax, ay, az
    """
    cols = ['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']
    df = pd.read_csv('/home/forge/Desktop/cs179-final/data/recording_2020-03-24_17-36-22_imu_gnss/recording_2020-03-24_17-36-22/imu.txt', delim_whitespace=True, names=cols)
    df['timestamp'] = df['timestamp'].astype(float) / 1e9  # Convert nanoseconds to seconds
    df = df.sort_values('timestamp')

    return df.to_numpy()

def read_images() -> list[tuple[float, np.ndarray, np.ndarray]]:
    """
    Yields (timestamp, img_left, img_right) tuples for each stereo pair.
    Images are loaded as PIL Images.
    """
    ret = []
    for i, f in enumerate(sorted(list(Path('/home/forge/Desktop/cs179-final/data/recording_2020-03-24_17-36-22_stereo_images_undistorted/recording_2020-03-24_17-36-22/undistorted_images/cam0').glob('*.png')))):
        timestamp = float(f.stem) / 1e9  # Convert nanoseconds to seconds
        img_left = cv2.imread(str(f), cv2.IMREAD_COLOR)
        img_right = cv2.imread(str(f.parent.parent / 'cam1' / f.name), cv2.IMREAD_COLOR)

        if img_left is None or img_right is None:
            continue
        yield (timestamp, img_left, img_right)

def sync_data(imu_data: np.ndarray, images: list[tuple[int, np.ndarray, np.ndarray]], gt_poses: list[tuple[float,np.ndarray, Rotation]]):
    '''
    Collapses the data into intervals between image timestamps.
    Returns a list of tuples (
        start_time (float),
        end_time (float),
        imu_data (2D array with columns: timestamp, gx, gy, gz, ax, ay, az),
        img_left_start (numpy array),
        img_right_start (numpy array),
        img_left_end (numpy array),
        img_right_end (numpy array),
        gt_pose_start (tuple with (ndarray, Rotation)),
        gt_pose_end (tuple with (ndarray, Rotation)),
    )
    ground truth poses are interpolated to the start and end times.
    imu_data contains all IMU data between start and end times.
    '''

    synced_data = []
    imu_data = imu_data
    for i in range(len(images) - 1):
        start_time, img_left_start, img_right_start = images[i]
        end_time, img_left_end, img_right_end = images[i + 1]

        # Get IMU data between start and end times
        mask = (imu_data[:, 0] >= start_time) & (imu_data[:, 0] <= end_time)
        imu_segment = imu_data[mask]

        # Interpolate ground truth poses by taking poses before/after start/end times and doing linear and slerp
        gt_pose_before_start_index = next((i for i in reversed(range(len(gt_poses))) if gt_poses[i][0] <= start_time), None)
        gt_pose_before_start = gt_poses[gt_pose_before_start_index] if gt_pose_before_start_index is not None else None
        gt_pose_after_start = gt_poses[gt_pose_before_start_index + 1] if gt_pose_before_start_index is not None and gt_pose_before_start_index + 1 < len(gt_poses) else None
        gt_pose_before_end_index = next((i for i in reversed(range(len(gt_poses))) if gt_poses[i][0] <= end_time), None)
        gt_pose_before_end = gt_poses[gt_pose_before_end_index] if gt_pose_before_end_index is not None else None
        gt_pose_after_end = gt_poses[gt_pose_before_end_index + 1] if gt_pose_before_end_index is not None and gt_pose_before_end_index + 1 < len(gt_poses) else None
        
        
        dist_to_before_start = start_time - gt_pose_before_start[0] if gt_pose_before_start else float('inf')
        dist_to_after_start = gt_pose_after_start[0] - start_time if gt_pose_after_start else float('inf')
        dist_to_before_end = start_time - gt_pose_before_end[0] if gt_pose_before_end else float('inf')
        dist_to_after_end = gt_pose_after_end[0] - end_time if gt_pose_after_end else float('inf')

        proportion_start = dist_to_before_start / (dist_to_before_start + dist_to_after_start) if (dist_to_before_start + dist_to_after_start) > 0 else 0
        proportion_end = dist_to_before_end / (dist_to_before_end + dist_to_after_end) if (dist_to_before_end + dist_to_after_end) > 0 else 0

        # WARNING: This is likely to be wrong from vibe coding
        # print(start_time, end_time)
        gt_pose_start = (
            gt_pose_before_start[1] * (1 - proportion_start) + gt_pose_after_start[1] * proportion_start,
            Slerp([gt_pose_before_start[0], gt_pose_after_start[0]], Rotation.concatenate([gt_pose_before_start[2], gt_pose_after_start[2]]))([start_time])[0]
        ) if gt_pose_before_start and gt_pose_after_start else (None, None)

        gt_pose_end = (
            gt_pose_before_end[1] * (1 - proportion_end) + gt_pose_after_end[1] * proportion_end,
            Slerp([gt_pose_before_end[0], gt_pose_after_end[0]], Rotation.concatenate([gt_pose_before_end[2], gt_pose_after_end[2]]))([end_time])[0]
        ) if gt_pose_before_end and gt_pose_after_end else (None, None)

        if gt_pose_start[0] is None or gt_pose_end[0] is None:
            print(f"Warning: Missing ground truth pose for interval {start_time} to {end_time}. Skipping this interval.")
            continue

        synced_data.append((
            start_time,
            end_time,
            imu_segment,
            img_left_start,
            img_right_start,
            img_left_end,
            img_right_end,
            gt_pose_start[1:],
            gt_pose_end[1:],
        ))
    return synced_data