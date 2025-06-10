from data_reading import SyncedDatum
from gtsam import NonlinearFactorGraph, PreintegrationParams, CombinedImuFactor, symbol
import numpy as np

def add_imu_factors(
    g: NonlinearFactorGraph,
    data: SyncedDatum,
    variables: tuple[str,str]):
    """creates factors for how likely are pose estimates based off of IMU readings"""
    # between pose reading there is a factor
    # how is a factor represented.
    # how is the factor function defined.

    # init preintergrator
    preint = PreintegrationParams

    imu_data = data[2]  # Extract IMU data block
    for row in imu_data:
        t, gx, gy, gz, ax, ay, az = row
        acc = np.array([ax, ay, az])
        gyro = np.array([gx, gy, gz])
        dt = 0.005  # You can also compute this dynamically if timestamps are not uniform
        preint.integrateMeasurement(acc, gyro, dt)

    # Add the actual factor
    g.add(CombinedImuFactor(
        symbol('x', prev_key), symbol('v', prev_key),
        symbol('x', curr_key), symbol('v', curr_key),
        symbol('b', prev_key),
        preint
    ))
