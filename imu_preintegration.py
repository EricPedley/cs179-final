from data_reading import SyncedDatum
from gtsam import NonlinearFactorGraph, PreintegrationParams, noiseModel, Vector3

def add_imu_factors(
    g: NonlinearFactorGraph,
    data: SyncedDatum,
    variables: tuple[str,str]
    """creates factors for how likely are pose estimates based off of IMU readings"""
    # between pose reading there is a factor
    # how is a factor represented.
    # how is the factor function defined.

    # init preintergrator
    preint = PreintegrationParams
)