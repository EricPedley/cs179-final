from data_reading import SyncedDatum
import gtsam
import numpy as np
from visual_odometry import add_vo_factors, print_factor_errors
from data_reading import read_images, load_reference_poses, read_imu, sync_data, SyncedDatum
from pyLSHash import LSHash # https://github.com/guofei9987/pyLSHash/tree/main

X = lambda i: gtsam.symbol('x', i)
V = lambda i: gtsam.symbol('v', i)
B = lambda i: gtsam.symbol('b', i)

def init_preint_params(
        gravity_mag: float = 9.81,
        acc_sigma: float = 0.05,
        gyro_sigma: float = 0.0025,
        acc_bias_rw_sigma: float = 1e-4,
        gyro_bias_rw_sigma: float = 1e-5,
        z_up: bool = False
    ) -> gtsam.PreintegrationParams:

    params = gtsam.PreintegrationCombinedParams.MakeSharedU(gravity_mag)
    if z_up:           # world Z+ upward
        params.setGravityInertial(gtsam.Point3(0, 0, -gravity_mag))

    params.setAccelerometerCovariance(np.eye(3) * acc_sigma**2)
    params.setGyroscopeCovariance    (np.eye(3) * gyro_sigma**2)
    params.setIntegrationCovariance  (np.eye(3) * 1e-8)
    params.setBiasAccCovariance      (np.eye(3) * acc_bias_rw_sigma**2)
    params.setBiasOmegaCovariance    (np.eye(3) * gyro_bias_rw_sigma**2)
    return params


def init_imu_bias(acc_bias=np.zeros(3), gyro_bias=np.zeros(3)) -> gtsam.imuBias.ConstantBias:
    """
    acc_bias  - np.array shape (3,)  initial accel bias in m/s²
    gyro_bias - np.array shape (3,)  initial gyro bias in rad/s
    """
    return gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)

def _pose3_from_tuple(pose_tuple) -> gtsam.Pose3:
    """pose_tuple = (np.array([tx,ty,tz]), scipy Rotation)  →  gtsam.Pose3"""
    t, R = pose_tuple
    return gtsam.Pose3(gtsam.Rot3(R.as_matrix()),
                       gtsam.Point3(float(t[0]), float(t[1]), float(t[2])))

def add_gt_pose_nodes(
        graph: gtsam.NonlinearFactorGraph,
        synced_data: list[SyncedDatum],
        initial_values: gtsam.Values,
        prior_sigma: float = 1e-3,
        between_sigma: float = 1e-2
    ):
    """
    For every interval in `synced_data` insert poses X(i), X(i+1) taken
    directly from `gt_pose_start` and `gt_pose_end`, then add a
    BetweenFactorPose3 with small noise.  A prior on X(0) is also added
    (if it does not already exist).
    """
    # noise models ------------------------------------------------------------
    prior_noise   = gtsam.noiseModel.Diagonal.Sigmas(
                        np.array([prior_sigma]*6))     # roll,pitch,yaw,x,y,z
    between_noise = gtsam.noiseModel.Diagonal.Sigmas(
                        np.array([between_sigma]*6))

    # flag to add prior only once
    added_prior = False

    for k, datum in enumerate(synced_data):
        # ---- start pose X(k) -----------------------------------------------
        pose_k = _pose3_from_tuple(datum.gt_pose_start)

        if not initial_values.exists(X(k)):
            initial_values.insert(X(k), pose_k)

        # add a prior the very first time we touch X(0)
        if k == 0 and not added_prior:
            graph.add(gtsam.PriorFactorPose3(X(0), pose_k, prior_noise))
            added_prior = True

        # ---- end pose X(k+1) ------------------------------------------------
        pose_k1 = _pose3_from_tuple(datum.gt_pose_end)
        if not initial_values.exists(X(k+1)):
            initial_values.insert(X(k+1), pose_k1)

        # ---- between factor -------------------------------------------------
        # relative transform T_k→k+1 (world_T_k.inverse() * world_T_k1)
        T_rel = pose_k.between(pose_k1)
        graph.add(gtsam.BetweenFactorPose3(
                    X(k), X(k+1), T_rel, between_noise))



def add_imu_factors(
        graph: gtsam.NonlinearFactorGraph,
        data: list[SyncedDatum],
        initial_values: gtsam.Values,
        preint_params: gtsam.PreintegrationParams,
        bias_prior: gtsam.imuBias.ConstantBias
    ):
    """
    For every consecutive camera pose interval in `data`
    (data[i] goes from image i to image i+1) create a
    CombinedImuFactor that links:
        pose_i   -> pose_{i+1}
        vel_i    -> vel_{i+1}
        bias_i   -> bias_{i+1}
    and insert reasonable initial guesses for the new
    velocity / bias variables.

    Assumes:
        - camera pose keys are x0, x1, x2, ... (already in graph)
        - we use matching velocity/bias index numbers.
    """
    # 1. make sure we have priors for the very first V(0), B(0)
    if not initial_values.exists(V(0)):
        initial_values.insert(V(0), np.zeros(3))
    if not initial_values.exists(B(0)):
        initial_values.insert(B(0), bias_prior)

    for k in range(len(data)):                 # interval k -> k+1
        imu_segment = data[k].imu_data         # (N × 7) block

        # 2. Pre-integrate this segment
        preint = gtsam.PreintegratedCombinedMeasurements(
                    preint_params, bias_prior)

        timestamps = imu_segment[:,0]
        for j in range(len(imu_segment)-1):
            t, gx,gy,gz, ax,ay,az = imu_segment[j]
            t_next = imu_segment[j+1,0]
            dt = float(t_next - t)
            preint.integrateMeasurement(
                np.array([ax,ay,az]),
                np.array([gx,gy,gz]),
                dt )

        # 3. create new initial guesses if not present
        idx_next = k+1
        if not initial_values.exists(V(idx_next)):
            initial_values.insert(V(idx_next), initial_values.atVector(V(k)))  # simple copy
        if not initial_values.exists(B(idx_next)):
            initial_values.insert(B(idx_next), bias_prior)

        # 4. add factor
        graph.add(gtsam.CombinedImuFactor(
                    X(k), V(k),
                    X(idx_next), V(idx_next),
                    B(k), B(idx_next),
                    preint))


if __name__ == '__main__':
    imgs = read_images()
    imu = read_imu()
    gt_poses = load_reference_poses()
    imgs_list = []
    for i, img in enumerate(imgs):
        if i>1000:
            break
        imgs_list.append(img)
    synced_data = sync_data(imu, imgs_list, gt_poses)
    initial_values = gtsam.Values()
    g = gtsam.NonlinearFactorGraph()
    keypoint_data = LSHash(
        hash_size=32,
        input_dim=256,
        num_hashtables=1,
    )
    initial_values.insert(X(0), gtsam.Pose3())

    # add local pose nodes using visual odomotry
    # add_vo_factors(g, synced_data[0], initial_values, (0,1), keypoint_data)
    # add_vo_factors(g, synced_data[1], initial_values, (1,2), keypoint_data)

    # add local pose factors using ground truth
    add_gt_pose_nodes(g, synced_data[:100], initial_values)
    
    # test imu preintegration
    preint_params = init_preint_params()
    preint_bias = init_imu_bias()
    add_imu_factors(g, synced_data[:100], initial_values, preint_params, preint_bias)

    g.print("Factor Graph:\n")
    initial_values.print("Initial Values:\n")
    print("initial error = {}".format(g.error(initial_values)))
    print_factor_errors(g, initial_values, top_n=20)

    params = gtsam.DoglegParams()
    params.setVerbosity("TERMINATION")
    optimizer = gtsam.DoglegOptimizer(g, initial_values, params)
    print("Optimizing:")
    result = optimizer.optimize()
    result.print("Final results:\n")
    print("final error = {}".format(g.error(result)))
