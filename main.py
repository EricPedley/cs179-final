from data_reading import read_images, load_reference_poses, read_imu, sync_data
from visualization import visualize_trajectory
from visual_odometry import compute_relative_pose, compute_matches, add_vo_factors, cam0_to_cam1, print_factor_errors
from imu_preintegration import add_imu_factors
import numpy as np
import cv2
import gtsam
from pyLSHash import LSHash
from gtsam import DoglegOptimizer, Values, symbol_shorthand
from imu_preintegration import init_preint_params, init_imu_bias, add_gt_pose_nodes
L = symbol_shorthand.L
X = symbol_shorthand.X
Y = symbol_shorthand.Y

imgs = read_images()
imu = read_imu()
gt_poses = load_reference_poses()
imgs_list = []
for i, img in enumerate(imgs):
    if i>1000:
        break
    imgs_list.append(img)
synced_data = sync_data(imu, imgs_list, gt_poses)


def create_graph():
    #Noise on prior pose - pretty confident
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
    )
    initial_estimates = Values()
    graph = gtsam.NonlinearFactorGraph()

    #Converting pose into Pose3 object that GTSAM can use
    pos = synced_data[0].gt_pose_start[0]
    rot = synced_data[0].gt_pose_start[1]
    R = rot.as_matrix()
    rot3 = gtsam.Rot3(R)
    point3 = gtsam.Point3(pos[0], pos[1], pos[2])
    pose3 = gtsam.Pose3(rot3, point3)
    #Using starting pose as PriorFactor
    graph.add(gtsam.PriorFactorPose3(X(0), pose3, PRIOR_NOISE))
    
    pose_count = 0
    
    keypoint_data = LSHash(
            hash_size=32,
            input_dim=256,
            num_hashtables=1,
    )
    initial_pose_ground_truth = gtsam.Pose3(gtsam.Rot3(synced_data[0].gt_pose_start[1].as_matrix()), synced_data[0].gt_pose_start[0])
    initial_estimates.insert(X(0), initial_pose_ground_truth)
    initial_estimates.insert(Y(0), initial_pose_ground_truth.compose(cam0_to_cam1))

    graph.push_back(gtsam.BetweenFactorPose3(
        X(0),
        Y(0),
        cam0_to_cam1,
        gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)  # 6DOF: very small noise
    ))
    #Iterate through synced data and call factor functions on each data point
    preint_params = init_preint_params()
    preint_bias = init_imu_bias()
    N_DATA = 20
    USE_GT_POSE = False # cheating a bit to help debugging
    if USE_GT_POSE:
        add_gt_pose_nodes(graph, synced_data[:N_DATA], initial_estimates)
    for data in synced_data[:N_DATA]:
        try:
            add_vo_factors(graph, data, initial_estimates, (pose_count,pose_count+1), keypoint_data, insert_initial_pose_values=not USE_GT_POSE)
            pose_count +=1
        except cv2.error as e:
            if "five-point.cpp" in str(e) and "npoints >= 0" in str(e):
                print(f"Skipping frame pair {pose_count}-{pose_count+1} due to insufficient correspondences.")
                continue
            else:
                raise 
    add_imu_factors(graph, synced_data[:N_DATA], initial_estimates, preint_params, preint_bias)
    # graph.print("Factor Graph:\n")
    #Run graph through Dogleg Optimizer 
    #print(initial_estimates)
    print("initial error = {}".format(graph.error(initial_estimates)))
    params = gtsam.DoglegParams()
    #Debugging statement
    params.setVerbosity("TERMINATION")
    optimizer = DoglegOptimizer(graph, initial_estimates, params)
    result = optimizer.optimize()
    # result.print("Final results:\n")
    print("final error = {}".format(graph.error(result)))
    print_factor_errors(graph, result, top_n=20)





def compare_error(estimates, real_values):
    min_length = min(len(estimates), len(real_values))
    pos_errors = []
    rot_errors = []
    for ind in range(min_length):
        real_pos, real_rot = real_values[ind].gt_pose_end
        e_pos, e_rot = estimates[ind].gt_pose_end
        pos_diff = real_pos - e_pos
        pos_error = np.linalg.norm(pos_diff)
        euler_real = real_rot.as_euler('XYZ', degrees=True)
        euler_estimate = e_rot.as_euler('XYZ', degrees=True)
        rot_error = np.linalg.norm(euler_real - euler_estimate)
        pos_errors.append(pos_error)
        rot_errors.append(rot_error)
    if min_length:
        avg_pos_error = sum(pos_errors) / len(pos_errors)
        avg_rot_error = sum(rot_errors) / len(rot_errors)
        return avg_pos_error, avg_rot_error
    return None, None




create_graph()

# %%



