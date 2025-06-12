import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import os
import time
import pysuperansac
import sys
import cv2
import io
import requests
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from scipy.spatial.transform import Rotation
import gtsam
from gtsam import symbol_shorthand

from pyLSHash import LSHash # https://github.com/guofei9987/pyLSHash/tree/main
from data_reading import read_images, load_reference_poses, read_imu, sync_data, SyncedDatum


X = symbol_shorthand.X
Y = symbol_shorthand.Y
L = symbol_shorthand.L


CV_WORLD_FRAME_TO_WORLD_FRAME_MAT_ROTATION = np.array(
    [[0, 0, 1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32
)

FLIP_Z_MATRIX = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ]
)
CV_WORLD_FRAME_TO_WORLD_FRAME_MAT = (
    FLIP_Z_MATRIX @ CV_WORLD_FRAME_TO_WORLD_FRAME_MAT_ROTATION
)

CV_TO_WORLD_ROT = Rotation.from_matrix(CV_WORLD_FRAME_TO_WORLD_FRAME_MAT)



def cv_to_pose(rvec, tvec):
    """
    opencv camera rvec and tvec to drone pose in ardupilot frame, taking in to account hard-coded camera angle
    """
    r_cv = Rotation.from_rotvec(rvec)
    t_cv = tvec

    def inv(x):
        return x.inv()

    pose_r = inv(CV_TO_WORLD_ROT * r_cv * inv(CV_TO_WORLD_ROT))
    pose_t = -CV_WORLD_FRAME_TO_WORLD_FRAME_MAT @ r_cv.inv().as_matrix() @ t_cv
    return pose_t, pose_r


A = CV_TO_WORLD_ROT.inv()


def pose_to_cv(pose_t, pose_r):
    """
    drone pose in ardupilot frame to opencv camera rvec and tvec, taking into account hard-coded camera angle
    """

    rvec_rot = A * pose_r.inv() * CV_TO_WORLD_ROT
    tvec = -rvec_rot.apply(pose_t @ CV_WORLD_FRAME_TO_WORLD_FRAME_MAT)
    return rvec_rot.as_rotvec(), tvec


K = np.array([
    [501.4757919305817, 0, 421.7953735163109],
    [0, 501.4757919305817, 167.65799492501083],
    [0, 0, 1]
])
# focal length is 3.5mm physically

# Parameters
device = torch.device('cuda')

# Initialize the detector and matcher
detector = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features='superpoint').eval().to(device)  # load the matcher

config = pysuperansac.RANSACSettings()
config.inlier_threshold = 5.0
config.min_iterations = 1000
config.max_iterations = 50
config.confidence = 0.999
config.sampler = pysuperansac.SamplerType.PROSAC
config.scoring = pysuperansac.ScoringType.MAGSAC
config.local_optimization = pysuperansac.LocalOptimizationType.NestedRANSAC
config.final_optimization = pysuperansac.LocalOptimizationType.IteratedLSQ
config.neighborhood_settings.neighborhood_grid_density = 6
config.neighborhood_settings.neighborhood_size = 6


# TODO: refactor out the matching+RANSAC into a separate function that returns the matches with their descriptors
def compute_matches(img1, img2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Returns the matches from doing RANSAC on 
    essential matrix computation 
    between two images as (points0, points1, descriptors, E)
    '''
    torch_img1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
    torch_img2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
    feats0 = detector.extract(torch_img1.to(device))  # auto-resize the image, disable with resize=None
    feats1 = detector.extract(torch_img2.to(device))
    
    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)py(img2).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
    splg_matches = np.concatenate([points0.cpu().detach().numpy(), points1.cpu().detach().numpy()], axis=1), matches01["scores"].cpu().detach().numpy()

    # Order by the score
    splg_matches = splg_matches[0][np.argsort(splg_matches[1])[::-1]], np.sort(splg_matches[1])[::-1]

    E, inliers, score, iterations = pysuperansac.estimateEssentialMatrix(
        np.ascontiguousarray(splg_matches[0]), 
        K, 
        K,
        [],
        [img1.shape[2], img1.shape[1], img2.shape[2], img2.shape[1]],
        config = config)

    mask = np.zeros((splg_matches[0].shape[0], 1), dtype=np.uint8)
    mask[inliers] = 1
    print(len(inliers), "inliers found out of", splg_matches[0].shape[0], "matches")
    
    points0 = splg_matches[0][inliers, :2]
    points1 = splg_matches[0][inliers, 2:]
    descriptors = torch.concatenate([feats0['descriptors'][matches[inliers, 0]], feats1['descriptors'][matches[inliers, 1]]], axis=0)
    return points0, points1, descriptors, E

def compute_relative_pose(E, points0, points1) -> tuple[np.ndarray, Rotation]:
    '''
    Returns the relative pose between two images as (t, R)
    where the pose of img2 is (pose1 + t) * R

    Returns t, r
    '''
    # decompose the essential matrix
    ret, R, t, mask = cv2.recoverPose(E, points0, points1, K, mask=None)
    rvec = Rotation.from_matrix(R).as_rotvec()
    tvec = t.flatten()
    pose_t, pose_r = cv_to_pose(rvec, tvec)
    return pose_t, pose_r

measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v
gtsam_K = gtsam.Cal3_S2(K[0,0], K[1,1], 0.0, K[0,2], K[1,2])

cam0_to_cam1 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(-0.3004961618953472, 0, 0))

def print_factor_errors(graph, values, top_n=10):
    """Print factor errors sorted by magnitude"""
    
    factor_errors = []
    
    # Compute error for each factor
    for i in range(graph.size()):
        factor = graph.at(i)
        try:
            error = factor.error(values)
            factor_errors.append((i, error, factor))
        except Exception as e:
            print(f"Could not compute error for factor {i}: {e}")
            factor_errors.append((i, float('inf'), factor))
    
    # Sort by error (descending)
    factor_errors.sort(key=lambda x: x[1], reverse=True)
    
    # Print top errors
    print(f"Top {min(top_n, len(factor_errors))} factor errors:")
    print("-" * 60)
    
    for rank, (factor_idx, error, factor) in enumerate(factor_errors[:top_n]):
        factor_type = type(factor).__name__
        keys = factor.keys()
        print(f"{rank+1:2d}. Factor {factor_idx:3d} ({factor_type})")
        print(f"    Error: {error:.6f}")
        print(f"    Keys:  {[gtsam.DefaultKeyFormatter(key) for key in keys]}")
        # print expected measurement for calibration factors
        if isinstance(factor, gtsam.GenericProjectionFactorCal3_S2):
            expected_measurement = factor.measured()
            actual_measurement = factor.whitenedError(values) + expected_measurement
            computed_err = np.linalg.norm(actual_measurement - expected_measurement)
            print(f"    Expected Measurement: {expected_measurement}")
            # print actual measurement
            # print(f"    Actual Measurement: {factor.whitenedError(values.atPose3(keys[0]), values.atPoint3(keys[1]), gtsam_K) + expected_measurement}")
            print(f"    Actual Measurement: {factor.whitenedError(values) + expected_measurement}")
            # This print was here to debug if my calculations actually match what gtsam calculates as the error
            # print(f"    Computed Error: {computed_err**2/2:.6f}")
        print()
    
    return factor_errors

def add_vo_factors(
    g: gtsam.NonlinearFactorGraph,
    data: SyncedDatum,
    initial_values: gtsam.Values,
    pose_variable_indices: tuple[int,int],
    keypoint_data: LSHash,
    first=False,
    insert_initial_pose_values=True
):
    '''
    1: compute keypoints matches and relative pose between images
    2. get pose estimate of first image from graph
    3. look up keypoint descriptors in the map
        4. for keypoints that do not exist in the map:
            a. use pose estimate + relative pose to triangulate keypoint
            b. add a new landmark variable to the map 
            c. add factors between the old and new poses, and that new landmark variable
        5. for keypoints that exist in the map:
            a. add factors between the new pose and those landmark variables
    
    map data structure capabilities:
    
    key: descriptor
    value: gtsam Position3 variable

    mutates `g` and `keypoint_data` to add new factors and landmarks
    '''

    # 1:
    vo_points0, vo_points1, vo_descriptors, vo_E = compute_matches(data.img_left_start, data.img_left_end)
    vo_t, vo_r = compute_relative_pose(vo_E, vo_points0, vo_points1)
    # vo_t = np.zeros(3)
    # vo_r = Rotation.identity()

    # 2:
    camL_start_pose = initial_values.atPose3(X(pose_variable_indices[0]))
    camL_end_pose = camL_start_pose * gtsam.Pose3(gtsam.Rot3(vo_r.as_matrix()), t=vo_t.reshape((3,1)))
    camR_start_pose = cam0_to_cam1.compose(camL_start_pose)
    if insert_initial_pose_values:
        initial_values.insert(X(pose_variable_indices[1]), camL_end_pose)
    initial_values.insert(Y(pose_variable_indices[1]), camL_end_pose.compose(cam0_to_cam1))
    # add BetweenFactor
    g.push_back(gtsam.BetweenFactorPose3(
        X(pose_variable_indices[1]),
        Y(pose_variable_indices[1]),
        cam0_to_cam1,
        gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)  # 6DOF: very small noise
    ))
    g.push_back(gtsam.BetweenFactorPose3(
        X(pose_variable_indices[0]),
        X(pose_variable_indices[1]),
        camL_start_pose.between(camL_end_pose),
        gtsam.noiseModel.Isotropic.Sigma(6, 1)  # 6DOF: larger noise than the baseline factor
    ))

    # 3
    def add_stereo_factors(imgL, imgR, img_index):
        stereo_points0, stereo_points1, stereo_descriptors, stereo_E = compute_matches(imgL,imgR)
        for p0, p1, d in list(zip(stereo_points0.astype(np.float64), stereo_points1.astype(np.float64), stereo_descriptors)):
            matches = keypoint_data.query(d.tolist())
            # 4
            if len(matches)==0: # keypoint does not exist in map
                index = len(keypoint_data.storage_instance.keys())
                # add (d, index) to hash table
                # triangulate and add to initial values
                camL_pose = camL_start_pose if img_index == 0 else camL_end_pose
                camR_pose = camR_start_pose if img_index == 0 else cam0_to_cam1.compose(camL_end_pose)
                R0 = camL_pose.matrix()[:3,:3]
                R1 = camR_pose.matrix()[:3,:3]
                proj_mat_0 = K @ np.concatenate((R0, R0@camL_pose.translation().reshape((3,1))), axis=1)
                proj_mat_1 = K @ np.concatenate((R1, R1@camR_pose.translation().reshape((3,1))), axis=1)

                points_3d_homogenous = cv2.triangulatePoints(proj_mat_0, proj_mat_1, p0.reshape((2,1)), p1.reshape((2,1)))
                points_3d = points_3d_homogenous[:3,0] / points_3d_homogenous[3,0]  # convert from homogeneous coordinates
                # verify that projecting into both cameras gives the same points
                p0_proj = cv2.projectPoints(
                    points_3d.reshape((1,3)), 
                    Rotation.from_matrix(camL_pose.rotation().matrix()).as_rotvec(), 
                    camL_pose.translation().flatten(), 
                    K, 
                    None
                )[0].reshape((2,))
                p1_proj = cv2.projectPoints(
                    points_3d.reshape((1,3)), 
                    Rotation.from_matrix(camR_pose.rotation().matrix()).as_rotvec(), 
                    camR_pose.translation().flatten(), 
                    K, 
                    None
                )[0].reshape((2,))
                # assert np.linalg.norm(p0_proj - p0) < 1, f"Projection error for camera 0: {np.linalg.norm(p0_proj - p0)}"
                # assert np.linalg.norm(p1_proj - p1) < 1, f"Projection error for camera 1: {np.linalg.norm(p1_proj - p1)}"
                if np.dot(points_3d - camL_pose.translation().flatten(), camL_pose.rotation().matrix()[:,2]) < 0:
                    continue # TODO: figure out how to handle this case

                if np.linalg.norm(points_3d - camL_pose.translation().flatten()) > 20:
                    # print("Warning: triangulated point is too far from camera, skipping")
                    continue

                g.push_back(gtsam.GenericProjectionFactorCal3_S2(p0.reshape((2,1)), measurement_noise, X(pose_variable_indices[img_index]), L(index), gtsam_K))
                # insert right camera pose
                g.push_back(gtsam.GenericProjectionFactorCal3_S2(p1.reshape((2,1)), measurement_noise, Y(pose_variable_indices[img_index]), L(index), gtsam_K))
                initial_values.insert(L(index), gtsam.Point3(points_3d[0], points_3d[1], points_3d[2]))
                keypoint_data.index(d.cpu().detach().numpy(), extra_data = index)
            # 5
            else: # get closest match and work with it
                index = matches[0][0][1]
                g.push_back(gtsam.GenericProjectionFactorCal3_S2(p0.reshape((2,1)), measurement_noise, X(pose_variable_indices[img_index]), L(index), gtsam_K))
                g.push_back(gtsam.GenericProjectionFactorCal3_S2(p1.reshape((2,1)), measurement_noise, Y(pose_variable_indices[img_index]), L(index), gtsam_K))
    if first:
        add_stereo_factors(data.img_left_start, data.img_right_start, 0)
    add_stereo_factors(data.img_left_end, data.img_right_end, 1)

if __name__ == '__main__':
    imgs = read_images()
    imu = read_imu()
    gt_poses = load_reference_poses()
    imgs_list = []
    for i, img in enumerate(imgs):
        if i>10:
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

    initial_pose_ground_truth = gtsam.Pose3(gtsam.Rot3(synced_data[0].gt_pose_start[1].as_matrix()), synced_data[0].gt_pose_start[0])
    initial_values.insert(X(0), initial_pose_ground_truth)
    initial_values.insert(Y(0), initial_pose_ground_truth.compose(cam0_to_cam1))
    add_vo_factors(g, synced_data[0], initial_values, (0,1), keypoint_data, first=True)
    add_vo_factors(g, synced_data[1], initial_values, (1,2), keypoint_data)
    g.push_back(gtsam.PriorFactorPose3(X(0), initial_pose_ground_truth, gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))))  # prior on first pose

    g.push_back(gtsam.PriorFactorPose3(X(1), gtsam.Pose3(gtsam.Rot3(synced_data[1].gt_pose_start[1].as_matrix()), synced_data[1].gt_pose_start[0]), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))))  # prior on first pose
    # In theory we shouldn't need this prior to make the system not underconstrained
    g.push_back(gtsam.PriorFactorPose3(X(2), gtsam.Pose3(gtsam.Rot3(synced_data[2].gt_pose_start[1].as_matrix()), synced_data[2].gt_pose_start[0]), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))))  # prior on first pose
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
    print_factor_errors(g, result, top_n=20)
