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
    [396.2, 0, 800/2],
    [0, 396.2, 400/2],
    [0, 0, 1]
])
# focal length is 3.5mm physically
PIXELS_TO_METERS = 3.5e-3 / 396.2  # focal length in meters divided by focal length in pixels

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

def add_vo_factors(
    g: gtsam.NonlinearFactorGraph,
    data: SyncedDatum,
    initial_values: gtsam.Values,
    pose_variable_indices: tuple[int,int],
    keypoint_data: LSHash
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
    points0, points1, descriptors, E = compute_matches(data.img_left_start, data.img_right_start)
    t, r = compute_relative_pose(E, points0, points1)

    # 2:
    cam0_pose = initial_values.atPose3(X(pose_variable_indices[0]))
    cam1_pose = cam0_pose * gtsam.Pose3(gtsam.Rot3(r.as_matrix()), t=t.reshape((3,1)))
    initial_values.insert(X(pose_variable_indices[1]), cam1_pose)

    # 3
    for p0, p1, d in list(zip(points0.astype(np.float64), points1.astype(np.float64), descriptors))[:5]:
        matches = keypoint_data.query(d.tolist())
        # 4
        if len(matches)==0: # keypoint does not exist in map
            index = len(keypoint_data.storage_instance.keys())
            # add (d, index) to hash table
            g.push_back(gtsam.GenericProjectionFactorCal3_S2(p0.reshape((2,1)), measurement_noise, X(pose_variable_indices[0]), L(index), gtsam_K))
            g.push_back(gtsam.GenericProjectionFactorCal3_S2(p1.reshape((2,1)), measurement_noise, X(pose_variable_indices[1]), L(index), gtsam_K))
            # triangulate and add to initial values
            R0 = cam0_pose.matrix()[:3,:3]
            R1 = cam1_pose.matrix()[:3,:3]
            proj_mat_0 = K @ np.concatenate((R0, cam0_pose.matrix()[:3, 3].reshape((3,1))), axis=1)
            proj_mat_1 = K @ np.concatenate((R1, cam1_pose.matrix()[:3, 3].reshape((3,1))), axis=1)

            points_3d_homogenous = cv2.triangulatePoints(proj_mat_0, proj_mat_1, p0.reshape((2,1)), p1.reshape((2,1)))
            points_3d = points_3d_homogenous[:3,0] / points_3d_homogenous[3,0]  # convert from homogeneous coordinates
            # verify that projecting into both cameras gives the same points
            p0_proj = cv2.projectPoints(
                points_3d.reshape((1,3)), 
                Rotation.from_matrix(cam0_pose.rotation().matrix()).as_rotvec(), 
                cam0_pose.translation().flatten(), 
                K, 
                None
            )[0].reshape((2,))
            p1_proj = cv2.projectPoints(
                points_3d.reshape((1,3)), 
                Rotation.from_matrix(cam1_pose.rotation().matrix()).as_rotvec(), 
                cam1_pose.translation().flatten(), 
                K, 
                None
            )[0].reshape((2,))
            # assert np.linalg.norm(p0_proj - p0) < 1e-3, f"Projection error for camera 0: {np.linalg.norm(p0_proj - p0)}"
            # assert np.linalg.norm(p1_proj - p1) < 1e-3, f"Projection error for camera 1: {np.linalg.norm(p1_proj - p1)}"

            # initial_values.insert(L(index), gtsam.Point3(points_3d[0], points_3d[1], points_3d[2]))
            keypoint_data.index(d.cpu().detach().numpy(), extra_data = index)
        # 5
        else: # get closest match and work with it
            index = matches[0][0][1]
            g.push_back(gtsam.GenericProjectionFactorCal3_S2(p0.reshape((2,1)), measurement_noise, X(pose_variable_indices[0]), L(index), gtsam_K))
            g.push_back(gtsam.GenericProjectionFactorCal3_S2(p1.reshape((2,1)), measurement_noise, X(pose_variable_indices[1]), L(index), gtsam_K))

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
    add_vo_factors(g, synced_data[0], initial_values, (0,1), keypoint_data)
    add_vo_factors(g, synced_data[1], initial_values, (1,2), keypoint_data)

    g.push_back(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(gtsam.Rot3(synced_data[0].gt_pose_start[1].as_matrix()), synced_data[0].gt_pose_start[0]), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))))  # prior on first pose

    # "cheat" a little bit to make the system not underconstrained
    g.push_back(gtsam.PriorFactorPose3(X(1), gtsam.Pose3(gtsam.Rot3(synced_data[1].gt_pose_start[1].as_matrix()), synced_data[1].gt_pose_start[0]), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))))  # prior on first pose
    # In theory we shouldn't need this prior to make the system not underconstrained
    g.push_back(gtsam.PriorFactorPose3(X(2), gtsam.Pose3(gtsam.Rot3(synced_data[2].gt_pose_start[1].as_matrix()), synced_data[2].gt_pose_start[0]), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))))  # prior on first pose
    g.print("Factor Graph:\n")
    initial_values.print("Initial Values:\n")
    
    params = gtsam.DoglegParams()
    params.setVerbosity("TERMINATION")
    optimizer = gtsam.DoglegOptimizer(g, initial_values, params)
    print("Optimizing:")
    result = optimizer.optimize()
    result.print("Final results:\n")
    print("initial error = {}".format(g.error(initial_values)))
    print("final error = {}".format(g.error(result)))
