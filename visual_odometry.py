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

    # 3
    for p0, p1, d in zip(points0, points1, descriptors):
        matches = keypoint_data.query(d.tolist())
        # 4
        if len(matches)==0: # keypoint does not exist in map
            index = len()
            # add (d, index) to hash table
            landmark = L(index)
            g.push_back(GenericProjectionFactorCal3_S2(p0, measurement_noise, X(pose_variable_indices[0]), L(index)))
            g.push_back(GenericProjectionFactorCal3_S2(p1, measurement_noise, X(pose_variable_indices[1]), L(index)))
        # 5
        else: # get closest match and work with it
            index = matches[0][0][1]
            landmark = L(index)
            g.push_back(GenericProjectionFactorCal3_S2(p0, measurement_noise, X(pose_variable_indices[0]), L(index)))
            g.push_back(GenericProjectionFactorCal3_S2(p1, measurement_noise, X(pose_variable_indices[1]), L(index)))

            point_cloud_hash.index(feature, extra_data = len(point_cloud))

if __name__ == '__main__':

    from data_reading import read_images, load_reference_poses, read_imu, sync_data
    from visualization import visualize_trajectory
    from visual_odometry import compute_relative_pose, compute_matches, add_vo_factors
    from imu_preintegration import add_imu_factors

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
    g = NonlinearFactorGraph()
    keypoint_data = LSHash(
        hash_size=32,
        input_dim=128 if USE_SIFT else 32,
        num_hashtables=1,
    )
    g.push_back()
    add_vo_factors(g, synced_data[0], initial_values, (0,1), keypoint_data)
