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
        print()
    
    return factor_errors

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
    [1,0,0],
    [0,1,0],
    [0, 0, 1]
], dtype=np.float32)

gtsam_K = gtsam.Cal3_S2(K[0,0], K[1,1], 0.0, K[0,2], K[1,2])

points = np.array([
    [0,0,1],
    [0,0,-1],
], dtype=np.float32)

obs = np.array([0,0]).reshape((2,1))

g = gtsam.NonlinearFactorGraph()
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v
g.push_back(gtsam.GenericProjectionFactorCal3_S2(obs, measurement_noise, X(0), L(0), gtsam_K))
initial_values = gtsam.Values()
initial_values.insert(X(0), gtsam.Pose3())
i = 0
initial_values.insert(L(0), gtsam.Point3(points[i][0], points[i][1], points[i][2]))

g.push_back(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))))  # prior on first pose

g.print("Factor Graph:\n")
initial_values.print("Initial Values:\n")
print("initial error = {}".format(g.error(initial_values)))
print_factor_errors(g, initial_values, top_n=10)

params = gtsam.DoglegParams()
params.setVerbosity("TERMINATION")
optimizer = gtsam.DoglegOptimizer(g, initial_values, params)
print("Optimizing:")
result = optimizer.optimize()
result.print("Final results:\n")
print("final error = {}".format(g.error(result)))
