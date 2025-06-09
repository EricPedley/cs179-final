import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt

def visualize_trajectory(poses_list: list[tuple[float, np.ndarray, Rotation]]):
    """
    Visualizes the trajectory of the poses in 3D space. This produces a matplotlib 3D plot
    with a slider to scrub through time, and shows each pose as an axis like in rviz.

    Args:
        poses_list (list[tuple[np.ndarray, Rotation]]): List of tuples containing position and rotation.
            Each tuple contains:
                - np.ndarray: Position as a 3D vector (x, y, z).
                - Rotation: Rotation object from scipy.spatial.transform.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract positions and rotations
    positions = np.array([pose[1] for pose in poses_list])
    rotations = [pose[2] for pose in poses_list]

    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', markersize=2, label='Trajectory')

    # # Plot each pose as an axis
    # for pos, rot in zip(positions, rotations):
    #     # Get the rotation matrix
    #     R_matrix = rot.as_matrix()
    #     # Define the axes (x, y, z)
    #     axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 0.1  # Scale the axes
    #     # Transform the axes by the rotation and translate by position
    #     transformed_axes = pos + R_matrix @ axes.T

    #     # Plot the axes
    #     ax.quiver(pos[0], pos[1], pos[2],
    #               transformed_axes[0, 0] - pos[0],
    #               transformed_axes[1, 0] - pos[1],
    #               transformed_axes[2, 0] - pos[2],
    #               color='r', length=0.1)
    #     ax.quiver(pos[0], pos[1], pos[2],
    #               transformed_axes[0, 1] - pos[0],
    #               transformed_axes[1, 1] - pos[1],
    #               transformed_axes[2, 1] - pos[2],
    #               color='g', length=0.1)
    #     ax.quiver(pos[0], pos[1], pos[2],
    #               transformed_axes[0, 2] - pos[0],
    #               transformed_axes[1, 2] - pos[1],
    #               transformed_axes[2, 2] - pos[2],
    #               color='b', length=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Visualization')
    plt.legend()
    plt.show()