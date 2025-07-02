import numpy as np
from math import cos, sin
from typing import Tuple
from numpy.typing import NDArray
import os
import sys

# Get the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the current directory to Python path
sys.path.append(current_dir)
# Now import from the same directory
from utils.utils import get_euler_angles_from_matrix

np.set_printoptions(suppress=True, precision=6)


def transformation_to_pose(
    T: NDArray[np.float64],
) -> Tuple[float, float, float, float, float, float]:
    """
    Convert a 4x4 transformation matrix to (x, y, z, theta_x, theta_y, theta_z) format.

    Args:
        T: 4x4 homogeneous transformation matrix

    Returns:
        Tuple of (x, y, z, theta_x, theta_y, theta_z) where:
        - (x, y, z) is the position
        - (theta_x, theta_y, theta_z) are the Euler angles in radians
    """
    # Extract position
    x, y, z = T[:3, 3]

    # Extract rotation matrix and convert to Euler angles
    R = T[:3, :3]
    theta_x, theta_y, theta_z = get_euler_angles_from_matrix(R)

    return (x, y, z, theta_x, theta_y, theta_z)


def get_forward_kinematics(
    dh: Tuple[float, ...],
    joint_angles: Tuple[float, ...],
    theta_offsets: NDArray[np.float64],
    return_rpy: bool = True,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute the forward kinematics of a 6-DOF robotic arm using DH parameters.

    Args:
        dh: Tuple of DH parameters (a_2, a_3, d_1, d_4, d_5, d_6)
        joint_angles: Tuple of 6 joint angles in radians
        theta_offsets: Array of 6 joint angle offsets in radians

    Returns:
        Tuple of (x, y, z, theta_x, theta_y, theta_z) representing the end effector pose
    """
    a_2, a_3, d_1, d_4, d_5, d_6 = dh
    theta_1, theta_2, theta_3, theta_4, theta_5, theta_6 = [
        angle - offset for angle, offset in zip(joint_angles, theta_offsets)
    ]

    # Initialize the transformation matrix as identity
    T = np.eye(4)

    # Compute the transformation matrix for each joint
    # Joint 1
    c1, s1 = cos(theta_1), sin(theta_1)
    T1 = np.array([[c1, -s1, 0, 0], [s1, c1, 0, 0], [0, 0, 1, d_1], [0, 0, 0, 1]])
    T = T @ T1

    # Joint 2
    c2, s2 = cos(theta_2), sin(theta_2)
    T2 = np.array([[c2, -s2, 0, 0], [0, 0, 1, 0], [-s2, -c2, 0, 0], [0, 0, 0, 1]])
    T = T @ T2

    # Joint 3
    c3, s3 = cos(theta_3), sin(theta_3)
    T3 = np.array([[c3, -s3, 0, a_2], [s3, c3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T = T @ T3

    # Joint 4
    c4, s4 = cos(theta_4), sin(theta_4)
    T4 = np.array([[c4, -s4, 0, a_3], [s4, c4, 0, 0], [0, 0, 1, d_4], [0, 0, 0, 1]])
    T = T @ T4

    # Joint 5
    c5, s5 = cos(theta_5), sin(theta_5)
    T5 = np.array([[c5, -s5, 0, 0], [0, 0, 1, d_5], [-s5, -c5, 0, 0], [0, 0, 0, 1]])
    T = T @ T5

    # Joint 6
    c6, s6 = cos(theta_6), sin(theta_6)
    T6 = np.array([[c6, -s6, 0, 0], [0, 0, 1, d_6], [-s6, -c6, 0, 0], [0, 0, 0, 1]])
    T = T @ T6

    # Convert the transformation matrix to pose format
    return transformation_to_pose(T)


def _dh(a: float, alpha: float, d: float, theta: float) -> NDArray[np.float64]:
    """Classic DH matrix  A = Rz(θ) Tz(d) Tx(a) Rx(α)."""
    sa, ca = np.sin(alpha), np.cos(alpha)
    st, ct = np.sin(theta), np.cos(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def forward_kinematics(
    dh: tuple[float, ...],
    thetas: tuple[float, ...] | NDArray[np.float64],
    theta_offsets: tuple[float, ...] | NDArray[np.float64] | None = None,
    return_rpy: bool = False,
) -> NDArray[np.float64] | tuple[float, float, float, float, float, float]:
    """
    Parameters
    ----------
    dh
        (a2, a3, d1, d4, d5, d6) – identical ordering to ``get_inverse_kinematics``.
    thetas
        Joint angles (θ₁ … θ₆) *with* the same zero reference as your IK.
    theta_offsets
        Same offsets vector you pass to IK (default zeros).
    return_rpy
        If True, returns (x, y, z, roll, pitch, yaw);
        otherwise returns the 4 × 4 transform.

    Notes
    -----
    The fixed geometry for a UR-family 6-DoF arm is:

        α = [π/2, 0, 0,  π/2, -π/2, 0]
        a = [   0, a2, a3,    0,     0, 0]
        d = [  d1,  0,  0,   d4,    d5, d6]
    """
    a2, a3, d1, d4, d5, d6 = dh
    if theta_offsets is None:
        theta_offsets = np.zeros(6)
    q = np.asarray(thetas, dtype=float) - np.asarray(theta_offsets, dtype=float)

    a = [0.0, a2, a3, 0.0, 0.0, 0.0]
    d = [d1, 0.0, 0.0, d4, d5, d6]
    alpha = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]

    T = np.eye(4)
    for ai, alpi, di, qi in zip(a, alpha, d, q):
        T = T @ _dh(ai, alpi, di, qi)

    if return_rpy:
        return transformation_to_pose(T)
    return T
