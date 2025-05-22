import numpy as np
from math import cos, sin, atan2, sqrt, pi
from typing import Tuple
from numpy.typing import NDArray

np.set_printoptions(suppress=True, precision=6)


def get_ijk_params_from_branch(
    shoulder_left: float, wrist_rh: float, elbow_up: float
) -> Tuple[float, float, float]:
    i = (shoulder_left - 0.5) * 2
    j = (wrist_rh - 0.5) * 2
    k = i * ((elbow_up - 0.5) * 2)
    return i, j, k


def get_inverse_kinematics(
    dh: Tuple[float, ...],
    transform: NDArray[np.float64],
    shoulder_left: float,
    wrist_rh: float,
    elbow_up: float,
    theta_offsets: NDArray[np.float64],
) -> Tuple[float, float, float, float, float, float] | None:
    a_2, a_3, d_1, d_4, d_5, d_6 = dh
    i, j, k = get_ijk_params_from_branch(shoulder_left, wrist_rh, elbow_up)
    r = transform[:3, :3]
    p = transform[:3, 3]
    A = p[1] - d_6 * r[1, 2]
    B = p[0] - d_6 * r[0, 2]
    if B**2 + A**2 - d_4**2 < 0:
        return None
    theta_1 = i * atan2(sqrt(B**2 + A**2 - d_4**2), d_4) + atan2(B, -A)
    s_1 = sin(theta_1)
    c_1 = cos(theta_1)
    C = c_1 * r[0, 0] + s_1 * r[1, 0]
    D = c_1 * r[1, 1] - s_1 * r[0, 1]
    E = s_1 * r[0, 0] - c_1 * r[1, 0]
    theta_5 = j * atan2(sqrt(E**2 + D**2), s_1 * r[0, 2] - c_1 * r[1, 2])
    s_5 = sin(theta_5)
    c_5 = cos(theta_5)
    theta_6 = atan2(D / s_5, E / s_5)
    s_6 = sin(theta_6)
    c_6 = cos(theta_6)
    F = c_5 * c_6
    theta_234 = atan2(r[2, 0] * F - s_6 * C, F * C + s_6 * r[2, 0])
    s_234 = sin(theta_234)
    c_234 = cos(theta_234)
    K_C = c_1 * p[0] + s_1 * p[1] - s_234 * d_5 + c_234 * s_5 * d_6
    K_S = p[2] - d_1 + c_234 * d_5 + s_234 * s_5 * d_6
    c_3 = (K_S**2 + K_C**2 - a_2**2 - a_3**2) / (2 * a_2 * a_3)
    if c_3**2 > 1:
        return None
    s_3 = sqrt(1 - c_3**2)
    theta_3 = k * atan2(s_3, c_3)
    s_3 = sin(theta_3)
    c_3 = cos(theta_3)
    theta_2 = atan2(K_S, K_C) - atan2(s_3 * a_3, c_3 * a_3 + a_2)
    theta_4 = theta_234 - theta_2 - theta_3
    return (
        theta_1 + theta_offsets[0],
        theta_2 + theta_offsets[1],
        theta_3 + theta_offsets[2],
        theta_4 + theta_offsets[3],
        theta_5 + theta_offsets[4],
        (((theta_6 + theta_offsets[5]) / pi * 180) % 360) * pi / 180,
    )
