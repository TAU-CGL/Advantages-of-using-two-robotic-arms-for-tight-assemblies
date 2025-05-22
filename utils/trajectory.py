from typing import List
import numpy as np
from math import pi

from utils.set_trajectory_ref_point import set_trajectory_ref_point
from utils.utils import (
    euler_to_matrix,
    get_euler_angles_from_matrix,
    get_transformation_matrix,
)

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def pose_to_matrix(x, y, z, rx, ry, rz):
    r = R.from_euler("zyx", [rz, ry, rx])
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def matrix_to_pose(T):
    r = R.from_matrix(T[:3, :3])
    euler = r.as_euler("zyx")
    return (*T[:3, 3], *euler[::-1])


def get_relative_pose(pose1, pose2):

    T1 = pose_to_matrix(*pose1)
    T2 = pose_to_matrix(*pose2)

    # Compose: T3 = T2 * T1
    T3 = T2 @ np.linalg.inv(T1)
    return matrix_to_pose(T3)


def adjust_trajectory_to_joint_limits(ik_trajectory):
    for j in range(ik_trajectory.shape[1]):
        while abs(ik_trajectory[0][j]) > pi:
            if ik_trajectory[0][j] > pi:
                ik_trajectory[0, j] -= 2 * pi
            elif ik_trajectory[0][j] < -pi:
                ik_trajectory[0, j] += 2 * pi
    for i in range(1, ik_trajectory.shape[0]):
        for j in range(ik_trajectory.shape[1]):
            diff = 2 * pi
            while abs(diff) > pi:
                diff = ik_trajectory[i, j] - ik_trajectory[i - 1, j]
                if diff > pi:
                    ik_trajectory[i, j] -= 2 * pi
                elif diff < -pi:
                    ik_trajectory[i, j] += 2 * pi
    return ik_trajectory


def place_trajectory(
    trajectory,
    initial_pose,
    position_factor,
    ref_point_relative_pose,
):
    trajectory = np.copy(trajectory)
    init_rot_matrix = np.dot(
        euler_to_matrix(initial_pose[3], initial_pose[4], initial_pose[5]),
        np.linalg.inv(
            euler_to_matrix(
                trajectory[0][3], trajectory[0][4], trajectory[0][5]
            )  # inverse of initial dynamic sub assembly angle relative to static sub assembly
        ),
    )
    trajectory[:, :3] *= position_factor
    # trajectory = set_trajectory_ref_point(
    #     np.copy(trajectory),
    #     ref_point_offset,
    #     dynamic_sub_assembly_rotation_relative_to_final_joint_mat,
    # )
    trajectory[:, :3] = np.dot(
        init_rot_matrix,
        (trajectory[:, :3] - trajectory[0, :3]).T,
    ).T
    offset = initial_pose[:3] - trajectory[0, :3]
    trajectory[:, :3] += offset
    offset_trajectory = trajectory
    for i in range(len(offset_trajectory)):
        original = offset_trajectory[i, 3:]
        mat = euler_to_matrix(original[0], original[1], original[2])
        rotated_mat = np.dot(init_rot_matrix, mat)
        rotated_euler = np.array(get_euler_angles_from_matrix(rotated_mat))
        offset_trajectory[i, 3:] = rotated_euler
    ref_point_relative_rot_matrix = euler_to_matrix(
        ref_point_relative_pose[3],
        ref_point_relative_pose[4],
        ref_point_relative_pose[5],
    )
    for i in range(len(offset_trajectory)):
        original = offset_trajectory[i, 3:]
        mat = euler_to_matrix(original[0], original[1], original[2])
        offset_trajectory[i, :3] += np.dot(mat, ref_point_relative_pose[:3])
        offset_trajectory[i, 3:] = get_euler_angles_from_matrix(
            np.dot(mat, ref_point_relative_rot_matrix)
        )
    return offset_trajectory


# dynamic trajectory is expected to be in the static part's frame (the static part should be at the origin)
def generate_parallel_trajectories(dynamic_trajectory: np.ndarray):

    dynamic_parallel_trajectory = np.zeros_like(dynamic_trajectory)
    static_parallel_trajectory = np.zeros_like(dynamic_trajectory)
    dynamic_parallel_trajectory[0] = dynamic_trajectory[0]
    for pose_index in range(1, len(dynamic_trajectory)):
        dynamic_parallel_trajectory[pose_index, :3] = dynamic_parallel_trajectory[
            pose_index - 1, :3
        ] + 0.5 * (
            dynamic_trajectory[pose_index, :3] - dynamic_trajectory[pose_index - 1, :3]
        )
        diff_rot_mat = R.from_euler(
            "zyx", dynamic_trajectory[pose_index, 3:][::-1]
        ).as_matrix() @ np.linalg.inv(
            R.from_euler(
                "zyx", dynamic_trajectory[pose_index - 1, 3:][::-1]
            ).as_matrix()
        )

        target_rot = R.from_matrix(
            diff_rot_mat
            @ R.from_euler(
                "zyx", dynamic_parallel_trajectory[pose_index - 1, 3:][::-1]
            ).as_matrix()
        ).as_euler("zyx")[::-1]
        slerp = Slerp(
            [0, 1],
            R.from_euler(
                "zyx",
                [
                    dynamic_parallel_trajectory[pose_index - 1, 3:][::-1],
                    target_rot[::-1],
                ],
            ),
        )
        dynamic_parallel_trajectory[pose_index, 3:] = slerp(0.5).as_euler("zyx")[::-1]
        static_parallel_trajectory[pose_index] = get_relative_pose(
            dynamic_trajectory[pose_index], dynamic_parallel_trajectory[pose_index]
        )
    return dynamic_parallel_trajectory, static_parallel_trajectory


def generate_alternative_part_trajectory(trajectory: np.ndarray):
    alternative_part_trajectory = np.zeros_like(trajectory)
    for pose_index in range(0, len(trajectory)):
        alternative_part_trajectory[pose_index] = get_relative_pose(
            trajectory[pose_index], np.zeros_like(trajectory[pose_index])
        )
    return alternative_part_trajectory


def get_alternating_trajectories(
    trajectories: List[np.ndarray],
    alternation_indices: List[int],
) -> List[np.ndarray]:
    assert all(trajectory.shape == trajectories[0].shape for trajectory in trajectories)
    parallel_trajectories = generate_parallel_trajectories(trajectories[0])
    period_length = trajectories[0].shape[0] // len(alternation_indices)
    alternating_trajectories = [
        np.zeros_like(trajectories[i]) for i in range(len(trajectories))
    ]
    for i, trajectory_index in enumerate(alternation_indices):
        if trajectory_index < len(trajectories):
            alternating_trajectories[trajectory_index][i * period_length :] = (
                place_trajectory(
                    trajectories[trajectory_index][i * period_length :, :],
                    alternating_trajectories[trajectory_index][i * period_length],
                    1,
                    np.array([0, 0, 0, 0, 0, 0]),
                )
            )
            for j in range(len(trajectories)):
                if j != trajectory_index:
                    alternating_trajectories[j][i * period_length :, :] = (
                        alternating_trajectories[j][i * period_length]
                    )
        else:
            for j in range(len(trajectories)):
                alternating_trajectories[j][i * period_length :] = place_trajectory(
                    parallel_trajectories[j][i * period_length :, :],
                    alternating_trajectories[j][i * period_length],
                    1,
                    np.array([0, 0, 0, 0, 0, 0]),
                )
    return alternating_trajectories
