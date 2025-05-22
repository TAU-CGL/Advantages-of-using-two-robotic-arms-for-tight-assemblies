import numpy as np
from utils.utils import euler_to_matrix


def adjust_pose_to_ref_point(
    pose, sub_assembly_rotation_relative_to_final_joint_mat, ref_point_offset
):
    pose_rot_mat = euler_to_matrix(pose[3], pose[4], pose[5])
    ref_pose = pose[:3] + np.dot(
        np.dot(pose_rot_mat, sub_assembly_rotation_relative_to_final_joint_mat),
        ref_point_offset,
    )
    return ref_pose


def set_trajectory_ref_point(
    trajectory,
    ref_point_offset,
    dynamic_sub_assembly_rotation_relative_to_final_joint_mat,
):
    for point_idx in range(len(trajectory)):
        point = trajectory[point_idx]
        point_rot_mat = euler_to_matrix(point[3], point[4], point[5])
        ref_point = point[:3] + np.dot(
            np.dot(
                point_rot_mat, dynamic_sub_assembly_rotation_relative_to_final_joint_mat
            ),
            ref_point_offset,
        )
        trajectory[point_idx][:3] = ref_point
    return trajectory
