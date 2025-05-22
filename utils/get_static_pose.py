import numpy as np
from utils.set_trajectory_ref_point import adjust_pose_to_ref_point
from utils.trajectory import get_relative_pose, matrix_to_pose, pose_to_matrix
from utils.utils import euler_to_matrix, get_euler_angles_from_matrix


def get_static_pose(
    initial_pose,
    static_part_arm_position,
    dynamic_part_arm_position,
    static_part_relative_position,
    static_part_relative_rotation,
):
    relative_pose = np.array(
        get_relative_pose(
            matrix_to_pose(
                np.linalg.inv(
                    pose_to_matrix(
                        *static_part_relative_position, *static_part_relative_rotation
                    )
                )
            ),
            initial_pose,
        )
    )
    relative_pose[0:3] += dynamic_part_arm_position - static_part_arm_position
    return relative_pose
