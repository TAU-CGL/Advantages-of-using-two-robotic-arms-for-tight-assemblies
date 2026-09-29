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
    static_part_arm_rotation=None,
    dynamic_part_arm_rotation=None,
):
    # static_part_arm_rotation/dynamic_part_arm_rotation: orientation (euler xyz, rad) of each
    # arm's base frame relative to a common world frame. Defaulting both to zero reproduces the
    # previous translation-only behavior (the two bases were assumed co-oriented).
    if static_part_arm_rotation is None:
        static_part_arm_rotation = np.zeros(3)
    if dynamic_part_arm_rotation is None:
        dynamic_part_arm_rotation = np.zeros(3)

    # target static-part pose, expressed in the dynamic arm's own base frame
    target_in_dynamic_base = pose_to_matrix(*initial_pose) @ pose_to_matrix(
        *static_part_relative_position, *static_part_relative_rotation
    )

    dynamic_base_in_world = pose_to_matrix(
        *dynamic_part_arm_position, *dynamic_part_arm_rotation
    )
    static_base_in_world = pose_to_matrix(
        *static_part_arm_position, *static_part_arm_rotation
    )

    # re-express that target pose in the static arm's own base frame
    target_in_static_base = (
        np.linalg.inv(static_base_in_world) @ dynamic_base_in_world @ target_in_dynamic_base
    )
    return np.array(matrix_to_pose(target_in_static_base))
