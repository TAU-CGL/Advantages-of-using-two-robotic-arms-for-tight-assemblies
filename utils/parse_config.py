import json
import math
from typing import Tuple, List
from numpy.typing import NDArray

import numpy as np


def get_dh_params(file_path: str) -> Tuple[float, ...]:
    with open(file_path, "r") as file:
        data = json.load(file)
    values_tuple: Tuple[float, ...] = tuple(data.values())
    return values_tuple


def parse_config(
    file_path: str,
) -> Tuple[
    str,
    str,
    Tuple[float, ...],
    List[float],
    float,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    with open(file_path, "r") as file:
        data = json.load(file)
    dh_path: str = data["dh"]
    path_file: str = data["path"]
    urdf_path: str = data["urdf_path"]
    ref_point_relative_pose: List[float] = data["ref_point_relative_pose"]
    initial_pose: List[float] = data["initial_pose"]
    theta_offsets: List[float] = data["theta_offsets"]
    position_factor: float = data["position_factor"]
    dh: Tuple[float, ...] = get_dh_params(dh_path)
    ref_point_relative_pose[3:] = list(map(math.radians, ref_point_relative_pose[3:]))
    initial_pose[-3:] = list(map(math.radians, initial_pose[-3:]))
    return (
        path_file,
        urdf_path,
        dh,
        initial_pose,
        position_factor,
        np.array(ref_point_relative_pose),
        np.radians(np.array(theta_offsets)),
    )


def parse_config_dual_arm(
    file_path: str,
) -> Tuple[
    str,
    str,
    str,
    str,
    Tuple[float, ...],
    Tuple[float, ...],
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    bool,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    with open(file_path, "r") as file:
        data = json.load(file)
    dynamic_dh_path: str = data["dynamic_part_arm_dh"]
    static_dh_path: str = data["static_part_arm_dh"]
    trajectory_file: str = data["dynamic_part_trajectory"]
    static_part_trajectory_file: str = data["static_part_trajectory"]
    dynamic_part_arm_urdf_path: str = data["dynamic_part_arm_urdf_path"]
    static_part_arm_urdf_path: str = data["static_part_arm_urdf_path"]
    dynamic_ref_point_relative_pose: List[float] = data[
        "dynamic_ref_point_relative_pose"
    ]
    static_ref_point_relative_pose: List[float] = data["static_ref_point_relative_pose"]
    theta_offsets: List[float] = data["theta_offsets"]
    position_factor: float = data["position_factor"]
    test_AB = False
    if "test_AB" in data:
        test_AB = data["test_AB"]
    dynamic_part_arm_position = [0, 0, 0]
    if "dynamic_part_arm_position" in data:
        dynamic_part_arm_position = data["dynamic_part_arm_position"]
    static_part_arm_position = [0, 0, 0]
    if "static_part_arm_position" in data:
        static_part_arm_position = data["static_part_arm_position"]
    static_part_relative_position = [0, 0, 0]
    if "static_part_relative_position" in data:
        static_part_relative_position = data["static_part_relative_position"]
    static_part_relative_rotation = [0, 0, 0]
    if "static_part_relative_rotation" in data:
        static_part_relative_rotation = data["static_part_relative_rotation"]
    dynamic_part_arm_dh: Tuple[float, ...] = get_dh_params(dynamic_dh_path)
    static_part_arm_dh: Tuple[float, ...] = get_dh_params(static_dh_path)
    dynamic_ref_point_relative_pose[3:] = list(
        map(math.radians, dynamic_ref_point_relative_pose[3:])
    )
    static_ref_point_relative_pose[3:] = list(
        map(math.radians, static_ref_point_relative_pose[3:])
    )
    return (
        trajectory_file,
        static_part_trajectory_file,
        dynamic_part_arm_urdf_path,
        static_part_arm_urdf_path,
        dynamic_part_arm_dh,
        static_part_arm_dh,
        position_factor,
        np.radians(np.array(theta_offsets)),
        np.array(dynamic_ref_point_relative_pose),
        np.array(static_ref_point_relative_pose),
        test_AB,
        np.array(dynamic_part_arm_position),
        np.array(static_part_arm_position),
        np.array(static_part_relative_position),
        np.radians(np.array(static_part_relative_rotation)),
    )
