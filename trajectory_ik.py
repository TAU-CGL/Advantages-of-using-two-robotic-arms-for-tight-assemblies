import uuid
import numpy as np
from math import pi
from typing import Tuple, List
from numpy.typing import NDArray
import networkx as nx

from collision_detection import check_collision_between_robots, check_self_collision
from utils.get_static_pose import get_static_pose
from utils.ik import get_inverse_kinematics
from utils.trajectory import adjust_trajectory_to_joint_limits, place_trajectory
from utils.utils import (
    get_transformation_matrix,
    min_max_edge_dijkstra,
)

np.set_printoptions(suppress=True, precision=6)


def generate_ik_of_trajectory(
    trajectory_file: str,
    dh: Tuple[float, ...],
    initial_pose: List[float],
    position_factor: float,
    ref_point_relative_pose: NDArray[np.float64],
    theta_offsets: NDArray[np.float64],
    robot_id: str,
    output_path: str,
    test_AB=False,
    delta=10,
) -> Tuple[int, float, bool]:
    trajectory = np.genfromtxt(
        trajectory_file,
        delimiter=",",
        skip_header=1,
    )
    placed_trajectory = place_trajectory(
        trajectory,
        initial_pose,
        position_factor,
        ref_point_relative_pose,
    )
    G = nx.DiGraph()
    G.add_node("s")
    G.add_node("t")
    previous_pose_solutions: list[Tuple[str, NDArray]] = []
    if test_AB:
        too_close_to_base = (
            np.min(np.linalg.norm(placed_trajectory[:, :2], axis=1)) < dh[3]
        )
        if too_close_to_base:
            return -1, 0, False
    for i in range(len(placed_trajectory)):
        T = get_transformation_matrix(
            placed_trajectory[i, 0],
            placed_trajectory[i, 1],
            placed_trajectory[i, 2],
            placed_trajectory[i, 3],
            placed_trajectory[i, 4],
            placed_trajectory[i, 5],
        )
        current_pose_solutions = []
        trajectory_blocked = True
        for shoulder in [0, 1]:
            for wrist in [0, 1]:
                for elbow in [1]:
                    # considering only "elbow up" branches, change elbow range to [0, 1] to consider elbow down as well
                    current_ik = get_inverse_kinematics(
                        dh, T, shoulder, wrist, elbow, theta_offsets
                    )
                    if current_ik != None:
                        current_node_id = uuid.uuid4()
                        current_ik = np.array(current_ik)
                        collision_detected = check_self_collision(current_ik, robot_id)
                        if not collision_detected:
                            node_added = False
                            if i == 0:
                                current_pose_solutions.append(
                                    (current_node_id, current_ik)
                                )
                                G.add_node(current_node_id, ik=current_ik)
                            for node_id, node_ik in previous_pose_solutions:
                                dists = (np.abs(current_ik - node_ik) * 180 / pi) % 360
                                dists = np.minimum(dists, 360 - dists)
                                weight = np.max(dists)
                                if weight < delta:
                                    if not node_added:
                                        current_pose_solutions.append(
                                            (current_node_id, current_ik)
                                        )
                                        G.add_node(current_node_id, ik=current_ik)
                                        node_added = True
                                    trajectory_blocked = False
                                    G.add_edge(
                                        node_id,
                                        current_node_id,
                                        weight=weight,
                                    )
        if i == 0:
            for node_id, _ in current_pose_solutions:
                trajectory_blocked = False
                G.add_edge("s", node_id)
        if trajectory_blocked:
            return i, 0, False
        previous_pose_solutions = current_pose_solutions
        current_pose_solutions = []
    for node_id, _ in previous_pose_solutions:
        G.add_edge(node_id, "t")
    min_max_edge_trajectory = min_max_edge_dijkstra(G, "s", "t", "weight")
    if min_max_edge_trajectory == None:
        return i, 0, False
    ik_trajectory_max_edge, graph_ik_trajectory = min_max_edge_trajectory
    ik_trajectory = []
    for node_index in range(1, len(graph_ik_trajectory) - 1):
        node = G.nodes[graph_ik_trajectory[node_index]]
        ik_trajectory.append(node["ik"])
    ik_trajectory = adjust_trajectory_to_joint_limits(np.array(ik_trajectory))
    np.savetxt(
        output_path,
        np.array(ik_trajectory),
        delimiter=",",
        fmt="%.5f",
    )
    return len(placed_trajectory) - 1, ik_trajectory_max_edge, True


def generate_dual_arm_ik_of_trajectory(
    trajectory: NDArray[np.float64],
    static_part_trajectory: NDArray[np.float64],
    dh_static: Tuple[float, ...],
    dh_dynamic: Tuple[float, ...],
    initial_pose: List[float],
    static_part_arm_position: NDArray[np.float64],
    dynamic_part_arm_position: NDArray[np.float64],
    static_part_relative_position: NDArray[np.float64],
    static_part_relative_rotation: NDArray[np.float64],
    position_factor: float,
    dynamic_ref_point_relative_pose: NDArray[np.float64],
    static_ref_point_relative_pose: NDArray[np.float64],
    theta_offsets: NDArray[np.float64],
    dynamic_robot_id: str,
    static_robot_id: str,
    test_AB=False,
    keep_in_center=False,
    makespan_objective=False,
    delta=50,
) -> Tuple[NDArray, NDArray, int, float, bool]:
    placed_trajectory = place_trajectory(
        trajectory,
        initial_pose,
        position_factor,
        dynamic_ref_point_relative_pose,
    )
    static_pose = get_static_pose(
        initial_pose,
        static_part_arm_position,
        dynamic_part_arm_position,
        static_part_relative_position,
        static_part_relative_rotation,
    )
    placed_static_trajectory = place_trajectory(
        static_part_trajectory,
        static_pose,
        position_factor,
        static_ref_point_relative_pose,
    )
    dynamic_initial_position = placed_trajectory[0, :3]
    if keep_in_center:
        for i in range(len(placed_trajectory)):
            placed_trajectory_offset = (
                placed_trajectory[i, :3] - dynamic_initial_position
            )
            placed_trajectory[i, :3] = dynamic_initial_position
            placed_static_trajectory[i, :3] = (
                placed_static_trajectory[i, :3] - placed_trajectory_offset
            )
    G = nx.DiGraph()
    G.add_node("s")
    G.add_node("t")
    previous_pose_solutions: list[Tuple[str, NDArray, NDArray]] = []
    if test_AB:
        too_close_to_base = (
            np.min(np.linalg.norm(placed_trajectory[:, :2], axis=1)) < dh_dynamic[3]
        )
        if too_close_to_base:
            return np.array([]), np.array([]), -1, 0, False
    for i in range(len(placed_trajectory)):
        T = get_transformation_matrix(
            placed_trajectory[i, 0],
            placed_trajectory[i, 1],
            placed_trajectory[i, 2],
            placed_trajectory[i, 3],
            placed_trajectory[i, 4],
            placed_trajectory[i, 5],
        )
        static_T = get_transformation_matrix(
            placed_static_trajectory[i, 0],
            placed_static_trajectory[i, 1],
            placed_static_trajectory[i, 2],
            placed_static_trajectory[i, 3],
            placed_static_trajectory[i, 4],
            placed_static_trajectory[i, 5],
        )
        static_iks = {}
        for static_shoulder in [0, 1]:
            for static_wrist in [0, 1]:
                for static_elbow in [1]:
                    static_ik = get_inverse_kinematics(
                        dh_static,
                        static_T,
                        static_shoulder,
                        static_wrist,
                        static_elbow,
                        theta_offsets,
                    )
                    if static_ik != None:
                        static_ik = np.array(static_ik)
                        collision_detected = check_self_collision(
                            static_ik, static_robot_id
                        )
                        if not collision_detected:
                            static_iks[
                                (static_shoulder, static_wrist, static_elbow)
                            ] = static_ik
        current_pose_solutions = []
        trajectory_blocked = True
        for dynamic_shoulder in [0, 1]:
            for dynamic_wrist in [0, 1]:
                for dynamic_elbow in [1]:
                    # considering only "elbow up" branches, change elbow range to [0, 1] to consider elbow down as well
                    current_dynamic_ik = get_inverse_kinematics(
                        dh_dynamic,
                        T,
                        dynamic_shoulder,
                        dynamic_wrist,
                        dynamic_elbow,
                        theta_offsets,
                    )
                    if current_dynamic_ik != None:
                        current_dynamic_ik = np.array(current_dynamic_ik)
                        self_collision_detected = check_self_collision(
                            current_dynamic_ik, dynamic_robot_id
                        )
                        if not self_collision_detected:
                            for (
                                static_shoulder,
                                static_wrist,
                                static_elbow,
                            ) in static_iks:
                                current_static_ik = static_iks[
                                    (static_shoulder, static_wrist, static_elbow)
                                ]
                                robots_collision_detected = (
                                    check_collision_between_robots(
                                        current_dynamic_ik,
                                        dynamic_robot_id,
                                        current_static_ik,
                                        static_robot_id,
                                    )
                                )
                                if not robots_collision_detected:
                                    current_node_id = uuid.uuid4()
                                    node_added = False
                                    if i == 0:
                                        current_pose_solutions.append(
                                            (
                                                current_node_id,
                                                current_dynamic_ik,
                                                current_static_ik,
                                            )
                                        )
                                        G.add_node(
                                            current_node_id,
                                            dynamic_ik=current_dynamic_ik,
                                            static_ik=current_static_ik,
                                        )
                                    for (
                                        node_id,
                                        dynamic_ik,
                                        static_ik,
                                    ) in previous_pose_solutions:
                                        dynamic_dists = (
                                            np.abs(current_dynamic_ik - dynamic_ik)
                                            * 180
                                            / pi
                                        ) % 360
                                        static_dists = (
                                            np.abs(current_static_ik - static_ik)
                                            * 180
                                            / pi
                                        ) % 360
                                        dynamic_dists = np.minimum(
                                            dynamic_dists, 360 - dynamic_dists
                                        )
                                        static_dists = np.minimum(
                                            static_dists, 360 - static_dists
                                        )
                                        dists = np.concatenate(
                                            (dynamic_dists, static_dists)
                                        )
                                        if makespan_objective:
                                            dists = np.max(dists)
                                        else:
                                            weight = np.sum(dynamic_dists) + np.sum(
                                                static_dists
                                            )
                                        if weight < delta:
                                            if not node_added:
                                                current_pose_solutions.append(
                                                    (
                                                        current_node_id,
                                                        current_dynamic_ik,
                                                        current_static_ik,
                                                    )
                                                )
                                                G.add_node(
                                                    current_node_id,
                                                    dynamic_ik=current_dynamic_ik,
                                                    static_ik=current_static_ik,
                                                )
                                                node_added = True
                                            trajectory_blocked = False
                                            G.add_edge(
                                                node_id,
                                                current_node_id,
                                                weight=weight,
                                            )
        if i == 0:
            for node_id, _, _ in current_pose_solutions:
                trajectory_blocked = False
                G.add_edge("s", node_id)
        if trajectory_blocked:
            return np.array([]), np.array([]), i, 0, False
        previous_pose_solutions = current_pose_solutions
        current_pose_solutions = []
    for node_id, _, _ in previous_pose_solutions:
        G.add_edge(node_id, "t")
    if makespan_objective:
        min_max_edge_trajectory = nx.dijkstra_path_length(
            G, "s", "t", "weight"
        ), np.array(nx.dijkstra_path(G, "s", "t", "weight"))
    else:
        min_max_edge_trajectory = min_max_edge_dijkstra(G, "s", "t", "weight")
    if min_max_edge_trajectory == None:
        return np.array([]), np.array([]), i, 0, False
    ik_trajectory_max_edge, graph_ik_trajectory = min_max_edge_trajectory
    dynamic_ik_trajectory = []
    static_ik_trajectory = []
    for node_index in range(1, len(graph_ik_trajectory) - 1):
        node = G.nodes[graph_ik_trajectory[node_index]]
        dynamic_ik_trajectory.append(node["dynamic_ik"])
        static_ik_trajectory.append(node["static_ik"])
    dynamic_ik_trajectory = adjust_trajectory_to_joint_limits(
        np.array(dynamic_ik_trajectory)
    )
    static_ik_trajectory = adjust_trajectory_to_joint_limits(
        np.array(static_ik_trajectory)
    )
    return (
        dynamic_ik_trajectory,
        static_ik_trajectory,
        len(placed_trajectory) - 1,
        ik_trajectory_max_edge,
        True,
    )
