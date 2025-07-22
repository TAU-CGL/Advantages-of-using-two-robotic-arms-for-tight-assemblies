import uuid
import numpy as np
from math import pi
from typing import Tuple, List
from numpy.typing import NDArray
import networkx as nx

from collision_detection import check_collision_between_robots, check_self_collision
from utils.get_static_pose import get_static_pose
from utils.ik import get_inverse_kinematics
from utils.trajectory import (
    adjust_trajectory_to_joint_limits,
    place_trajectory,
    pose_to_matrix,
    matrix_to_pose,
    get_relative_pose,
)
from utils.utils import (
    euler_to_matrix,
    get_euler_angles_from_matrix,
    get_transformation_matrix,
    min_max_edge_dijkstra,
)
from utils.fk import forward_kinematics

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


# the relative path should be the poses of the dynamic arm's end effector relative to the static arm's end effector,
# meaning both are after considering the reference point
def generate_balanced_dual_arm_ik_of_trajectory(
    relative_trajectory: NDArray[np.float64],
    dynamic_initial_pose: List[float],
    dynamic_branch: Tuple[int, ...],
    static_branch: Tuple[int, ...],
    dh_static: Tuple[float, ...],
    dh_dynamic: Tuple[float, ...],
    static_part_ref_point: NDArray[np.float64],
    static_part_arm_position: NDArray[np.float64],
    dynamic_part_arm_position: NDArray[np.float64],
    static_part_relative_position: NDArray[np.float64],
    static_part_relative_rotation: NDArray[np.float64],
    theta_offsets: NDArray[np.float64],
    dynamic_robot_id: str,
    static_robot_id: str,
    test_AB=False,
    makespan_objective=False,
    delta=10,
) -> Tuple[NDArray, NDArray, int, float, bool]:
    current_dynamic_pose = np.array(dynamic_initial_pose)
    # Step 1: Find IK of the dynamic part in the dynamic_initial_pose and the provided branch
    dynamic_initial_T = get_transformation_matrix(
        dynamic_initial_pose[0],
        dynamic_initial_pose[1],
        dynamic_initial_pose[2],
        dynamic_initial_pose[3],
        dynamic_initial_pose[4],
        dynamic_initial_pose[5],
    )

    dynamic_initial_ik = get_inverse_kinematics(
        dh_dynamic,
        dynamic_initial_T,
        dynamic_branch[0],
        dynamic_branch[1],
        dynamic_branch[2],
        theta_offsets,
    )

    if dynamic_initial_ik is None:
        # print("1")
        return np.array([]), np.array([]), 0, 0, False

    dynamic_initial_ik = np.array(dynamic_initial_ik)

    # Check for self-collision
    if check_self_collision(dynamic_initial_ik, dynamic_robot_id):
        # print("2")
        return np.array([]), np.array([]), 0, 0, False

    # Step 2: Find the initial static pose when dynamic is at dynamic_initial_pose
    static_relative_pose = np.array(
        get_relative_pose(
            relative_trajectory[0],
            np.zeros_like(relative_trajectory[0]),
        )
    )

    static_initial_pose = get_static_pose(
        current_dynamic_pose,
        static_part_arm_position,
        dynamic_part_arm_position,
        static_relative_pose[:3],
        static_relative_pose[3:],
    )

    static_initial_pose = place_trajectory(
        np.array([static_initial_pose]),
        static_initial_pose,
        1,
        static_part_ref_point,
    )[0]

    # Step 3: Compute IK of the static part in the pose from previous step and the provided branch
    static_initial_T = get_transformation_matrix(
        static_initial_pose[0],
        static_initial_pose[1],
        static_initial_pose[2],
        static_initial_pose[3],
        static_initial_pose[4],
        static_initial_pose[5],
    )

    static_initial_ik = get_inverse_kinematics(
        dh_static,
        static_initial_T,
        static_branch[0],
        static_branch[1],
        static_branch[2],
        theta_offsets,
    )

    if static_initial_ik is None:
        # print("3")
        return np.array([]), np.array([]), 0, 0, False

    static_initial_ik = np.array(static_initial_ik)

    # Check for self-collision
    if check_self_collision(static_initial_ik, static_robot_id):
        # print("4")
        return np.array([]), np.array([]), 0, 0, False

    # Check for collision between robots
    if check_collision_between_robots(
        dynamic_initial_ik, dynamic_robot_id, static_initial_ik, static_robot_id
    ):
        # print("5")
        return np.array([]), np.array([]), 0, 0, False

    # Step 4: Keep the IK solutions as the current configuration for each arm
    current_dynamic_ik = dynamic_initial_ik.copy()
    current_static_ik = static_initial_ik.copy()

    # Initialize trajectories
    dynamic_trajectory = [current_dynamic_ik]
    static_trajectory = [current_static_ik]
    total_movement = 0.0

    # Step 5: For each step in relative_trajectory
    for i in range(1, len(relative_trajectory)):
        # Step 5.1: Find the next pose that the dynamic part should be in
        # Get current dynamic pose using forward kinematics

        # Apply the relative trajectory step to get the target pose
        target_dynamic_pose = place_trajectory(
            np.array(relative_trajectory[i - 1 : i + 1]),
            current_dynamic_pose,
            1,
            np.zeros(6),
        )[1]

        # Convert to transformation matrix
        target_dynamic_T = get_transformation_matrix(
            target_dynamic_pose[0],
            target_dynamic_pose[1],
            target_dynamic_pose[2],
            target_dynamic_pose[3],
            target_dynamic_pose[4],
            target_dynamic_pose[5],
        )

        # Step 5.2: Perform IK on this pose for every possible branch (8 options)
        best_solution = None
        best_dynamic_pose = None
        min_movement = float("inf")

        for ratio in [0.25, 0.5, 0.75]:
            for shoulder in [0, 1]:
                for wrist in [0, 1]:
                    for elbow in [0, 1]:  # Consider both elbow up and down
                        # Get IK solution for dynamic arm
                        original_dynamic_ik_solution = get_inverse_kinematics(
                            dh_dynamic,
                            target_dynamic_T,
                            shoulder,
                            wrist,
                            elbow,
                            theta_offsets,
                        )

                        if original_dynamic_ik_solution is None:
                            continue

                        original_dynamic_ik_solution = np.array(
                            original_dynamic_ik_solution
                        )

                        # Check self-collision for dynamic arm

                        # Step 5.3: For each such IK solution
                        # Step 5.3.1: Move a way from the current configuration to the solution
                        dynamic_ik_solution = (
                            ratio * original_dynamic_ik_solution
                            + (1 - ratio) * current_dynamic_ik
                        )

                        if check_self_collision(dynamic_ik_solution, dynamic_robot_id):
                            continue
                        # Step 5.3.2: Perform forward kinematics
                        half_way_dynamic_pose = np.array(
                            forward_kinematics(
                                dh_dynamic,
                                dynamic_ik_solution,
                                theta_offsets,
                                return_rpy=True,
                            )
                        )
                        # Step 5.3.3: Find the pose of the static part according to the current step
                        static_relative_pose = np.array(
                            get_relative_pose(
                                relative_trajectory[i],
                                np.zeros_like(relative_trajectory[i]),
                            )
                        )

                        target_static_pose = get_static_pose(
                            half_way_dynamic_pose,
                            static_part_arm_position,
                            dynamic_part_arm_position,
                            static_relative_pose[:3],
                            static_relative_pose[3:],
                        )

                        target_static_pose = place_trajectory(
                            np.array([target_static_pose]),
                            target_static_pose,
                            1,
                            static_part_ref_point,
                        )[0]

                        # Convert to transformation matrix
                        target_static_T = get_transformation_matrix(
                            target_static_pose[0],
                            target_static_pose[1],
                            target_static_pose[2],
                            target_static_pose[3],
                            target_static_pose[4],
                            target_static_pose[5],
                        )

                        # Step 5.3.4: Perform IK to the static part for every possible branch (8 options)
                        for static_shoulder in [0, 1]:
                            for static_wrist in [0, 1]:
                                for static_elbow in [
                                    0,
                                    1,
                                ]:  # Consider both elbow up and down
                                    static_ik_solution = get_inverse_kinematics(
                                        dh_static,
                                        target_static_T,
                                        static_shoulder,
                                        static_wrist,
                                        static_elbow,
                                        theta_offsets,
                                    )

                                    if static_ik_solution is None:
                                        continue

                                    static_ik_solution = np.array(static_ik_solution)

                                    # Check self-collision for static arm
                                    if check_self_collision(
                                        static_ik_solution, static_robot_id
                                    ):
                                        continue

                                    # Check collision between robots
                                    if check_collision_between_robots(
                                        dynamic_ik_solution,
                                        dynamic_robot_id,
                                        static_ik_solution,
                                        static_robot_id,
                                    ):
                                        continue

                                    # Calculate movement cost
                                    dynamic_movement = np.max(
                                        np.abs(dynamic_ik_solution - current_dynamic_ik)
                                    )
                                    static_movement = np.max(
                                        np.abs(static_ik_solution - current_static_ik)
                                    )
                                    total_movement_cost = max(
                                        dynamic_movement, static_movement
                                    )

                                    # Update best solution if this one has less movement
                                    if (
                                        total_movement_cost < min_movement
                                        and total_movement_cost < np.deg2rad(delta)
                                    ):
                                        min_movement = total_movement_cost
                                        best_solution = (
                                            dynamic_ik_solution,
                                            static_ik_solution,
                                        )
                                        best_dynamic_pose = half_way_dynamic_pose

        # Step 5.4: Pick the solution with least movement and make it the new current configuration
        if best_solution is None:
            # print("6")
            return np.array([]), np.array([]), i, 0, False

        current_dynamic_ik, current_static_ik = best_solution

        # Add to trajectories
        dynamic_trajectory.append(current_dynamic_ik)
        static_trajectory.append(current_static_ik)
        current_dynamic_pose = best_dynamic_pose
        total_movement += min_movement

    # Convert trajectories to numpy arrays
    dynamic_trajectory = np.array(dynamic_trajectory)
    static_trajectory = np.array(static_trajectory)

    # Adjust trajectories to joint limits
    dynamic_trajectory = adjust_trajectory_to_joint_limits(dynamic_trajectory)
    static_trajectory = adjust_trajectory_to_joint_limits(static_trajectory)

    return (
        dynamic_trajectory,
        static_trajectory,
        len(relative_trajectory),
        np.rad2deg(total_movement),
        True,
    )
