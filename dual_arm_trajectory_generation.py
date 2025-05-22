import argparse
import json
from math import pi
import math
import os
import numpy as np
from time import time

from collision_detection import load_urdf
from trajectory_ik import generate_dual_arm_ik_of_trajectory
from utils.parse_config import parse_config_dual_arm
from itertools import product

from utils.trajectory import get_alternating_trajectories

NUM_DECISION_POINTS = 5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path to the trajectory's config file")
    parser.add_argument(
        "config_file", type=str, help="Path to the trajectory's config file"
    )
    return parser.parse_args()


def save_experiment_report(
    no_of_trajectories: int,
    trajectories_attempted: int,
    times: list[float],
    deltas: list[float],
    dynamic_deltas: list[float],
    initial_poses: list[float],
    output_dir: str,
):
    experiment_report = {
        "number_of_trajectories": no_of_trajectories,
        "success_rate": no_of_trajectories / trajectories_attempted,
        "average_time": np.mean(times),
        "average_delta": np.mean(deltas),
        "average_dynamic_delta": np.mean(dynamic_deltas),
        "best_delta": np.min(deltas),
        "best_dynamic_delta": np.min(dynamic_deltas),
    }

    with open(os.path.join(output_dir, "experiment_report.json"), "w") as json_file:
        json.dump(experiment_report, json_file, indent=4)

    np.savetxt(
        os.path.join(output_dir, "initial_poses.csv"),
        np.array(initial_poses),
        delimiter=",",
        fmt="%.8f",
    )


if __name__ == "__main__":
    args = parse_arguments()
    (
        trajectory_file,
        static_part_trajectory_file,
        dynamic_part_arm_urdf_path,
        static_part_arm_urdf_path,
        dynamic_part_arm_dh,
        static_part_arm_dh,
        position_factor,
        theta_offsets,
        dynamic_ref_point_relative_pose,
        static_ref_point_relative_pose,
        test_AB,
        dynamic_part_arm_position,
        static_part_arm_position,
        static_part_relative_position,
        static_part_relative_rotation,
    ) = parse_config_dual_arm(args.config_file)
    output_ts_dir_path = os.path.join(
        "./paths/outputs/industrial_06397/hausdorff",
        "5",
    )

    if not os.path.exists(output_ts_dir_path):
        os.makedirs(output_ts_dir_path, exist_ok=True)
    trajectory = np.genfromtxt(
        trajectory_file,
        delimiter=",",
        skip_header=1,
    )
    start_time = time()
    no_of_trajectories = 100
    times = []
    deltas = []
    dynamic_deltas = []
    initial_poses = []
    trajectories_attempted = 0
    failure_steps = np.zeros(len(trajectory) + 1)
    dynamic_robot_id = load_urdf(
        dynamic_part_arm_urdf_path,
        dynamic_part_arm_position,
    )
    static_robot_id = load_urdf(
        static_part_arm_urdf_path,
        static_part_arm_position,
    )

    original_dynamic_trajectory = np.genfromtxt(
        trajectory_file,
        delimiter=",",
        skip_header=0,
    )
    original_static_trajectory = np.genfromtxt(
        static_part_trajectory_file,
        delimiter=",",
        skip_header=0,
    )
    alternating_trajectories = {}
    for indices in product([0, 1, 2], repeat=NUM_DECISION_POINTS):
        print(f"Generating alternating trajectory for indices: {indices}")
        dynamic_trajectory, static_trajectory = get_alternating_trajectories(
            [original_dynamic_trajectory, original_static_trajectory],
            list(indices),
        )
        alternating_trajectories[indices] = (dynamic_trajectory, static_trajectory)
    while len(times) < no_of_trajectories:
        trajectories_attempted += 1
        trajectory_output_dir_path = os.path.join(
            output_ts_dir_path, f"trajectory_{len(times) + 1}"
        )
        if not os.path.exists(trajectory_output_dir_path):
            os.makedirs(trajectory_output_dir_path, exist_ok=True)
        initial_pose = [
            (np.random.rand() - 0.5) * 1,
            (np.random.rand() - 1),
            (np.random.rand() * 0.7 + 0.2),
            (np.random.rand() - 0.5) * 2 * pi,
            (np.random.rand() - 0.5) * 2 * pi,
            (np.random.rand() - 0.5) * 2 * pi,
        ]
        dynamic_trajectory_output_path = os.path.join(
            trajectory_output_dir_path, "dynamic_ik_trajectory.csv"
        )
        static_trajectory_output_path = os.path.join(
            trajectory_output_dir_path, "static_ik_trajectory.csv"
        )
        overall_ik_success = False
        min_delta = math.inf
        min_delta_indices = None
        dynamic_delta = math.inf
        max_step_reached = -1
        best_dynamic_part_ik = None
        best_static_part_ik = None
        failed_prefixes = []
        for indices in product([0, 1, 2], repeat=NUM_DECISION_POINTS):
            period_length = original_dynamic_trajectory.shape[0] // len(indices)
            did_prefix_fail = False
            for i in range(1, NUM_DECISION_POINTS):
                if indices[:i] in failed_prefixes:
                    did_prefix_fail = True
                    break
            if did_prefix_fail:
                continue
            dynamic_trajectory, static_trajectory = alternating_trajectories[indices]
            dynamic_part_ik, static_part_ik, step_reached, delta, ik_success = (
                generate_dual_arm_ik_of_trajectory(
                    dynamic_trajectory,
                    static_trajectory,
                    dynamic_part_arm_dh,
                    static_part_arm_dh,
                    initial_pose,
                    static_part_arm_position,
                    dynamic_part_arm_position,
                    static_part_relative_position,
                    static_part_relative_rotation,
                    position_factor,
                    dynamic_ref_point_relative_pose,
                    static_ref_point_relative_pose,
                    theta_offsets,
                    dynamic_robot_id,
                    static_robot_id,
                )
            )
            print(indices, step_reached)
            if step_reached >= max_step_reached:
                max_step_reached = step_reached
            if ik_success:
                overall_ik_success = True
                if np.max(indices) == np.min(indices) == 0:
                    dynamic_delta = delta
                if delta < min_delta:
                    min_delta = delta
                    min_delta_indices = indices
                    best_dynamic_part_ik = dynamic_part_ik
                    best_static_part_ik = static_part_ik
                    print("new best delta", indices, delta)
            else:
                if np.max(indices) == np.min(indices) == 0:
                    break
                failed_period = step_reached // period_length + 1
                failed_prefixes.append(indices[:failed_period])
        if overall_ik_success:
            trajectory_report = {
                "time": time() - start_time,
                "initial_pose": initial_pose,
                "dynamic_delta": dynamic_delta,
                "best_delta": min_delta,
                "best_delta_indices": min_delta_indices,
            }
            print(
                f"Finished trajectory No. {len(times) + 1}. Got the following report:"
            )
            print(trajectory_report)
            np.savetxt(
                dynamic_trajectory_output_path,
                np.array(best_dynamic_part_ik),
                delimiter=",",
                fmt="%.5f",
            )
            np.savetxt(
                static_trajectory_output_path,
                np.array(best_static_part_ik),
                delimiter=",",
                fmt="%.5f",
            )
            with open(
                os.path.join(trajectory_output_dir_path, "report.json"), "w"
            ) as json_file:
                json.dump(trajectory_report, json_file, indent=4)

            times.append(time() - start_time)
            initial_poses.append(initial_pose)
            deltas.append(min_delta)
            dynamic_deltas.append(dynamic_delta)
            print(f"succeeded. Best delta: {min_delta}. Best so far: {np.min(deltas)}")
            start_time = time()
        else:
            print(max_step_reached)
            failure_steps[max_step_reached] += 1

    save_experiment_report(
        no_of_trajectories,
        trajectories_attempted,
        times,
        deltas,
        dynamic_deltas,
        initial_poses,
        output_ts_dir_path,
    )
    np.save(
        os.path.join(output_ts_dir_path, "failure_steps.npy"),
        failure_steps,
    )
