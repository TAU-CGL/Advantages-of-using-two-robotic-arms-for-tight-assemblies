import argparse
import json
from math import pi
import math
import os
import numpy as np
from time import time

from collision_detection import load_urdf
from trajectory_ik import generate_balanced_dual_arm_ik_of_trajectory
from utils.parse_config import parse_config_dual_arm
from itertools import product

from utils.trajectory import get_relative_pose, place_trajectory


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path to the trajectory's config file")
    parser.add_argument(
        "config_file", type=str, help="Path to the trajectory's config file"
    )
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    return parser.parse_args()


def save_experiment_report(
    no_of_trajectories: int,
    trajectories_attempted: int,
    times: list[float],
    makespans: list[float],
    initial_poses: list[float],
    output_dir: str,
):
    experiment_report = {
        "number_of_trajectories": no_of_trajectories,
        "success_rate": no_of_trajectories / trajectories_attempted,
        "average_time": np.mean(times),
        "average_makespan": np.mean(makespans),
        "best_makespan": np.min(makespans),
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
    output_ts_dir_path = args.output_dir

    if not os.path.exists(output_ts_dir_path):
        os.makedirs(output_ts_dir_path, exist_ok=True)
    trajectory = np.genfromtxt(
        trajectory_file,
        delimiter=",",
        skip_header=1,
    )
    start_time = time()
    no_of_trajectories = 50
    times = []
    makespans = []
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
    original_dynamic_trajectory_initial_pose = np.copy(original_dynamic_trajectory[0])
    original_dynamic_trajectory_initial_pose[:3] *= position_factor
    relative_trajectory = place_trajectory(
        original_dynamic_trajectory,
        original_dynamic_trajectory_initial_pose,
        position_factor,
        dynamic_ref_point_relative_pose,
    )
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
        min_makespan = math.inf
        max_step_reached = -1
        best_dynamic_part_ik = None
        best_static_part_ik = None
        for indices in product([0, 1], repeat=6):
            (
                dynamic_part_ik,
                static_part_ik,
                step_reached,
                total_movement,
                ik_success,
            ) = generate_balanced_dual_arm_ik_of_trajectory(
                relative_trajectory,
                initial_pose,
                indices[:3],
                indices[3:],
                dynamic_part_arm_dh,
                static_part_arm_dh,
                static_ref_point_relative_pose,
                static_part_arm_position,
                dynamic_part_arm_position,
                static_part_relative_position,
                static_part_relative_rotation,
                theta_offsets,
                dynamic_robot_id,
                static_robot_id,
            )
            # print(indices, step_reached)
            if step_reached >= max_step_reached:
                max_step_reached = step_reached
            if ik_success:
                overall_ik_success = True
                if total_movement < min_makespan:
                    min_makespan = total_movement
                    best_dynamic_part_ik = dynamic_part_ik
                    best_static_part_ik = static_part_ik
                    print("new best makespan", indices, total_movement)
        if overall_ik_success:
            trajectory_report = {
                "time": time() - start_time,
                "initial_pose": initial_pose,
                "best_makespan": min_makespan,
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
            makespans.append(min_makespan)
            print(
                f"succeeded. Best makespan: {min_makespan}. Best so far: {np.min(makespans)}"
            )
            start_time = time()
        else:
            print(max_step_reached)
            failure_steps[max_step_reached] += 1

    save_experiment_report(
        no_of_trajectories,
        trajectories_attempted,
        times,
        makespans,
        initial_poses,
        output_ts_dir_path,
    )
    np.save(
        os.path.join(output_ts_dir_path, "failure_steps.npy"),
        failure_steps,
    )
