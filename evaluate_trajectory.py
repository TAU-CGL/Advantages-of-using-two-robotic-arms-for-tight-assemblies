import math
import numpy as np


def evaluate_trajectory_makespan(dynamic_path: str, static_path: str):
    static_trajectory = np.genfromtxt(
        static_path,
        delimiter=",",
        skip_header=0,
    )
    dynamic_trajectory = np.genfromtxt(
        dynamic_path,
        delimiter=",",
        skip_header=0,
    )
    trajectory_length = 0
    for i in range(1, len(static_trajectory)):
        max_dynamic = max(np.abs(dynamic_trajectory[i] - dynamic_trajectory[i - 1]))
        max_static = max(np.abs(static_trajectory[i] - static_trajectory[i - 1]))
        trajectory_length += max(max_dynamic, max_static)
    return trajectory_length


def print_differences(dynamic_path: str, static_path: str):
    static_trajectory = np.genfromtxt(
        static_path,
        delimiter=",",
        skip_header=0,
    )
    dynamic_trajectory = np.genfromtxt(
        dynamic_path,
        delimiter=",",
        skip_header=0,
    )
    diffs = []
    for i in range(1, len(static_trajectory)):
        dynamic_movement = np.max(
            np.abs(dynamic_trajectory[i] - dynamic_trajectory[i - 1])
        )
        static_movement = np.max(
            np.abs(static_trajectory[i] - static_trajectory[i - 1])
        )
        diffs.append(np.degrees(np.abs(dynamic_movement - static_movement)))
    return np.mean(diffs), np.median(diffs), np.max(diffs), np.min(diffs)


print(
    print_differences(
        "paths/outputs/balanced-guess/abc-lite/50/trajectory_15/dynamic_ik_trajectory.csv",
        "paths/outputs/balanced-guess/abc-lite/50/trajectory_15/static_ik_trajectory.csv",
    ),
    # print_differences(
    #     "paths/outputs/balanced-inequally/abc-lite/50/trajectory_28/dynamic_ik_trajectory.csv",
    #     "paths/outputs/balanced-inequally/abc-lite/50/trajectory_28/static_ik_trajectory.csv",
    # ),
)
