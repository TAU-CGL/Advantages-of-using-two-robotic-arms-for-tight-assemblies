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


print(
    np.degrees(
        evaluate_trajectory_makespan(
            "paths/outputs/industrial_06397/makespan/parallel_100/trajectory_84/dynamic_ik_trajectory.csv",
            "paths/outputs/industrial_06397/makespan/parallel_100/trajectory_84/static_ik_trajectory.csv",
        )
    )
)
