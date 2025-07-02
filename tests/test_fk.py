from random import uniform
import numpy as np
import pytest
from math import pi
from utils.fk import forward_kinematics
from utils.ik import get_inverse_kinematics
from utils.utils import get_transformation_matrix, get_euler_angles_from_matrix

# Test DH parameters (example values - adjust based on your robot)
DH_PARAMS = (
    -0.425,
    -0.3922,
    0.1625,
    0.1333,
    0.0997,
    0.2746,
)  # (a_2, a_3, d_1, d_4, d_5, d_6)
FK_THETA_OFFSETS = np.array([pi, 0.0, 0.0, 0.0, 0.0, 0.0])  # Adjust based on your robot
IK_THETA_OFFSETS = np.array([pi, 0.0, 0.0, 0.0, 0.0, 0.0])  # Adjust based on your robot


def normalize_angle(angle):
    """Normalize angle to be between 0 and 2π."""
    return angle % (2 * pi)


def test_forward_kinematics():
    # Test case 1: Home position
    joint_angles = (0, np.deg2rad(350), np.deg2rad(10), 0, np.deg2rad(40), 0)

    # Get end-effector pose using forward kinematics
    end_pose = forward_kinematics(DH_PARAMS, joint_angles, FK_THETA_OFFSETS, True)
    print(end_pose)
    # Convert pose to transformation matrix
    transform = get_transformation_matrix(*end_pose, adjust=False)

    # Try all branch combinations
    best_solution = None
    min_max_diff = float("inf")

    for shoulder in [1]:
        for wrist in [1]:
            for elbow in [1]:
                recovered_angles = get_inverse_kinematics(
                    DH_PARAMS, transform, shoulder, wrist, elbow, IK_THETA_OFFSETS
                )

                if recovered_angles is not None:
                    print(recovered_angles)
                    # Calculate the maximum difference between normalized original and recovered angles
                    max_diff = max(
                        abs(normalize_angle(orig) - normalize_angle(recov))
                        for orig, recov in zip(joint_angles, recovered_angles)
                    )

                    # If this solution is better than our current best, update it
                    if max_diff < min_max_diff:
                        min_max_diff = max_diff
                        best_solution = recovered_angles

    # Assert that we found at least one valid solution
    assert best_solution is not None, "No valid IK solution found for home position"

    # Assert that the best solution is close enough to the original angles
    print(f"Original:   {joint_angles}")
    print(f"Best solution:  {best_solution}")
    print(f"Max angle difference: {min_max_diff}")

    # Check if the solution is close enough (within 1e-6)
    for orig, recov in zip(joint_angles, best_solution):
        diff = min(
            abs(normalize_angle(orig) - normalize_angle(recov)),
            2 * pi - abs(normalize_angle(orig) - normalize_angle(recov)),
        )
        assert (
            diff < 1e-6
        ), f"Angle mismatch: original={orig}, recovered={recov}, diff={diff}"


def test_forward_kinematics_multiple_poses():
    # Test multiple poses
    test_poses = [
        (0, np.deg2rad(350), np.deg2rad(10), 0, np.deg2rad(40), 0),
    ]

    for i in range(100):
        test_poses.append(
            (
                np.deg2rad(uniform(0, 360)),
                np.deg2rad(uniform(0, 360)),
                np.deg2rad(uniform(0, 360)),
                np.deg2rad(uniform(0, 360)),
                np.deg2rad(uniform(0, 360)),
                np.deg2rad(uniform(0, 360)),
            ),
        )

    for joint_angles in test_poses:
        # Get end-effector pose using forward kinematics
        end_pose = forward_kinematics(DH_PARAMS, joint_angles, FK_THETA_OFFSETS, True)

        # Convert pose to transformation matrix
        transform = get_transformation_matrix(*end_pose, adjust=False)

        # Try all branch combinations
        best_solution = None
        min_max_diff = float("inf")

        for shoulder in [0, 1]:
            for wrist in [0, 1]:
                for elbow in [0, 1]:
                    recovered_angles = get_inverse_kinematics(
                        DH_PARAMS, transform, shoulder, wrist, elbow, IK_THETA_OFFSETS
                    )

                    if recovered_angles is not None:
                        # Calculate the maximum difference between normalized original and recovered angles
                        max_diff = max(
                            min(
                                abs(normalize_angle(orig) - normalize_angle(recov)),
                                2 * pi
                                - abs(normalize_angle(orig) - normalize_angle(recov)),
                            )
                            for orig, recov in zip(joint_angles, recovered_angles)
                        )

                        # If this solution is better than our current best, update it
                        if max_diff < min_max_diff:
                            min_max_diff = max_diff
                            best_solution = recovered_angles

        # Assert that we found at least one valid solution
        assert (
            best_solution is not None
        ), f"No valid IK solution found for pose {joint_angles}"

        # Assert that the best solution is close enough to the original angles
        print(f"Original:   {joint_angles}")
        print(f"Best solution:  {best_solution}")
        print(f"Max angle difference: {min_max_diff}")

        assert min_max_diff < 1e-6, f"Angle mismatch for pose {joint_angles}"
