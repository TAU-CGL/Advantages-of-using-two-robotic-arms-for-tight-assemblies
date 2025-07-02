import numpy as np
from utils.ik import get_inverse_kinematics
from utils.utils import get_transformation_matrix


print(
    np.rad2deg(
        np.array(
            get_inverse_kinematics(
                (-0.425, -0.3922, 0.1625, 0.1333, 0.0997, 0.2746),
                get_transformation_matrix(
                    0.98725,
                    0.34366,
                    0.1366,
                    np.deg2rad(-90),
                    np.deg2rad(40),
                    np.deg2rad(-180),
                ),
                1,
                1,
                1,
                np.array([0, 0, 0, 0, 0, np.pi]),
            )
        )
    )
)
