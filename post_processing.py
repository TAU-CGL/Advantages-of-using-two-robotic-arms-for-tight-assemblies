import os
import sys
import json
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 2)

base_path = r"paths\key_results\dynamic_trajectories"  # adjust if needed
# two_arms_path = r"paths\key_results\dual_arm\comperison"
# two_arms_path = r"paths\key_results\dynamic_trajectories"  # single arm alpha_z
two_arms_path = sys.argv[1] if len(sys.argv) > 1 else "paths/key_results/dual_arm/alphaZ/balanced_new_base"

# this file converts trajectories computed in this project to adopt them to the physical environment
# it reads files of the name static_ik_trajectory.csv (or dynamic_) and precedes them with improved_
# specifically:
#   1) rotate base angle of the arms by 180 degrees to fit the physical robots configuration
#   2) shift the second joint of the "static" arm to the positive range as it cannot make a full rotation toward the negative

def improve_csv(filename):
    # Load the CSV file
    df = pd.read_csv(filename, header=None)

    # Work on the first column (assumed numeric)
    col = df.iloc[:, 0]

    # Find min and max
    col_min = col.min()
    col_max = col.max()

    # Compare distances from zero
    if abs(col_max) > abs(col_min):
        # max is further from zero → subtract π
        df.iloc[:, 0] = col - np.pi
    else:
        # min is further from zero → add π
        df.iloc[:, 0] = col + np.pi

    # Work on the second column (assumed numeric)
    col = df.iloc[:, 1]

    # Find min and max
    col_min = col.min()

    # Compare distances from zero
    if col_min > 0:
        # min is positive → subtract 2 * π
        df.iloc[:, 1] = col - 2 * np.pi

    # Prepare output filename
    dirname, basename = os.path.split(filename)
    improved_name = os.path.join(dirname, "improved_" + basename)

    # Save to new CSV
    df.to_csv(improved_name, index=False, header=False)

    print(f"Saved improved CSV as: {improved_name}")


data = []

for i in range(1, 101):  # trajectories 1 to 100
    # folder_1 = os.path.join(base_path, f"trajectory_{i}")
    # file_path_1 = os.path.join(folder_1, "dynamic_ik_trajectory.csv")
    #
    # if os.path.exists(file_path_1):
    #     first_row_1 = pd.read_csv(file_path_1, header=None).iloc[0].tolist()
    # else:
    #     print(f"⚠️ File not found: {file_path_1}")

    folder = os.path.join(two_arms_path, f"trajectory_{i}")
    for initial in ["static_", "dynamic_"]:
        file_path = os.path.join(folder, initial + "ik_trajectory.csv")
        if os.path.exists(file_path):
            improve_csv(file_path)

    # if os.path.exists(file_path_2):
    #     first_row_2 = pd.read_csv(file_path_2, header=None).iloc[0].tolist()
    #
    # else:
    #     print(f"⚠️ File not found: {file_path_2}")

    # print(f'trajectory {i}:')
    # print(first_row_1)
    # print(first_row_2)




