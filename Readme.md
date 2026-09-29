# Dual Arm Assembly Operation

## Configuration File

Run the main script dual_arm_trajectory_generation.py with configuration file:
experiments/dual_arm/\*.json
An experiment config supports the following parameters:

- dynamic_part_arm_dh - Path to the dynamic arm dh parameters json file.
- static_part_arm_dh - Path to the static arm dh parameters json file.
- dynamic_part_trajectory - Path to the dynamic part's trajectory csv file, x,y,z,theta_x,theta_y,theta_z format. This is the free flying trajectory of dynamic part, assuming the static part is in the origin.
- static_part_trajectory - Path to the static part's trajectory, when switching roles with the dynamic part. Calculated once by inverting the dynamic part matrix at each step, and keeping in a file. This is done using `from utils.trajectory import generate_alternative_part_trajectory`.
- dynamic_part_arm_urdf_path - Path to the urdf file of the robot holding the dynamic sub-assembly, for collision detection.
- static_part_arm_urdf_path - Path to the urdf file of the robot holding the static sub-assembly, for collision detection.
- dynamic_ref_point_relative_pose - The pose dynamic arm's TCP relative to the dynamic part's reference point. See "Flow" section for more details.
- static_ref_point_relative_pose - The pose static arm's TCP relative to the static part's reference point. See "Flow" section for more details.
- dynamic_part_arm_position - Location of the base of the dynamic arm in world coordinates.
- static_part_arm_position - Location of the base of the static arm in world coordinates.
- static_part_relative_position - Initial position of the static part relative to the dynamic one.
- static_part_relative_rotation - Initial rotation of the static part relative to the dynamic one.
- theta_offsets - Robotic arm's "home" position adjustments.
- position_factor - Scale factor of the sub-assemblies from the free-flying path planning phase to the robotic arm manipulation path generation.

## Flow

1. Duplicate an existing Coppelia scence, e.g. coppeliaSimScenes/double_ur5_abc_lite_lab_new_urdf.ttt.
2. Load the new free flying models to the scene.
3. Scale those models within the Coppelia to the desired size.
4. Delete the original parts from the scene.
5. Position both parts. The scene includes dummy objects, that represent the end of the grippers (The TCP). Add the part as a decendant of the dummy object in Coppelia, with [0, …0] relative-to-parent pose. Then use the translation command in Coppelia on the part, with "own frame" settings, to move it from the reference point to the grasp point. Now the grasping point is aligned with the fingers. Do this for the static and the dynamic parts.
6. Generate the static trajectory – using `from utils.trajectory import generate_alternative_part_trajectory` on the dynamic trajectory.
7. Retrieve the parts' ref point relative pose – By now making each dummy a decendent of the part and extracting the relative-to-parent pose.
8. Build the configuration file with parameters above.
9. Run the dual_arm_trajectory_generation.py script with the config. It runs the dynamic programming, together with collision detection within pybullet (using the urdf of the robots from config). It searches among various random starting points, and their limitations are described in the code itself in dual_arm_trajectory_generation.py.
10. Coppelia serves only as visualization tool.
    a. In the directory kept in output_ts_dir_path will be put a few folders.
    Folder for each successful placement its best trajectory. Within this
    directory:
    i. Dynamic_ik_trajectory
    ii. Static_ik_trajectory
    iii. Report json – initial placement, Hausdorff max, etc.
    b. Another json file experiment report.
    c. Another file. Csv of all successful starting poses. Can be plotted using utils/plotting/plot_initial_poses.py

11. In the Copellia scene there is a script attached to the "ArmsContainer" object. At its beginning, a path to the IK trajectory for each of the arms
    can be updated.
12. Run the copellia.
13. Similarly run the physical robots. But, there is a difference. Need to change the first joint by 180 degrees the base joint.

## Generating a New Path (Balanced Dual Arm)

### 1. Set the relative pose between the robotic arms

Copy an existing config from `experiments/dual_arm/` (e.g. `physical_dual_arm_alpha_z.json`) and edit the arm placement fields. All poses are expressed in the **dynamic arm's base frame**. That arm is usually kept at the origin.

- `dynamic_part_arm_position` - `[x, y, z]` in meters, normally `[0, 0, 0]`.
- `dynamic_part_arm_rotation` - optional, `[rx, ry, rz]` in degrees, defaults to `[0, 0, 0]`.
- `static_part_arm_position` - `[x, y, z]` in meters of the static arm's base.
- `static_part_arm_rotation` - optional, `[rx, ry, rz]` in degrees of the static arm's base, defaults to `[0, 0, 0]` (bases co-oriented).

Rotations are Euler angles applied in z-y-x order (`R = Rz * Ry * Rx`, see `pose_to_matrix` in `utils/trajectory.py`). If you measure the static base pose on the physical robots, express it relative to the dynamic arm's base. The DH, URDF, trajectory, reference point and `position_factor` fields are described in the "Configuration File" section above.

Note: the arm base rotation is used for the IK computation. Collision detection (`load_urdf` in `collision_detection.py`) currently places each URDF by its position only.

### 2. Run the generation script

```
python balanced_dual_arm_trajectory_generation.py experiments/dual_arm/<your_config>.json paths/key_results/dual_arm/<assembly>/<run_name>
```

The first argument is the config and the second is the output directory, which is created if missing. The script samples random initial poses of the assembly and runs the balanced IK for each one, stopping after `no_of_trajectories` successful placements (set in the script). The sampling region for the initial pose and the IK branches that are tried are also set in the script.

### 3. Find the best path

The output directory contains:

- `trajectory_<N>/` - one folder per successful initial pose, with:
  - `dynamic_ik_trajectory.csv`, `static_ik_trajectory.csv` - joint angles (radians) per step for each arm.
  - `report.json` - `initial_pose`, computation `time` and `best_makespan`.
- `experiment_report.json` - success rate, average time, average and best makespan across all trajectories.
- `initial_poses.csv` - the initial pose of every successful trajectory.
- `failure_steps.npy` - how far failed attempts got along the path.

The best path is the `trajectory_<N>` with the **lowest `best_makespan`** (total joint motion). The console prints it while running (`Best so far: ... at trajectory #N`). To find it afterwards:

```
python -c "import json,glob; r=sorted((json.load(open(f))['best_makespan'],f) for f in glob.glob('paths/key_results/dual_arm/<assembly>/<run_name>/trajectory_*/report.json')); print(r[0])"
```

### 4. Prepare the CSV for the physical robots

The physical robots' base joint is rotated 180 degrees compared to the model, and the static arm's second joint cannot rotate fully toward the negative range. Convert the generated trajectories with:

```
python post_processing.py paths/key_results/dual_arm/<assembly>/<run_name>
```

For every `trajectory_<N>` folder (N = 1..100) this writes `improved_dynamic_ik_trajectory.csv` and `improved_static_ik_trajectory.csv` next to the originals:

1. Joint 1 (base) is shifted by 180 degrees (plus or minus pi, whichever keeps it closer to zero).
2. Joint 2 is shifted by -2*pi if all of its values are positive.

Send the `improved_*` files of the best trajectory to the robots. The originals are kept for the Coppelia simulation (see step 11 in the "Flow" section).
