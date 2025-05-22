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
