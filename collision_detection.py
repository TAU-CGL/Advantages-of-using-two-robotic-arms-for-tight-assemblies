from time import sleep
import pybullet as p
import numpy as np

np.set_printoptions(suppress=True, precision=6)
physics_client = p.connect(p.DIRECT)


def check_self_collision(joint_angles, robot_id):
    # return False
    num_joints = p.getNumJoints(robot_id)
    for i in range(len(joint_angles)):
        p.resetJointState(robot_id, i + 1, joint_angles[i])

    collision = False
    for i in range(len(joint_angles) - 1):
        for j in range(i + 2, num_joints):
            contact_points = p.getClosestPoints(
                robot_id, robot_id, distance=0, linkIndexA=i, linkIndexB=j
            )
            if len(contact_points) > 0:
                collision = True
                break
        if collision:
            break
    return collision


def check_collision_between_robots(joint_angles1, robot_id1, joint_angles2, robot_id2):
    # return False
    for i in range(len(joint_angles1)):
        p.resetJointState(robot_id1, i + 1, joint_angles1[i])

    for i in range(len(joint_angles2)):
        p.resetJointState(robot_id2, i + 1, joint_angles2[i])

    collision = False
    for link1 in range(p.getNumJoints(robot_id1)):
        for link2 in range(p.getNumJoints(robot_id2)):
            contact_points = p.getClosestPoints(
                robot_id1, robot_id2, distance=0, linkIndexA=link1, linkIndexB=link2
            )
            if len(contact_points) > 0:
                collision = True
                break
        if collision:
            break
    # sleep(10)
    return collision


def load_urdf(urdf_path, position=[0, 0, 0], orientation=[0, 0, 0, 1]):
    robot_id = p.loadURDF(urdf_path, position, orientation, useFixedBase=True)
    return robot_id
