"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import numpy as np
import ipdb
import re

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import transforms3d.affines as aff

from pybullet_tools.kuka_primitives import BodyConf, BodyPath, Command
from pybullet_tools.utils import *

from data_params import Params
from elastic_band import Object

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

ROBOT_FRAME = "kinova_base"


class RunningAverage(object):
    def __init__(self, length, init_vals) -> None:
        self.length = length
        self.values = [init_vals] * length
        self.count = 0
        self.avg = init_vals

    def update(self, val):
        self.values[self.count % self.length] = val
        self.count += 1
        self.avg = sum(self.values) / min(self.count, self.length)

def sigint_handler(signal, frame):
    # Force scripts to exit cleanly
    exit()

# Taken from meta cobot 
def ParseCameraExtrinsics(in_path, serial, name):
    '''
    translation xyz: -0.218569 0.208541 0.794479
    quaternion wxyz: 0.502517 0.50733 0.49782 -0.492207
    '''
    infile = os.path.join(in_path, serial, "extrinsics.txt")
    with open(infile, 'r') as f_extrinsic:
        trans_line = f_extrinsic.readline()
        words = trans_line.split(' ')
        tx = float(words[2])
        ty = float(words[3])
        tz = float(words[4])

        q_line = f_extrinsic.readline()
        words = q_line.split(' ')
        w = float(words[2])
        rx = float(words[3])
        ry = float(words[4])
        rz = float(words[5])

    print("Reading extrinsics for camera {} at {}".format(
        name, infile
    ))

    mat = ComposeAffine([tx, ty, tz], [rx, ry, rz, w])

    return mat

# Taken from meta cobot 
def ComposeAffine(trans, quat, Z=np.ones(3)):
    rot_mat = R.from_quat(quat).as_matrix()
    M = aff.compose(trans, rot_mat, Z)
    return M

def pose_to_msg(pose: np.ndarray, frame) -> PoseStamped:
    """
    Convert a pose to a geometry_msgs.msg.PoseStamped() message

    :param pose: Pose (pos_dim + rot_dim)
    :return: geometry_msgs.msg.PoseStamped() message
    """
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = pose[0]
    msg.pose.position.y = pose[1]
    msg.pose.position.z = pose[2]

    msg.pose.orientation.x = pose[3]
    msg.pose.orientation.y = pose[4]
    msg.pose.orientation.z = pose[5]
    msg.pose.orientation.w = pose[6]

    return msg

def msg_to_pose(msg):
    pose = np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    ])
    return pose


def interpolate_rotations(start_quat, stop_quat, alpha):
    key_times = [0, 1]
    rotations = R.from_quat(
        np.vstack([start_quat, stop_quat]))
    slerp = Slerp(key_times, rotations)
    alpha = 0.3   # alpha*cur_ori_quat + alpha*local_target_ori
    interp_rot = slerp([alpha])[0]
    return interp_rot.as_quat()


def plan_step(robot: int, tool_link: int, target_pose: Pose, obstacles: list,
              custom_limits, ik_kwargs, grasp):
    """
    Plan a low-level control trajectory to move the robot to the target pose.

    :param robot: ID of the robot
    :param tool_link: ID of the tool link
    :param target_pose: Target pose
    :param obstacles: List of IDs of the obstacles
    :param custom_limits: Custom joint limits for the IK solver
    :param ik_kwargs: Additional arguments for the IK solver
    :param grasp: Grasp object
    :return:
    """
    # get current joint config of robot
    current_conf = BodyConf(robot)

    # calculate joint config of target pose
    # target pose should be sufficiently close to current pose
    q_approach = inverse_kinematics(robot, tool_link, target_pose,
                                    max_iterations=500,
                                    custom_limits=custom_limits,
                                    **ik_kwargs)

    if q_approach is None:
        # Try one more time. This strangely always works after the first try
        q_approach = inverse_kinematics(robot, tool_link, target_pose,
                                        max_iterations=500,
                                        custom_limits=custom_limits,
                                        **ik_kwargs)
        if q_approach is None:
            raise Exception("IK Failed!")

    if any(pairwise_collision(robot, b) for b in obstacles):
        raise Exception("Collision!")

    # plan_direct_joint_motion(robot idx, joint idxs, Pose, [obs], limits)
    current_conf.assign()
    path = plan_direct_joint_motion(robot, current_conf.joints, q_approach, obstacles=obstacles,
                                    custom_limits=custom_limits, attachments=[grasp.attachment()],
                                    disabled_collisions={grasp.body})
    if path is None:
        raise Exception("Joint Path Failed!")

    # path = refine_path(robot, current_conf.joints, path, num_steps=10)
    return Command([BodyPath(robot, path, attachments=[grasp])])


def load_floor():
    floor_ids = []
    dxs = [-2, 0, 2]
    dys = [-1, 0, 1]
    with HideOutput():
        for dx in dxs:
            for dy in dys:
                floor = load_model('models/short_floor.urdf')
                set_pose(floor, Pose(Point(dx, dy, 0), Euler(0, 0, 0)))
                floor_ids.append(floor)

    return floor_ids

def calc_pose_error(pose_quat1, pose_quat2, pos_scale=1, rot_scale=0.1):
    pose_quat1 = pose_quat1.flatten()
    pose_quat2 = pose_quat2.flatten()
    pos_error = np.linalg.norm(pose_quat1[0:3] - pose_quat2[0:3])
    rot_error = np.arccos(np.abs(pose_quat1[3:] @ pose_quat2[3:]))
    pose_error = pos_scale * pos_error + rot_scale * rot_error
    return pose_error

def command_kinova_gripper(pub, cmd_open):
    msg = Bool(cmd_open)
    for i in range(5):
        pub.publish(msg)
        rospy.sleep(0.1)