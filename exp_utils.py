"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import numpy as np
import ipdb

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from pybullet_tools.kuka_primitives import BodyConf, BodyPath, Command
from pybullet_tools.utils import *


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
