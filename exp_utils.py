"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import numpy as np
import ipdb

from pybullet_tools.kuka_primitives import BodyConf, BodyPath, Command
from pybullet_tools.utils import *


def sigint_handler(signal, frame):
    # Force scripts to exit cleanly
    exit()


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
