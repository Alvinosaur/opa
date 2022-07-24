"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import numpy as np

# define robot limits and params
# inverse kinematics solver params
ik_kwargs = dict(pos_tolerance=5e-1, ori_tolerance=5e-1)
custom_limits = {
    1: np.array([-355, 355]) * np.pi / 180.0,
    2: np.array([-140, 140]) * np.pi / 180.0,
    3: np.array([-355, 355]) * np.pi / 180.0,
    4: np.array([-140, 140]) * np.pi / 180.0,
    5: np.array([-355, 355]) * np.pi / 180.0,
    6: np.array([-140, 140]) * np.pi / 180.0,
    7: np.array([-355, 355]) * np.pi / 180.0,
}

# scaling factor to transform pybullet simulation to training data scale
Sim2Net = 5.0
POS_DIM = 3

# use of 100% goal radius is not necessary for convergence, and reducing the scale
# allows stronger interactions with nearby objects
goal_radius_scale = 0.5

# Number of adaptation steps for Position and Rotation Networks
num_pos_net_updates = 10
num_rot_net_updates = 20

# Arbitrary orientation for tasks that don't involve orientation
some_ori = np.array([0, 0, 0, 1.])

# Initial robot pose, pose to approach and grasp object
init_joints = (2.604, -0.219, -3.08, 1.415, 0.925, -0.592, 2.330)
approach_pose = (np.array([0., 0., 0.1]), (0.0, 0.0, 0.0, 1.0))
grasp_pose = ((-0.0105, 0.0, 0.10000099912285805),
              (0.7071, 0, -0.7071, 0))
