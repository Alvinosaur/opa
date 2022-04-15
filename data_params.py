"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Params:
    # Elastic Band params
    f_internal = 0.2  # internal cohesion pulling "bubbles" together
    f_alpha = 0.6  # scaling factor applied to collective external + internal forces
    agent_radius = 0.4  # agent represented as circle/sphere
    buffer_pos = 0.2  # extra buffer for object influence on agent
    buffer_rot = 0.2  # extra buffer for object influence on agent
    n_eband_iters = 15  # number of steps for elastic band to converge

    # Training environment sizes
    lb_2D = np.array([0, 0], dtype=int)
    ub_2D = np.array([5, 5], dtype=int)
    lb_3D = np.array([0, 0, 0], dtype=int)
    ub_3D = np.array([5, 5, 5], dtype=int)

    # Training Object Indices
    # NOTE: Assume that objects are correctly distinguished during training (since synthetic data)
    #   These types/indices decide the following type of object:
    #       1. position attract vs repel
    #       2. orientation care about vs ignore
    #       3. true orientation offset for training (fixed across all samples)
    #   Rather than permute all possible combos of [repel, attract, ignore, care]
    #   assume that we ignore orientation of repelled objects which is reasonable
    #   this assumption only applies to data generation, model can freely still learn any combo
    n_train_objects = 2
    IGNORE_ROT_IDX = REPEL_IDX = 0
    CARE_ROT_IDX = ATTRACT_IDX = 1
    object_alphas = np.array([-0.3, 0.3])  # (REPEL_IDX, ATTRACT_IDX)
    ori_offsets_2D = np.array([0, np.pi / 2, 0, 0])  # (IGNORE_ROT_IDX, CARE_ROT_IDX, start, goal)
    ori_offsets_3D = np.vstack([
        # (IGNORE_ROT_IDX)
        # This is arbitrary since rotation is ignored
        [0.0, 0.0, 0.0, 1.0],

        # (CARE_ROT_IDX)
        # arbitrary, nonzero rotation so model must learn something non-trivial
        # NOTE: the below angle offset is actually Gimbal Lock, but we only ever
        #   convert Euler -> Rotation matrix, never the reverse so it's fine
        [R.from_euler("ZYX", [-np.pi / 3, np.pi / 2, 3 * np.pi / 4]).as_quat()],

        # start and goal
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    obj_labels = ["Repel/Ignore", "Attract/Care"]
    obj_colors = ["tab:red", "tab:green"]
    obj_colors_rgb = np.array([(255, 0, 0), (0, 255, 0)])
    goal_color_rgb = (128, 128, 0)  # goal: yellow
    start_color_rgb = (0, 255, 255)  # start: cyan
    agent_color_rgb = (255, 0, 255)  # agent: pink

    # Model always expects the following order of objects/pseudo-objects:
    # [obj1, obj2, ... objN, start, goal]
    START_IDX = -2
    GOAL_IDX = -1

    object_radii_mu_2D = 0.8
    object_radii_std_2D = object_radii_mu_2D / 4
    object_radii_mu_3D = 1.2
    object_radii_std_3D = object_radii_mu_3D / 8

    max_steps = 100
    dstep_2D = dstep_3D = np.linalg.norm(ub_2D) / max_steps
    dtheta = np.pi / 30  # (or pi/2 / 5

    model_root = "saved_model_files"
    data_root = "data"
