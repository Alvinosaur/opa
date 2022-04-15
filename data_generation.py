"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import shutil
from typing import *
import argparse
import random
import multiprocessing
import ipdb

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from data_params import Params
from elastic_band import ElasticBand, Object
import viz_2D

seed = 444
np.random.seed(seed)
random.seed(seed)


def as_symm_matrix(a: np.ndarray) -> np.ndarray:
    """
    Convert 3x1 vector to 3x3 symmetric matrix
    :param a: 3x1 vector
    :return: 3x3 symmetric matrix
    """
    a = a.flatten()
    assert a.shape[0] == 3

    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])


def rand_quat() -> np.ndarray:
    """
    Generate a random quaternion: http://planning.cs.uiuc.edu/node198.html
    :return: quaternion np.ndarray(4,)
    """
    u, v, w = np.random.uniform(0, 1, 3)
    return np.array([np.sqrt(1 - u) * np.sin(2 * np.pi * v),
                     np.sqrt(1 - u) * np.cos(2 * np.pi * v),
                     np.sqrt(u) * np.sin(2 * np.pi * w),
                     np.sqrt(u) * np.cos(2 * np.pi * w)])


def gen_config(num_objects: int, radius_mu: float, radius_std: float, max_tries=200,
               is_rot=False, is_3D=False):
    """
    Generate random environment configuration of start, goal, and object poses and radii

    :param num_objects: number of objects to generate
    :param radius_mu: mean of radius of objects
    :param radius_std: standard deviation of radius of objects
    :param max_tries: maximum number of tries to generate a valid configuration
    :param is_rot: if True, generate orientation-data specific scenario
    :param is_3D:
    :return:
        start_pose: start pose
        goal_pose: goal pose
        objects: objects in the form List[Object]
        object_poses: array of object poses
        object_radii: array of object radii
        goal_radius: radius of goal
    """
    if is_3D:
        lb = Params.lb_3D
        ub = Params.ub_3D
        sample_ori_func = rand_quat

    else:
        lb = Params.lb_2D
        ub = Params.ub_2D
        sample_ori_func = lambda: np.random.uniform(low=0, high=2 * np.pi, size=1)

    # sample start, goal pose and radii
    start_ori, goal_ori = sample_ori_func(), sample_ori_func()
    goal_radius = np.random.normal(loc=radius_mu, scale=radius_std)
    while True:
        # sample start and goal positions to be sufficiently far apart
        min_dist = 0.8 * ub[0]
        start_goal_dist = 0
        while start_goal_dist < min_dist:
            start_pos = np.random.uniform(low=lb, high=ub)
            goal_pos = np.random.uniform(low=lb, high=ub)
            start_goal_dist = np.linalg.norm(start_pos - goal_pos)

        start_pose = (start_pos, start_ori)
        goal_pose = (goal_pos, goal_ori)

        start_goal_vec = (goal_pos - start_pos)
        start_goal_dist = np.linalg.norm(start_goal_vec)
        start_goal_vec /= start_goal_dist

        objects = []
        other_objs_pos = np.array([])
        object_poses = []
        object_radii = np.array([])
        orth_dists = []
        count = 0

        terminated = False
        while len(objects) < num_objects:
            count += 1
            if count > max_tries:
                terminated = True
                break

            # random object radius with a minimum value
            sampled_radius = max(np.random.normal(loc=radius_mu, scale=radius_std), 0.6 * radius_mu)

            # where along the start-goal vector to place the object
            start_goal_prop = np.random.uniform(0.2, 0.8)

            # orthogonal distance from the start-goal vector
            if is_rot:
                # force agent to cross over object and either ignore or care about its orientation
                orth_dist = 0.0
            else:
                orth_dist = np.random.uniform(low=0.1, high=2) * sampled_radius

            # direction of offset from start-goal vector
            if not is_3D:  # 2D
                # place object on either left or right side of start-goal vector
                rot_angle = np.random.choice([-np.pi / 2, np.pi / 2])  # (left, right)
                orth_vec = start_goal_vec @ np.array([
                    [np.cos(rot_angle), -np.sin(rot_angle)],
                    [np.sin(rot_angle), np.cos(rot_angle)]
                ])

            else:
                # find orthogonal vector to start-goal vector
                orth_vec = np.cross(start_goal_vec, np.array([0, 0, 1]))
                if np.allclose(orth_vec, 0, atol=1e-3):
                    orth_vec = np.cross(start_goal_vec, np.array([0, 1, 0]))

                # Axis Angle formulation: rotate about the start_goal_vec "axis" by "angle"
                rand_angle = np.random.uniform(low=0, high=2 * np.pi)
                w_hat = as_symm_matrix(start_goal_vec)
                rot = (np.eye(3) + np.sin(rand_angle) *
                       w_hat + (1 - np.cos(rand_angle)) * w_hat @ w_hat)

                # rotate orth_vec by rand_angle about start_goal_vec
                orth_vec = rot @ orth_vec

            # overall object position along start-goal vec offset by orthogonal component
            obj_pos = start_goal_prop * start_goal_dist * start_goal_vec + start_pos
            obj_pos += orth_dist * orth_vec

            # Skip if too much overlap with other objects
            if len(objects) > 0:
                dists = np.linalg.norm(other_objs_pos[:] - obj_pos, axis=-1)
                intersects = np.where(dists <= 0.8 * (object_radii + sampled_radius))[0]
                if len(intersects) > 0:
                    continue

            # Randomly Sample Object orientation
            obj_ori = sample_ori_func()

            # Add object to list
            obj_pose = (obj_pos, obj_ori)
            objects.append(Object(pos=obj_pos.copy(), radius=sampled_radius, ori=obj_ori.copy()))
            if len(other_objs_pos) == 0:
                other_objs_pos = obj_pos[np.newaxis]
            else:
                other_objs_pos = np.vstack([other_objs_pos, obj_pos[np.newaxis]])
            object_poses.append(obj_pose)
            object_radii = np.append(object_radii, sampled_radius)
            orth_dists.append(orth_dist)

        if not terminated:
            break

    return start_pose, goal_pose, objects, object_poses, object_radii, goal_radius


@ignore_warnings(category=ConvergenceWarning)
def generate_traj_helper(waypoints: np.ndarray, dstep: float):
    """
    Use Gaussian Process to generate a smooth trajectory between waypoints.

    :param waypoints: array of coarse waypoints to generate trajectory between.
    :param dstep: step size between final trajectory waypoints
    :return:
    """
    # define GP
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    var = 0.2
    n_tries = 20
    gp = GaussianProcessRegressor(kernel=kernel, alpha=var ** 2,
                                  n_restarts_optimizer=n_tries)

    # approximate total trajectory length by summing distance between adjacent waypoints
    piecewise_dists = np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=-1)
    total_dist = np.sum(piecewise_dists)
    num_traj_timesteps = int(total_dist / dstep)

    # Fit GP to coarse waypoints
    # NOTE: predicted trajectory starts at time = 1, not time = 0 (start)
    #   so does not include start state. This is why wpt_timesteps is from [0, T]
    #   but traj_timesteps is from [1, T]
    T = 100  # some number > 1
    wpt_timesteps = np.linspace(0, T, num=len(waypoints))
    gp.fit(wpt_timesteps[:, np.newaxis], waypoints)  # time vs (x,y,z)

    # Finely sample points from GP
    traj_timesteps = np.linspace(wpt_timesteps[1], T, num=num_traj_timesteps - 1)
    traj = gp.predict(traj_timesteps[:, np.newaxis])
    return traj


def fill_ori_transition_2D(start: float, end: float, dtheta: float,
                           start_i: int, max_num_steps: int, ori_traj: np.ndarray):
    """
    Given a start and end orientation(theta on xy plane) and a max dtheta, interpolate between
    start and goal in the fewest number of steps, allowing for wraparound.
    Directly modifies traj.

    :param start: start theta
    :param end: end theta
    :param dtheta: max dtheta
    :param start_i: index of start in traj
    :param max_num_steps: max number of steps to interpolate
    :param ori_traj: trajectory to be modified (N x 1 (theta))
    :return: index of final modified step in trajectory
    """
    # Bring start and end to nearest absolute values, ex: -90 is closer to 0 than 270
    start %= 2 * np.pi
    end %= 2 * np.pi
    if abs(start - end) > abs((start - 2 * np.pi) - end):
        start -= 2 * np.pi
    elif abs(start - end) > abs((end - 2 * np.pi) - start):
        end -= 2 * np.pi

    # Find number of steps to interpolate
    decimal_num_steps = abs(start - end) / dtheta
    num_steps = int(min(max_num_steps - 1, np.ceil(decimal_num_steps)))

    # Interpolate between start and end
    # NOTE: dtraj includes 0, so overall traj[start_i] = start no change
    dtraj = np.arange(0, num_steps + 1) * dtheta
    if end < start: dtraj *= -1
    final_i = start_i + num_steps
    res = start + dtraj

    # if overshot, set the last step to end exactly
    if num_steps > decimal_num_steps:
        res[-1] = end

    # if have extra steps remaining, fill the rest with the end theta
    if num_steps < max_num_steps - 1:
        res = np.append(res, [end] * (max_num_steps - 1 - num_steps))
        final_i = start_i + max_num_steps - 1

    # directly modify traj
    ori_traj[start_i:final_i + 1] = res[:, np.newaxis]
    return final_i


def fill_ori_transition_3D(start: np.ndarray, end: np.ndarray,
                           dtheta: float, start_i: int,
                           max_num_steps: int, ori_traj: np.ndarray):
    """
    Given a start and end orientation(quaternion) and a max step_size, interpolate between
    start and goal in the fewest number of steps using Slerp.
    Directly modifies traj.

    :param start: start quaternion
    :param end: end quaternion
    :param dtheta: max dtheta
    :param start_i: index of start in traj
    :param max_num_steps: max number of steps to interpolate
    :param ori_traj: trajectory to be modified  (N x 4 (qx, qy, qz, qw))
    :return: index of final modified step in trajectory
    """
    # if start and end are same, just fill with end
    if np.isclose(np.abs(start @ end), 1.0, atol=1e-5):
        ori_traj[start_i:start_i + max_num_steps] = end
        return

    # Source: https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf, max dist: [0, pi/2]
    dist = np.arccos(np.abs(start @ end))
    true_num_steps = np.ceil(dist / dtheta)
    num_steps = int(min(max_num_steps - 1, true_num_steps))
    final_i = start_i + num_steps

    # Define slerp with scipy and interpolate
    key_times = [0, true_num_steps]
    rotations = R.from_quat(np.vstack([start, end]))
    slerp = Slerp(key_times, rotations)
    times = list(range(num_steps + 1))
    interp_rots = slerp(times)
    res = [rot.as_quat() for rot in interp_rots]  # qx qy qz qw

    # if have extra steps remaining, fill the rest with the end orientation
    if num_steps < max_num_steps - 1:
        res = res + [res[-1]] * (max_num_steps - 1 - num_steps)
        final_i = start_i + max_num_steps - 1

    # directly modify traj
    ori_traj[start_i:final_i + 1] = res
    return final_i


def generate_ori_traj(pos_traj: np.ndarray, start_ori: float,
                      goal_ori: float, goal_radius: float, agent_radius: float,
                      offset_objects: List[Object], object_types, dtheta,
                      is_3D: bool,
                      buffer=0.0):
    """
    Given trajectory of positions, start/goal orientation, and objects,
    generate orientation trajectory in either 2D or 3D space. For each object,
    if we care about its orientation, check position distance to it along trajectory.
    If close enough, then find the point on traj closest to object. Orientation
    interacts with each object in two phases:
        Phase 1: Approaching object: try to match object ori from start of interaction
                 to the nearest point to object
        Phase 2: Leaving object: revert back to start ori

    Finally, try to match the goal orientation when close enough.

    :param pos_traj: trajectory of positions (N x pos_dim)
    :param start_ori: start orientation
    :param goal_ori: goal orientation
    :param goal_radius: radius of goal
    :param offset_objects: list of objects ***WITH ORIENTATION OFFSET ALREADY APPLIED***
    :param object_types: array of object indices (Params.IGNORE_ROT_IDX, Params.CARE_ROT_IDX)
    :param dtheta: change in orientation step size
    :param agent_radius: radius of agent
    :param is_3D: whether to use 3D orientation
    :param buffer: buffer for interacting with nearby objects:
           if dist between agent and obj < sum_radii + buffer, then should interact
    :return: orientation trajectory (N x ori_dim(1 or 4))
    """
    goal_pos = pos_traj[-1]
    ori_traj = np.zeros((len(pos_traj), len(start_ori)))

    fill_ori_transition = fill_ori_transition_3D if is_3D else fill_ori_transition_2D

    # Fill with just initial theta
    ori_traj[:, :] = start_ori

    for obj_i, obj in enumerate(offset_objects):
        obj_ori = obj.ori
        if object_types[obj_i] == Params.IGNORE_ROT_IDX:
            continue

        dists = np.linalg.norm(pos_traj[:] - obj.pos, axis=1)
        active = np.where(dists <= obj.radius + agent_radius + buffer)[0]
        if len(active) > 0:  # if any traj positions are within distance of obj
            first_active = active[0]
            last_active = active[-1]

            # final index to try matching object ori, after target_idx, should revert
            # back to start_ori (phase 2)
            nearest_i = np.argmin(dists)
            target_idx = int((last_active + nearest_i) / 2)

            cur_ori = ori_traj[first_active]
            # ! Phase 1: lin interp from current ori to object ori once within distance
            # either limited by number of active steps or can reach obj theta before nearest_i
            # if nearest_i == first_active: num_steps = min(0+1, ...)
            # even if num_steps == 0, dtraj = [0] so no effect
            final_i2 = fill_ori_transition(start=cur_ori, end=obj_ori, dtheta=dtheta,
                                           start_i=first_active,
                                           max_num_steps=target_idx - first_active + 1,
                                           ori_traj=ori_traj)

            # ! Phase 2: revert back to start_ori
            phase2_len = final_i2 - first_active + 1
            try:
                # try to flip traj back to original
                # copy the flipped version of Phase 2
                if is_3D:
                    # Cannot simply flip the applied quaternions
                    fill_ori_transition(start=ori_traj[final_i2], end=ori_traj[first_active],
                                        dtheta=dtheta,
                                        start_i=final_i2,
                                        max_num_steps=phase2_len,
                                        ori_traj=ori_traj)
                else:
                    ori_traj[final_i2:final_i2 + phase2_len] = np.flip(ori_traj[first_active:final_i2 + 1])

            except:
                # This error happens when object is too close to goal
                # since our model does not look ahead but rather responds locally to object
                # the object's theta takes priority: agent's final theta may not match goal theta in this case
                cur_ori = ori_traj[final_i2]
                fill_ori_transition(start=cur_ori, end=goal_ori, dtheta=dtheta,
                                    start_i=final_i2, max_num_steps=len(ori_traj) - final_i2,
                                    ori_traj=ori_traj)

    # ! Finally move back to goal orientation: only react to goal once close enough
    dists = np.linalg.norm(pos_traj[:] - goal_pos, axis=1)
    active = np.where(dists <= goal_radius + agent_radius + buffer)[0]
    first_active = active[0]
    cur_ori = ori_traj[first_active]
    fill_ori_transition(start=cur_ori, end=goal_ori, dtheta=dtheta,
                        start_i=first_active,
                        max_num_steps=len(ori_traj) - first_active,
                        ori_traj=ori_traj)

    return ori_traj


def generate_trajs(cur_pose: Tuple[np.ndarray, np.ndarray], goal: Tuple[np.ndarray, np.ndarray],
                   goal_radius: float, agent_radius: float,
                   f_internal: float, f_alpha: float,
                   offset_objects: List[Object], object_types: np.ndarray,
                   n_eband_iters: int,
                   dstep: float, dtheta: float,
                   is_3D: bool, is_rot: bool, viz=False) -> np.ndarray:
    """
    Generate either position-specific or orientation-specific trajectories based on is_rot.
    If is_rot is True, position trajectory is simple linear interpolation from cur_pose to goal.
    If is_rot is False, position trajectory is generated by Eband.

    :param cur_pose: current agent pose (pos, ori)
    :param goal: goal pose (pos, ori)
    :param goal_radius: radius of goal
    :param agent_radius: radius of agent
    :param f_internal: internal force weight between eband bubbles
    :param f_alpha: alpha external force weight from objects onto eband bubbles
    :param offset_objects: list of objects ***WITH ORIENTATION OFFSET ALREADY APPLIED***
    :param object_types: array of object types (0=repel pos/ignore ori, 1=attract pos/care about ori)
    :param n_eband_iters: number of eband iterations
    :param dstep: step size for position
    :param dtheta: step size for orientation
    :param is_3D:
    :param is_rot: whether to generate orientation-specific trajectories
    :param viz: whether to visualize
    :return: trajectory (N x pos_dim+ori_dim)
    """
    if is_rot:
        # position is straight linear interp between start and goal
        # no "attract" or "repel" to objects
        total_dist = np.linalg.norm(goal[0] - cur_pose[0])
        num_traj_timesteps = int(total_dist / dstep)
        pos_traj = np.linspace(cur_pose[0], goal[0], num_traj_timesteps)

        ori_traj = generate_ori_traj(pos_traj=pos_traj, offset_objects=offset_objects, start_ori=cur_pose[1],
                                     goal_ori=goal[1], dtheta=dtheta, agent_radius=agent_radius,
                                     buffer=Params.buffer_rot,
                                     goal_radius=goal_radius, object_types=object_types,
                                     is_3D=is_3D)

    else:
        # use Eband to generate position data trajectory
        object_alphas = Params.object_alphas[object_types]
        eband = ElasticBand(start=cur_pose[0], end=goal[0], agent_radius=agent_radius,
                            f_internal=f_internal, f_alpha=f_alpha,
                            objects=offset_objects, object_alphas=object_alphas,
                            buffer=Params.buffer_pos)  # use the true object alphas

        if viz:
            # visualize the elastic band 0th iteration
            all_pts = np.array([bubble.pos for bubble in eband.bubbles])
            pos_traj = generate_traj_helper(waypoints=all_pts, dstep=dstep)
            eband.visualize(title="Elastic Band Iteration 0", traj=pos_traj)

        for t in range(n_eband_iters):
            eband.update()

            if viz:
                all_pts = np.array([bubble.pos for bubble in eband.bubbles])
                pos_traj = generate_traj_helper(waypoints=all_pts, dstep=dstep)
                eband.visualize("Elastic Band Iteration %d" % (t + 1), traj=pos_traj)

        all_pts = np.array([bubble.pos for bubble in eband.bubbles])
        pos_traj = generate_traj_helper(waypoints=all_pts, dstep=dstep)

        # orientation trajectory is not used
        ori_dim = 4 if is_3D else 1
        ori_traj = np.zeros((len(pos_traj), ori_dim))
        ori_traj[:, -1] = 1  # for valid quaternion, overall doesn't matter

    return np.hstack([pos_traj, ori_traj])


def gen_training_sample_expert(object_types: np.ndarray,
                               object_radii_mu: float, object_radii_std: float,
                               is_rot: bool,
                               is_3D: bool):
    """
    High level sample generator for training.
    First sample random env scenario (start, goal, objects).
    Second apply rotation offsets to sampled object orientations, and use this
        along with positions to generate trajectory.

    :param object_types: array of object types (0=repel pos/ignore ori, 1=attract pos/care about ori)
    :param object_radii_mu: mean of object radii
    :param object_radii_std: std of object radii
    :param is_rot: whether to generate orientation-specific trajectories
    :param is_3D: whether to use 3D
    :return: dictionary containing relevant data
    """
    dstep = Params.dstep_3D if is_3D else Params.dstep_2D
    possible_ori_offsets = Params.ori_offsets_3D if is_3D else Params.ori_offsets_2D

    # Generate random environment scenario (start, goal, objects)
    start_pose, goal_pose, objects, object_poses, object_radii, goal_radius = gen_config(
        radius_mu=object_radii_mu, radius_std=object_radii_std, is_3D=is_3D, is_rot=is_rot,
        num_objects=len(object_types))

    # Apply orientation offset to objects to only generate expert traj
    # NOTE: at training, agent only observes the original, unmodified orientation
    #   of each object but still must predict the expert traj that used modified orientations
    #   forcing the agent to learn the correct offset
    for i, obj_type in enumerate(object_types):
        if is_3D:
            # Compose quaternions handled by scipy
            # NOTE: R * R_offset (right-multiply), order matters!!
            #   right-multiply results in a rotation offset "relative" to the original orientation
            #   left-multiply results in a rotation offset "relative" to the world/base/spatial frame
            #       which usually is not what we want
            objects[i].ori = (R.from_quat(objects[i].ori) *
                              R.from_quat(possible_ori_offsets[obj_type])).as_quat()
        else:
            # Here order does not matter: rotation is always about the
            # fixed z-axis (shared by both objects and the world frame)
            objects[i].ori += possible_ori_offsets[obj_type]

    expert_traj = generate_trajs(cur_pose=start_pose, goal=goal_pose,
                                 agent_radius=Params.agent_radius, f_internal=Params.f_internal,
                                 f_alpha=Params.f_alpha,
                                 offset_objects=objects, object_types=object_types,
                                 n_eband_iters=Params.n_eband_iters,
                                 dstep=dstep, dtheta=Params.dtheta, is_3D=is_3D, is_rot=is_rot,
                                 goal_radius=goal_radius)  # , viz=True)

    # NOTE: use "object_poses", not "objects", because "objects" var is modified by ori offset
    object_poses = np.vstack([np.hstack(obj_pose) for obj_pose in object_poses])

    return dict(states=expert_traj, object_poses=object_poses, object_types=object_types,
                object_radii=object_radii, goal_radius=goal_radius,
                is_rot=is_rot, is_3D=is_3D)


def gen_batch_samples_debug(is_rot: bool, is_3D: bool):
    if is_3D:
        object_radii_mu = Params.object_radii_mu_3D
        object_radii_std = Params.object_radii_std_3D
    else:
        object_radii_mu = Params.object_radii_mu_2D
        object_radii_std = Params.object_radii_std_2D

    # generate training samples
    samples = {"all_samples": []}

    # 10 random configurations
    for i in range(10):
        if is_rot:
            object_types = np.random.choice([Params.IGNORE_ROT_IDX, Params.CARE_ROT_IDX], size=1)
        else:
            num_objects = 2
            # num_objects = 8  # ablation study for scalability with # objects
            object_types = ([Params.REPEL_IDX] * (num_objects // 2) +
                            [Params.ATTRACT_IDX] * (num_objects // 2))
            np.random.shuffle(object_types)

        results = gen_training_sample_expert(object_radii_mu=object_radii_mu,
                                             object_radii_std=object_radii_std,
                                             object_types=object_types,
                                             is_3D=is_3D, is_rot=is_rot)

        samples["all_samples"].append(results)

    np.savez(f"{Params.data_root}/debug_data.npz", **samples)


def gen_train_test_sample(args: Tuple[int, bool, bool, str]):
    """
    Generate both a train and test sample, store into file and folder
    specified by input
    :param: args(tuple):
        sample_id: unique index of this sample
        is_rot: whether to generate orientation-specific data
        is_3D: whether to generate 2D or 3D data
        data_folder: path to data folder

    :return:
    """
    sample_id, is_rot, is_3D, data_folder = args
    # Get data parameters for 2D or 3D scenario
    if is_3D:
        object_radii_mu = Params.object_radii_mu_3D
        object_radii_std = Params.object_radii_std_3D
    else:
        object_radii_mu = Params.object_radii_mu_2D
        object_radii_std = Params.object_radii_std_2D

    if is_rot:
        # Rotation data only uses one object in a sample to avoid confounding influence
        # from multiple objects
        # NOTE: repel_idx == ignore_rot_idx, attract_idx == care_rot_idx
        #   so only need one object type to represent both position and orientation
        object_types = np.random.choice([Params.IGNORE_ROT_IDX, Params.CARE_ROT_IDX], size=1)
    else:
        # Position data uses an attractor and repulsor object
        object_types = np.array([Params.REPEL_IDX, Params.ATTRACT_IDX])

    # generate training sample
    train_sample = gen_training_sample_expert(object_radii_mu=object_radii_mu,
                                              object_radii_std=object_radii_std,
                                              object_types=object_types,
                                              is_rot=is_rot, is_3D=is_3D)
    np.savez(f"{data_folder}_train/" + "traj_%d" % sample_id, **train_sample)

    test_sample = gen_training_sample_expert(object_radii_mu=object_radii_mu,
                                             object_radii_std=object_radii_std,
                                             object_types=object_types,
                                             is_rot=is_rot, is_3D=is_3D)
    np.savez(f"{data_folder}_test/" + "traj_%d" % sample_id, **test_sample)
    print("Finished %d" % sample_id)


def data_collect(data_name: str, is_rot: bool, is_3D: bool, num_trajs=3000):
    """
    Generate training and test data for 2D or 3D scenario and for rotation or position data.
    Spawn multiple processes (configurable, see num_workers) to generate data in parallel.

    :param data_name: name of data to define data folder path
    :param is_rot: whether to generate rotation data
    :param is_3D: whether to generate 2D or 3D data
    :param num_trajs: how many samples to generate
    :return:
    """
    start_time = time.time()
    data_folder = f"{Params.data_root}/{data_name}"
    os.mkdir(f"{data_folder}_train/")
    os.mkdir(f"{data_folder}_test/")

    # Save scripts used for data generation for reference
    shutil.copy(f"data_generation.py", f"{data_folder}_train/data_generation.py")
    shutil.copy(f"data_params.py", f"{data_folder}_train/data_params.py")
    shutil.copy(f"elastic_band.py", f"{data_folder}_train/elastic_band.py")

    # Spawn pool of tasks(samples to generate) and let processes execute them
    args = [(i, is_rot, is_3D, data_folder) for i in range(num_trajs)]
    num_workers = max(1, multiprocessing.cpu_count() - 3)  # save some threads for normal laptop use
    with multiprocessing.Pool(num_workers) as p:
        p.map(gen_train_test_sample, args)

    print(f"Time taken with {num_workers} processes: {time.time() - start_time}s")


def visualize_data_2D(data_name: str, use_debug_data=False, num_samples=30, show_rot=False):
    """
    Visualize only ground truth data sample for 2D scenario, no policy.
    Loop through num_samples randomly selected samples, load data, and
    plot static trajectory and objects.

    :param data_name: name of data to define data folder path
    :param use_debug_data: whether to use debug data
    :param num_samples: number of samples to visualize
    :param show_rot: whether to show rotation data
    :return:
    """
    train_folder = f"{Params.data_root}/{data_name}_train/"
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(num_samples):
        if use_debug_data:
            rand_file = f"{Params.data_root}/debug_data.npz"

            data = np.load(rand_file, allow_pickle=True)
            traj_idx = min(i, len(data["all_samples"]) - 1)
            print("traj_idx:", traj_idx)
            data = data["all_samples"][traj_idx]
        else:
            files = os.listdir(train_folder)
            files = [f for f in files if f.endswith(".npz")]
            rand_file = os.path.join(train_folder, random.choice(files))
            data = np.load(rand_file, allow_pickle=True)

        expert_traj = data["states"]
        object_poses = data["object_poses"]
        object_types = data["object_types"]
        object_radii = data["object_radii"]
        goal_radius = data["goal_radius"]
        show_rot = data["is_rot"]
        start_pose = expert_traj[0]
        goal_pose = expert_traj[-1]

        ori_offsets = Params.ori_offsets_2D[object_types]
        objects = [
            Object(pos=obj_pose[0:2], radius=obj_rad,
                   ori=obj_pose[-1] + offset)
            for (obj_pose, obj_rad, offset) in zip(object_poses, object_radii, ori_offsets)
        ]

        viz_2D.draw(ax=ax, object_types=object_types,
                    agent_radius=Params.agent_radius, offset_objects=objects,
                    pred_traj=expert_traj,
                    expert_traj=expert_traj, title="Expert Trajectory",
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    goal_radius=goal_radius,
                    show_rot=show_rot, hold=True)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', action='store', type=str, help="name for data folder")
    parser.add_argument('--num_samples', action='store', type=int, default=3000,
                        help="how many samples to generate for full dataset")
    parser.add_argument('--is_rot', action='store_true', help="generate orientation-specific data")
    parser.add_argument('--visualize', action='store_true', help="visualize data")
    parser.add_argument('--is_debug', action='store_true', help="visualize/generate debug data")
    parser.add_argument('--is_3D', action='store_true', help="generate 3D data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists(Params.data_root):
        os.mkdir(Params.data_root)

    if args.visualize:
        visualize_data_2D(args.data_name, use_debug_data=args.is_debug, num_samples=10)
    elif args.is_debug:
        gen_batch_samples_debug(is_rot=args.is_rot, is_3D=args.is_3D)
    else:
        assert args.data_name, "Need to specify --data_name!"
        data_collect(args.data_name, is_rot=args.is_rot, is_3D=args.is_3D)
