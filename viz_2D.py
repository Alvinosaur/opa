"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import os
import numpy as np
import random
import argparse
from typing import *
import json
import ipdb
import signal

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
from learn2learn import LearnedOptimizer, LearnedOptimizerGroup

from online_adaptation import perform_adaptation, perform_adaptation_rls, perform_adaptation_learn2learn, perform_adaptation_learn2learn_group, write_log, params2str, DEVICE
from elastic_band import Object
from data_params import Params
from model import Policy, PolicyNetwork, pose_to_model_input, decode_ori
from recursive_least_squares.rls import RLS


def sigint_handler(signal, frame):
    # Force scripts to exit cleanly
    exit()


signal.signal(signal.SIGINT, sigint_handler)

seed = 444
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

agent_color = "plum"
expert_color = "purple"
goal_color = "tab:olive"
start_color = "tab:cyan"
FORCE_VEC_SCALE = 3
DRAW_DURATION = 0.03
MAX_DRAW_DURATION = 10000
POS_DIM = 2
ROT_DIM = 2


def draw_vec(pos: np.ndarray, theta_rad: float, radius: float, color: str, width=0.025, **kwargs):
    """
    Draws a vector from pos to pos + radius * [cos(theta_rad), sin(theta_rad)]
    """
    arrow_len = radius
    x0, y0 = pos
    plt.arrow(x0, y0, arrow_len * np.cos(theta_rad), arrow_len * np.sin(theta_rad),
              color=color, length_includes_head=True, width=width, shape='full', **kwargs)


def draw(ax: plt.Axes, start_pose: np.ndarray, goal_pose: np.ndarray,
         goal_radius: float, agent_radius: float,
         object_types: np.ndarray, offset_objects: List[Object],
         pred_traj: np.ndarray, expert_traj: np.ndarray,
         title="", object_forces=None, show_rot=True, show_expert=True, hold=False, save_path=None):
    """
    Draws 2D scene with objects, agent, expert traj, and predicted traj. Also
    optionally draws model-predicted "force" each object exerts on the agent.

    :param ax: matplotlib axis to draw on
    :param start_pose: start pose of agent (3,)
    :param goal_pose: goal pose of agent (3,)
    :param goal_radius: radius of goal
    :param agent_radius: radius of agent
    :param object_types: list of object types
    :param offset_objects: list of objects ***WITH ORIENTATION OFFSET ALREADY APPLIED***
    :param pred_traj: predicted trajectory (N x 3)
    :param expert_traj: expert trajectory (N x 3)
    :param title: title of plot
    :param object_forces: predicted object forces (n_objects x 2)
    :param show_rot: whether to show rotation
    :param show_expert: whether to show expert trajectory
    :param hold: if True, show plot until user closes window, otherwise show for DRAW_DURATION
    :return:
    """
    # Clear axes
    ax.clear()
    ax.set_title(title)

    # Draw agent
    cur_pose = pred_traj[0]
    cur_pos, cur_theta = cur_pose[:POS_DIM], cur_pose[-1]
    ax.add_patch(plt.Circle(cur_pos, color=agent_color,
                 radius=agent_radius, alpha=0.4, label="Agent"))
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], '-', color=agent_color,
            linewidth=2, alpha=1.0, label="Agent")
    if show_rot:
        for i in range(len(pred_traj)):
            draw_vec(pos=pred_traj[i, :POS_DIM], theta_rad=pred_traj[i, -1], radius=agent_radius, color=agent_color,
                     alpha=1.0)

    # Draw goal
    goal_pos, goal_theta = goal_pose[:POS_DIM], goal_pose[-1]
    ax.add_patch(plt.Circle(goal_pos, goal_radius,
                 color=goal_color, alpha=0.4, label="Goal"))
    if show_rot:
        draw_vec(pos=goal_pos, theta_rad=goal_theta,
                 radius=goal_radius, color="black")

    # Draw start
    # NOTE: start_radius is act distance to agent, but for viz, just use fixed radius
    # start_radius = np.linalg.norm(cur_pose[:POS_DIM] - expert_traj[0][:POS_DIM])
    start_radius = agent_radius
    ax.add_patch(plt.Circle(
        start_pose[:POS_DIM], start_radius, color=start_color, alpha=0.4, label="Start"))
    if show_rot:
        draw_vec(pos=start_pose[:POS_DIM], theta_rad=start_pose[-1], radius=start_radius,
                 color="black", alpha=0.6)

    # Draw objects
    ori_offsets = Params.ori_offsets_2D[object_types]
    for i, (obj_i, obj, offset) in enumerate(zip(object_types, offset_objects, ori_offsets)):
        obj_ori = obj.ori.item() if isinstance(obj.ori, np.ndarray) else obj.ori
        ax.add_patch(plt.Circle(obj.pos, obj.radius, color=Params.obj_colors[obj_i],
                                alpha=0.4, label=Params.obj_labels[obj_i]))
        if show_rot:
            draw_vec(pos=obj.pos, theta_rad=obj_ori,
                     radius=obj.radius, color="green", label="Offset Ori")
            # NOTE to get original ori, need to subtract the already-applied offset
            draw_vec(pos=obj.pos, theta_rad=obj_ori - offset,
                     radius=obj.radius, color="black", label="Orig Ori")

    # Draw expert trajectory
    if show_expert:
        ax.plot(expert_traj[:, 0], expert_traj[:, 1], '-', color=expert_color,
                linewidth=2, alpha=1.0, label="Expert")
        if show_rot:
            for i in range(0, len(expert_traj), 5):
                draw_vec(pos=expert_traj[i, :POS_DIM], theta_rad=expert_traj[i, -1], radius=1.0 * agent_radius,
                         color=expert_color, alpha=0.7)

    # Optionally draw contributed position forces from objects/goal onto agent
    if object_forces is not None:
        # Normalize contributed obj forces onto agent as unit vectors and extract attention weights
        # Only visualize first timestep in rollout
        object_forces = object_forces[0]
        vec_mags = np.linalg.norm(object_forces, axis=-1, keepdims=True)
        object_forces /= vec_mags
        obj_weights = vec_mags / vec_mags.sum(0)[:, np.newaxis]

        # Draw force vectors and attention weights
        for i, (obj_force, obj_weight) in enumerate(zip(object_forces, obj_weights)):
            vec = np.vstack([
                cur_pos,  # start
                cur_pos + FORCE_VEC_SCALE * obj_force * obj_weight  # end
            ])
            theta_rad = np.arctan2(
                vec[1, 1] - vec[0, 1], vec[1, 0] - vec[0, 0])
            fake_radius = np.linalg.norm(vec[1] - vec[0])

            # Case on whether this is an object or goal
            if i == len(object_forces) - 1:
                label = "Goal"
                color = goal_color
                obj_pos = goal_pos
            else:
                obj_i = object_types[i]
                label = Params.obj_labels[obj_i]
                color = Params.obj_colors[obj_i]
                obj_pos = offset_objects[i].pos

            draw_vec(pos=pred_traj[0, :POS_DIM], theta_rad=theta_rad, radius=fake_radius,
                     alpha=1, color=color,
                     width=0.05, label=label + " Contribution")
            ax.text(*obj_pos, "(%s) att: %.2f" % (label, obj_weight))

    # Graph settings
    lb, ub = Params.lb_2D, Params.ub_2D
    ax.set_aspect('equal')
    ax.set_xlim(lb[0], ub[0])
    ax.set_xbound(lb[0], ub[0])
    ax.set_xticks(list(range(lb[0], ub[0] + 1)))
    ax.set_ylim(lb[1], ub[1])
    ax.set_ybound(lb[1], ub[1])
    ax.set_yticks(list(range(lb[1], ub[1] + 1)))
    plt.xlabel("X", fontsize=20)
    plt.ylabel("Y", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=20, pad=10)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=15)

    plt.draw()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        if hold:
            plt.pause(2.0)  # Give enough time to plot before ipdb
            ipdb.set_trace()
        else:
            plt.pause(DRAW_DURATION)


def viz_helper(policy: Policy, start: np.ndarray, goal: np.ndarray, goal_radius,
               object_poses: np.ndarray, object_radii: np.ndarray, object_types: np.ndarray,
               objects: List[Object], expert_traj: np.ndarray, ax: plt.Axes, viz_args: dict):
    """
    Runs policy in closed-loop and visualizes its behavior.

    :param policy: Policy to run
    :param start: Start pose (3,)
    :param goal: Goal pose (3,)
    :param goal_radius: Goal radius
    :param object_poses: Object poses (n_objects x 3)
    :param object_radii: Object radii (n_objects,)
    :param object_types: Object types (n_objects,)
    :param objects: List of objects
    :param expert_traj: Expert trajectory (N x 3)
    :param ax: Axes to draw on
    :param viz_args: Extra visualization arguments
    :return:
    """
    # Convert numpy to torch tensors
    # Goal
    goal_rot_radii = torch.tensor([goal_radius], device=DEVICE).view(1, 1)
    goal_tensor = torch.from_numpy(
        pose_to_model_input(goal[np.newaxis])).to(torch.float32).to(DEVICE)

    # Start
    start_tensor = torch.from_numpy(
        pose_to_model_input(start[np.newaxis])).to(torch.float32).to(DEVICE)
    agent_radius_tensor = torch.tensor(
        [Params.agent_radius], device=DEVICE).view(1, 1)
    start_objects = torch.cat(
        [start_tensor, agent_radius_tensor], dim=-1).unsqueeze(1)

    # Object
    object_poses_torch = torch.from_numpy(
        pose_to_model_input(object_poses)).to(torch.float32).to(DEVICE).unsqueeze(0)
    object_radii_torch = torch.from_numpy(object_radii).to(
        torch.float32).to(DEVICE).view(1, -1, 1)
    objects_torch = torch.cat([object_poses_torch, object_radii_torch], dim=-1)
    object_types_tensor = torch.from_numpy(
        object_types).to(torch.long).to(DEVICE).unsqueeze(0)

    cur_pose = np.copy(start)
    global_wpt_idx = 1
    step = 0
    # Closed loop visualizing model rollout and executing only first action
    while (step < viz_args["max_num_steps"] and
           np.linalg.norm(cur_pose[:POS_DIM] - goal[:POS_DIM]) > viz_args["dist_tol"]):
        cur_pose_tensor = torch.from_numpy(pose_to_model_input(
            cur_pose[np.newaxis])).to(torch.float32).to(DEVICE)
        object_forces_rollout = []
        rollout_traj = []
        for k in range(viz_args["rollout_horizon"]):
            # Define "object" inputs into policy
            # Current
            current = torch.cat(
                [cur_pose_tensor, agent_radius_tensor], dim=-1).unsqueeze(1)
            # Goal
            goal_radii = torch.norm(
                goal_tensor[:, 0:POS_DIM] - cur_pose_tensor[:, 0:POS_DIM], dim=-1).unsqueeze(0)
            goal_rot_objects = torch.cat(
                [goal_tensor, goal_rot_radii], dim=-1).unsqueeze(1)
            goal_objects = torch.cat(
                [goal_tensor, goal_radii], dim=-1).unsqueeze(1)
            # Start
            start_rot_radii = torch.norm(start_tensor[:, 0:POS_DIM] - cur_pose_tensor[:, 0:POS_DIM],
                                         dim=-1).unsqueeze(0)
            start_rot_objects = torch.cat(
                [start_tensor, start_rot_radii], dim=-1).unsqueeze(1)

            # Get policy output, form into action
            with torch.no_grad():
                pred_vec, pred_ori, object_forces = policy(current=current,
                                                           start=start_rot_objects,
                                                           goal=goal_objects, goal_rot=goal_rot_objects,
                                                           objects=objects_torch,
                                                           object_indices=object_types_tensor,
                                                           calc_rot=viz_args["calc_rot"],
                                                           calc_pos=viz_args["calc_pos"],
                                                           is_training=True)
            if pred_vec is not None:
                object_forces_rollout.append(object_forces)
                cur_pose_tensor[:, 0:POS_DIM] += pred_vec * Params.dstep_2D
            else:
                try:
                    cur_pose_tensor[:, 0:POS_DIM] = torch.from_numpy(
                        expert_traj[global_wpt_idx + k, 0:POS_DIM]).to(torch.float32).to(DEVICE)
                except IndexError:
                    # no more expert waypoints to view
                    break
            if pred_ori is not None:
                cur_pose_tensor[:, POS_DIM:] = pred_ori

            # Decode model output and store into rollout trajectory
            pred_pose = cur_pose_tensor.detach().cpu().numpy()
            pred_pos = pred_pose[0:1, 0:POS_DIM]
            pred_ori = decode_ori(pred_pose[0:1, POS_DIM:])
            rollout_traj.append(np.hstack([pred_pos, pred_ori]))

        rollout_traj = np.vstack(rollout_traj)
        cur_pose = rollout_traj[0]
        if len(object_forces_rollout) > 0:
            object_forces_rollout = torch.cat(
                object_forces_rollout, dim=0).detach().cpu().numpy()
        else:
            object_forces_rollout = None

        try:
            draw(ax, start_pose=start, goal_pose=goal,
                 goal_radius=goal_radius, agent_radius=Params.agent_radius,
                 object_types=object_types, offset_objects=objects,
                 pred_traj=rollout_traj, expert_traj=expert_traj,
                 object_forces=object_forces_rollout,
                 title="", show_rot=viz_args["calc_rot"])
        except KeyboardInterrupt:
            exit()

        global_wpt_idx = np.argmin(np.linalg.norm(
            cur_pose[:POS_DIM] - expert_traj[:, :POS_DIM], axis=1)) + 1
        global_wpt_idx = min(global_wpt_idx, expert_traj.shape[0] - 1)
        step += 1


def viz_main(policy: Policy, test_data_root: str, calc_pos, calc_rot):
    """
    Visually evalutes 2D policy on randomly sampled test environments.

    :param policy: Policy to evaluate
    :param test_data_root: Root directory of test data
    :param calc_pos: Whether to calculate position
    :param calc_rot: Whether to calculate rotation
    :return:
    """
    _, ax = plt.subplots(figsize=(14, 6))
    plt.draw()
    plt.pause(0.1)

    viz_args = dict(
        rollout_horizon=5,
        max_num_steps=200,
        lb=Params.lb_2D,
        ub=Params.ub_2D,
        dist_tol=0.3,
        calc_pos=calc_pos,
        calc_rot=calc_rot
    )

    N = 10
    for i in range(N):
        # Load data sample
        rand_file_idx = np.random.choice(100)
        print("file idx: %d" % rand_file_idx)
        data = np.load(os.path.join(test_data_root, "traj_%d.npz" %
                       rand_file_idx), allow_pickle=True)
        expert_traj = data["states"]
        goal_radius = data["goal_radius"].item()
        object_poses = np.copy(data["object_poses"])
        object_radii = np.copy(data["object_radii"])
        object_types = np.copy(data["object_types"])
        theta_offsets = Params.ori_offsets_2D[object_types]
        objects = [
            Object(pos=object_poses[i][0:POS_DIM], radius=object_radii[i],
                   ori=object_poses[i][-1] + theta_offsets[i]) for i in range(len(object_types))
        ]

        viz_helper(policy, start=expert_traj[0], goal=expert_traj[-1],
                   goal_radius=goal_radius, object_poses=object_poses,
                   object_radii=object_radii[:,
                                             np.newaxis], object_types=object_types,
                   objects=objects, expert_traj=expert_traj, ax=ax, viz_args=viz_args)


def viz_adaptation(policy: Policy, num_updates, test_data_root: str, train_pos, train_rot, is_rot_ignore, adaptation_func, save_res_dir=None, log_file=None):
    """
    Visually evalutes 2D policy with varying number of adaptation steps, where
    adaptation is performed directly on expert data.

    :param policy: Policy to evaluate
    :param test_data_root: Root directory of test data
    :param train_pos: Whether to train position
    :param train_rot: Whether to train rotation
    :return:
    """
    _, ax = plt.subplots(figsize=(14, 6))
    plt.draw()
    plt.pause(0.1)
    # view traj at these update indices
    num_updates_series = list(range(num_updates + 1))

    N = 10
    sample_i = 0
    seen = set()
    while sample_i < N:
        # Load data
        rand_file_idx = np.random.choice(3000)
        if rand_file_idx in seen:
            continue
        data = np.load(os.path.join(test_data_root, "traj_%d.npz" %
                       rand_file_idx), allow_pickle=True)
        object_types = data["object_types"]
        if train_rot:
            if is_rot_ignore and object_types[0] != Params.IGNORE_ROT_IDX:
                continue
            if not is_rot_ignore and object_types[0] != Params.CARE_ROT_IDX:
                continue

        print("OBJECT TYPES: ", object_types)
        seen.add(rand_file_idx)
        sample_i += 1
        print("file idx: %d" % rand_file_idx)
        assert data["is_rot"].item() == train_rot
        expert_traj = data["states"]
        T = expert_traj.shape[0]
        goal_radius = data["goal_radius"].item()
        object_poses = data["object_poses"]
        object_radii = data["object_radii"]
        # NOTE: this is purely for viz, model should NOT know this!
        theta_offsets = Params.ori_offsets_2D[object_types]
        start = expert_traj[0]
        goal = expert_traj[-1]
        num_objects = len(object_types)
        # NOTE: rather, object indices should be actual indices, not types
        object_idxs = np.arange(num_objects)

        # (traj, goals_r, objs, objs_r, obj_idxs)
        # batch_data = (expert_traj, goal_radius,
        #               # repeat for T timesteps unless object can move
        #               object_poses[np.newaxis, :, :].repeat(T, axis=0),
        #               object_radii[np.newaxis, :, np.newaxis].repeat(
        #                   T, axis=0),  # repeat for T timesteps
        #               object_idxs)

        batch_data = (expert_traj, np.array([goal_radius]),
                      # repeat for T timesteps unless object can move
                      object_poses,
                      object_radii[..., np.newaxis],
                      object_idxs)

        objects = [
            Object(pos=object_poses[i][0:POS_DIM], radius=object_radii[i],
                   ori=object_poses[i][-1] + theta_offsets[i]) for i in range(len(object_types))
        ]

        # Reset learned object features for objects
        pos_obj_types = [None] * num_objects  # None means no pos preference
        pos_requires_grad = [train_pos] * num_objects
        rot_obj_types = [None] * num_objects  # None means no rot preference
        rot_requires_grad = [train_rot] * num_objects
        policy.init_new_objs(pos_obj_types=pos_obj_types, rot_obj_types=rot_obj_types,
                             pos_requires_grad=pos_requires_grad, rot_requires_grad=rot_requires_grad, use_rand_init=False)

        # Observe model performance after various number of update steps
        total_updates = max(num_updates_series) + 1
        if train_pos:
            target_params = [
                policy.policy_network.pos_pref_feat_train[Params.REPEL_IDX],
                policy.policy_network.pos_pref_feat_train[Params.ATTRACT_IDX]
            ]
        else:
            if object_types[0] == Params.IGNORE_ROT_IDX:
                target_params = [
                    policy.policy_network.rot_pref_feat_train[object_types[0]],
                ]
            else:
                target_params = [
                    policy.policy_network.rot_pref_feat_train[object_types[0]],
                    Params.ori_offsets_2D[Params.CARE_ROT_IDX],
                ]
        write_log(log_file, f"Target params: {params2str(target_params)}")
        _, pred_trajs = adaptation_func(policy=policy, batch_data=[batch_data],
                                        train_pos=train_pos, train_rot=train_rot,
                                        n_adapt_iters=total_updates, dstep=Params.dstep_2D,
                                        verbose=True, clip_params=False, ret_trajs=True, log_file=log_file)
        pred_trajs = np.vstack(pred_trajs)

        for update_i in num_updates_series:
            pred_traj = pred_trajs[update_i]

            # Convert <cos, sin> representation back to theta
            pred_traj = np.hstack([pred_traj[:, :POS_DIM],
                                   np.arctan2(pred_traj[:, -1], pred_traj[:, -2])[:, np.newaxis]])

            if save_res_dir is not None:
                save_path = os.path.join(save_res_dir,
                                         f"sample_{sample_i}_adapt_step_{update_i}.png")
            else:
                save_path = None
            draw(ax, start_pose=start, goal_pose=goal,
                 goal_radius=goal_radius, agent_radius=Params.agent_radius,
                 object_types=object_types, offset_objects=objects,
                 pred_traj=pred_traj, expert_traj=expert_traj,
                 title=f"Num updates: {update_i}", show_rot=train_rot, hold=False, save_path=save_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store',
                        type=str, help="trained model name")
    parser.add_argument('--data_name', action='store',
                        type=str, help="name for data folder")
    parser.add_argument('--loaded_epoch', action='store', type=int, default=-1)
    parser.add_argument('--num_updates', action='store', type=int, default=4)
    parser.add_argument('--calc_rot', action='store_true',
                        help="calculate rotation")
    parser.add_argument('--is_rot_ignore', action='store_true')
    parser.add_argument('--calc_pos', action='store_true',
                        help="calculate position")
    parser.add_argument('--adapt', action='store_true',
                        help="Viz adaptation. If False, viz policy with pretrained object features.")
    parser.add_argument('--use_sgd', action='store_true',
                        help="Use vanilla SGD to adapt object features.")
    parser.add_argument('--use_rls', action='store_true',
                        help="Use rls to adapt object features.")
    parser.add_argument('--use_learn2learn', action='store_true',
                        help="Use Learn2learn to adapt object features.")
    parser.add_argument('--learn2learn_name', action='store',
                        type=str, help="trained learned optimizer")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    assert args.model_name is not None

    # Load trained model arguments
    with open(os.path.join(Params.model_root, args.model_name, "train_args_pt_1.json"), "r") as f:
        train_args = json.load(f)
    test_data_root = f"{Params.data_root}/{args.data_name}_test/"

    # load model
    assert not train_args["is_3D"], "viz_2D only for 2D models"
    network = PolicyNetwork(n_objects=Params.n_train_objects, pos_dim=POS_DIM, rot_dim=ROT_DIM,
                            pos_preference_dim=train_args['pos_preference_dim'],
                            rot_preference_dim=train_args['rot_preference_dim'],
                            hidden_dim=train_args['hidden_dim'],
                            device=DEVICE).to(DEVICE)
    network.load_state_dict(
        torch.load(os.path.join(Params.model_root, args.model_name, "model_%d.h5" % args.loaded_epoch)))
    policy = Policy(network)

    # ######### Visualize pretrained behavior #######
    if not args.adapt:
        viz_main(policy, test_data_root,
                 calc_pos=args.calc_pos, calc_rot=args.calc_rot)

    # ######### Visualize adaptation behavior #######
    # define scene objects and whether or not we care about their pos/ori
    else:
        if args.use_rls:
            alpha = 0.5
            lmbda = 0.9
            print(
                f"Using RLS(alpha_{alpha}_lmbda_{lmbda}) to adapt object features.")
            save_res_dir = f"eval_adaptation_results/RLS(alpha_{alpha}_lmbda_{lmbda})"
            rls = RLS(alpha=alpha, lmbda=lmbda)
            adaptation_func = lambda *args, **kwargs: perform_adaptation_rls(
                rls=rls, *args, **kwargs)
        elif args.use_learn2learn:
            print("Using Learn2Learn to adapt object features.")
            save_res_dir = f"eval_adaptation_results/learn2learn_group_train_dot_prod_eval_L2"
            learned_opts = LearnedOptimizerGroup(
                pos_opt_path=os.path.join(
                    Params.model_root, "learned_opt_pos_pref_rand_init_20_unroll_3_redo2_dot_prod"),
                rot_opt_path=os.path.join(
                    Params.model_root, "learned_opt_rot_pref_rand_init_30_unroll_2"),
                rot_offset_opt_path=os.path.join(
                    Params.model_root, "learned_opt_rot_offset_rand_init_30_unroll_2"),
                device=DEVICE,
                pos_epoch=3, rot_epoch=4, rot_offset_epoch=3)
            adaptation_func = lambda *args, **kwargs: perform_adaptation_learn2learn_group(
                learned_opts=learned_opts, is_3D=train_args["is_3D"],
                *args, **kwargs)

            # learned_opt = LearnedOptimizer(device=DEVICE, max_steps=2)
            # learned_opt.to(DEVICE)
            # learned_opt.train()
            # if args.learn2learn_name is not None:
            #     save_res_dir = f"eval_adaptation_results/{args.model_name}"
            #     learned_opt.load_state_dict(
            #         torch.load(os.path.join(Params.model_root, args.model_name, args.learn2learn_name)))
            # else:
            #     save_res_dir = None
            #     print(
            #         "WARNING: No learned optimizer name provided. Using random init weights!")
            # adaptation_func = lambda *args, **kwargs: perform_adaptation_learn2learn(
            #     learned_opt=learned_opt, *args, **kwargs)
        elif args.use_sgd:
            print("Using SGD to adapt object features.")
            optim_params = {'lr': 0.1, 'momentum': 0.9}
            adaptation_func = lambda *args, **kwargs: perform_adaptation(
                Optimizer=torch.optim.SGD, optim_params=optim_params, *args, **kwargs)
            save_res_dir = "eval_adaptation_results/SGD(lr_%.1f_momentum_%.1f)" % (
                optim_params['lr'], optim_params['momentum'])
        else:
            print("Using Adam to adapt object features.")
            optim_params = {'lr': 0.1}
            adaptation_func = lambda *args, **kwargs: perform_adaptation(
                Optimizer=torch.optim.Adam, optim_params=optim_params, *args, **kwargs)
            save_res_dir = "eval_adaptation_results/Adam"

        if save_res_dir is not None:
            if args.calc_pos:
                save_res_dir += "_pos"
            if args.calc_rot:
                save_res_dir += "_rot"
            if args.is_rot_ignore:
                save_res_dir += "_ignore"
            os.makedirs(f"{save_res_dir}/samples", exist_ok=True)
            log_file = open(f"{save_res_dir}/qualitative_output.txt", "w")
        else:
            log_file = None

        viz_adaptation(policy, args.num_updates, test_data_root,
                       train_pos=args.calc_pos, train_rot=args.calc_rot,
                       adaptation_func=adaptation_func, is_rot_ignore=args.is_rot_ignore, save_res_dir=f"{save_res_dir}/samples", log_file=log_file)
