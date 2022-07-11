"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import numpy as np
import os
import argparse
import ipdb
import json
import typing
from pynput import keyboard

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import torch

from model import Policy, PolicyNetwork, pose_to_model_input, decode_ori
from train import perform_adaptation, DEVICE
from data_params import Params
from exp_params import *
from exp_utils import *
from elastic_band import Object
from viz_3D import Viz3DROSPublisher, pose_to_msg, msg_to_pose
from exp1_cup_low_table_sim import robot_table_surface_projections

World2Net = 5.0
Net2World = 1 / World2Net

ee_min_pos_world = np.array([-0.5, -0.6, -0.16])
ee_max_pos_world = np.array([0.725, -0.2, 0.35])
num_objects = 1
object_radii = np.array([1.5] * num_objects)  # defined on net scale
goal_rot_radius = np.array([1.5])
goal_pos_world = np.array([-0.47447, -0.514564, 0.0])
goal_ori_quat = np.array([0.641, 0.755, -0.077, 0.109])
start_pos_world = np.array([0.45, -0.57, -0.16])
start_ori_quat = np.array([0.641, 0.755, -0.077, 0.109])
scanner_pos_world = np.array([-0.08, -0.79, 0.375])
scanner_ori_quat = np.array([0,0,0,1.0])
# desired scanner ori offset: [0.55218, 0.622621, -0.421491, 0.360256]
obstacles = []

# Need more adaptation steps because this example is more difficult
custom_num_pos_net_updates = 30
custom_num_rot_net_updates = 40

# Global info updated by callbacks
cur_pos_world, cur_ori_quat = None, None
perturb_pose_traj_world = []
is_intervene = False
need_update = False

DEBUG = False
if DEBUG:
    dstep = 0.01
else:
    dstep = 0.2


def is_intervene_cb(key):
    global is_intervene, need_update
    if key == keyboard.Key.space:
        is_intervene = not is_intervene
        if not is_intervene:  # True -> False, end of intervention
            need_update = True

def obj_pose_cb(msg):
    # TODO:
    global is_intervene
    if is_intervene:
        pass

def robot_pose_cb(msg):
    global is_intervene, cur_pos_world, cur_ori_quat, perturb_pose_traj_world, DEBUG
    pose = msg_to_pose(msg)
    if not DEBUG:
        cur_pos_world = pose[0:3]
        cur_ori_quat = pose[3:]
        if is_intervene:
            perturb_pose_traj_world.append(np.concatenate([cur_pos_world, cur_ori_quat]))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store',
                        type=str, help="trained model name")
    parser.add_argument('--loaded_epoch', action='store', type=int)
    parser.add_argument('--view_ros', action='store_true',
                        help="visualize 3D scene with ROS Rviz")
    parser.add_argument('--user', action='store', type=str,
                        default="some_user", help="user name to save results")
    parser.add_argument('--trial', action='store', type=int, default=0)
    args = parser.parse_args()

    return args

def reach_start_pos():
    global cur_pos_world, cur_ori_quat
    global start_pos_world, start_ori_quat
    if DEBUG:
        cur_pos_world = np.copy(start_pos_world)
        cur_ori_quat = np.copy(start_ori_quat)
    else:
        start_dist = 1e10
        dEE_pos = 1e10
        prev_pos_world = np.copy(cur_pos_world)
        while not rospy.is_shutdown() and (
            cur_pos_world is None or start_dist > 0.1 or dEE_pos > 1e-2):
            pose_pub.publish(pose_to_msg(start_pose_world))
            if cur_pos_world is None:
                print("Waiting to receive robot pos")
            else:
                start_dist = np.linalg.norm(cur_pos_world - start_pos_world)
                dEE_pos = np.linalg.norm(cur_pos_world - prev_pos_world)
                prev_pos_world = np.copy(cur_pos_world)
                print("Waiting to reach start pos, error: %.3f, change: %.3f" % (start_dist, dEE_pos))
            rospy.sleep(0.4)

if __name__ == "__main__":
    args = parse_arguments()
    argparse_dict = vars(args)
    rospy.init_node('exp1_OPA')

    # Robot EE pose
    rospy.Subscriber('/kinova/pose_tool_in_base_fk',
                     PoseStamped, robot_pose_cb, queue_size=1)

    # Target pose topic
    pose_pub = rospy.Publisher("/kinova_demo/pose_cmd", PoseStamped, queue_size=10)
    is_intervene_pub = rospy.Publisher("/is_intervene", Bool, queue_size=10)

    # Listen for keypresses marking start/stop of human intervention
    listener = keyboard.Listener(
        on_press=is_intervene_cb)
    listener.start()

    # create folder to store user results
    user_dir = os.path.join("user_trials", args.user, "exp_real_world")
    os.makedirs(user_dir, exist_ok=True)

    # Save experiment input arguments
    with open(os.path.join(user_dir, "args.json"), "w") as outfile:
        json.dump(argparse_dict, outfile, indent=4)

    # Load trained model arguments
    with open(os.path.join(Params.model_root, args.model_name, "train_args_pt_1.json"), "r") as f:
        train_args = json.load(f)

    # load model
    is_3D = train_args["is_3D"]
    assert is_3D, "Run experiments with 3D models to control EE in 3D space"
    pos_dim, rot_dim = 3, 6
    network = PolicyNetwork(n_objects=Params.n_train_objects, pos_dim=pos_dim, rot_dim=rot_dim,
                            pos_preference_dim=train_args['pos_preference_dim'],
                            rot_preference_dim=train_args['rot_preference_dim'],
                            hidden_dim=train_args['hidden_dim'],
                            device=DEVICE).to(DEVICE)
    network.load_state_dict(
        torch.load(os.path.join(Params.model_root, args.model_name, "model_%d.h5" % args.loaded_epoch)))
    policy = Policy(network)

    # define scene objects and whether or not we care about their pos/ori
    # In exp3, don't know anything about the scanner or table. Initialize both
    # position and orientation features as None with requires_grad=True
    calc_rot, calc_pos = True, True
    train_rot, train_pos = True, True
    # NOTE: not training "object_types", purely object identifiers
    object_idxs = np.arange(num_objects)
    pos_obj_types = [None] * num_objects
    pos_requires_grad = [train_pos] * num_objects
    rot_obj_types = [None] * num_objects
    rot_requires_grad = [train_rot] * num_objects
    policy.init_new_objs(pos_obj_types=pos_obj_types, rot_obj_types=rot_obj_types,
                         pos_requires_grad=pos_requires_grad, rot_requires_grad=rot_requires_grad)

    # Set start robot pose
    start_pose_world = np.concatenate([start_pos_world, start_ori_quat])
    start_pose_net = np.concatenate(
        [start_pos_world * World2Net, start_ori_quat])

    # Define final goal pose (drop-off bin)
    goal_pose_net = np.concatenate([goal_pos_world * World2Net, goal_ori_quat])

    # Define inspection zone pose
    object_pos_net = World2Net * scanner_pos_world
    object_poses_net = np.concatenate(
        [object_pos_net, scanner_ori_quat], axis=-1)[np.newaxis]

    # Convert np arrays to torch tensors for model input
    start_tensor = torch.from_numpy(
        pose_to_model_input(start_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
    agent_radius_tensor = torch.tensor(
        [Params.agent_radius], device=DEVICE).view(1, 1)
    goal_rot_radii = torch.from_numpy(goal_rot_radius).to(
        torch.float32).to(DEVICE).view(1, 1)
    object_radii_torch = torch.from_numpy(object_radii).to(
        torch.float32).to(DEVICE).view(num_objects, 1)
    object_idxs_tensor = torch.from_numpy(object_idxs).to(
        torch.long).to(DEVICE).unsqueeze(0)
    object_poses_tensor = torch.from_numpy(pose_to_model_input(
        object_poses_net)).to(torch.float32).to(DEVICE)
    objects_torch = torch.cat(
        [object_poses_tensor, object_radii_torch], dim=-1).unsqueeze(0)

    # Optionally view with ROS Rviz
    if args.view_ros:
        viz_3D_publisher = Viz3DROSPublisher(num_objects=num_objects)
        # TODO:
        object_colors = [
            (0, 255, 0),
        ]
        all_object_colors = object_colors + [
            Params.start_color_rgb,
            Params.goal_color_rgb,
            Params.agent_color_rgb,
        ]
        force_colors = object_colors + [Params.goal_color_rgb]

    it = 0
    tol = 0.07
    perturb_pose_traj_world = []
    for exp_iter in range(3):
        reach_start_pos()
        error = 1e10
        while not rospy.is_shutdown() and error > tol:
            it += 1
            # print("cur pos: ", cur_pos_world)
            cur_pos_net = cur_pos_world * World2Net
            cur_pose_net = np.concatenate([cur_pos_net, cur_ori_quat])
            error = np.linalg.norm(goal_pos_world - cur_pos_world)

            if need_update and not DEBUG:
                is_intervene = False
                is_intervene_pub.publish(Bool(is_intervene))
                assert len(perturb_pose_traj_world) > 1, "Need intervention traj of > 1 steps"
                dist = np.linalg.norm(
                    perturb_pose_traj_world[-1][0:POS_DIM] - perturb_pose_traj_world[0][0:POS_DIM])
                # 1 step for start, 1 step for goal at least
                T = max(2, int(np.ceil(dist / dstep)))
                perturb_pos_traj_world_interp = np.linspace(
                    start=perturb_pose_traj_world[0][0:POS_DIM], stop=perturb_pose_traj_world[-1][0:POS_DIM], num=T)
                final_perturb_ori = perturb_pose_traj_world[-1][POS_DIM:]
                perturb_ori_traj = np.copy(final_perturb_ori)[np.newaxis, :].repeat(T, axis=0)
                perturb_pos_traj_net = perturb_pos_traj_world_interp * World2Net
                perturb_pose_traj_net = np.hstack([np.vstack(perturb_pos_traj_net), perturb_ori_traj])
                batch_data = [
                    (perturb_pose_traj_net, start_pose_net, goal_pose_net, goal_rot_radius,
                        object_poses_net[np.newaxis].repeat(T, axis=0),
                        object_radii[np.newaxis].repeat(T, axis=0),
                        object_idxs)]

                # Update policy in position first, then orientation after with diff num iters
                perform_adaptation(policy=policy, batch_data=batch_data,
                                train_pos=True, train_rot=False,
                                n_adapt_iters=custom_num_pos_net_updates, dstep=dstep,
                                verbose=True, clip_params=True)

                T = 2
                perturb_pose_traj_net = perturb_pose_traj_net[[0, -1], :]
                batch_data = [
                    (perturb_pose_traj_net, start_pose_net, goal_pose_net, goal_rot_radius,
                        object_poses_net[np.newaxis].repeat(T, axis=0),
                        object_radii[np.newaxis].repeat(T, axis=0),
                        object_idxs)]
                # Update policy
                print("Care and Ignore feats: ",
                        policy.policy_network.rot_pref_feat_train[Params.CARE_ROT_IDX].detach().cpu().numpy(),
                        policy.policy_network.rot_pref_feat_train[Params.IGNORE_ROT_IDX].detach().cpu().numpy())
                print("Old pref feats: ", torch.cat(policy.obj_rot_feats, dim=0).detach().cpu().numpy())
                print("Old rot offset: ", torch.stack(policy.obj_rot_offsets).detach().cpu().numpy())
                perform_adaptation(policy=policy, batch_data=batch_data,
                                    train_pos=False, train_rot=True,
                                    n_adapt_iters=num_rot_net_updates, dstep=dstep,
                                    verbose=True, clip_params=True)
                print("New pref feats: ", torch.cat(policy.obj_rot_feats, dim=0).detach().cpu().numpy())
                print("New rot offset: ", torch.stack(policy.obj_rot_offsets).detach().cpu().numpy())

                # reset the intervention data
                perturb_pose_traj_world = []
                need_update = False
                continue  # start new trial with updated weights

            else:
                with torch.no_grad():
                    # Define "object" inputs into policy
                    # current
                    cur_pose_tensor = torch.from_numpy(
                        pose_to_model_input(cur_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
                    current = torch.cat(
                        [cur_pose_tensor, agent_radius_tensor], dim=-1).unsqueeze(1)
                    # goal
                    goal_tensor = torch.from_numpy(
                        pose_to_model_input(goal_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
                    goal_radii = goal_radius_scale * torch.norm(
                        goal_tensor[:, :pos_dim] -
                        cur_pose_tensor[:, :pos_dim],
                        dim=-1).unsqueeze(0)
                    goal_rot_objects = torch.cat(
                        [goal_tensor, goal_rot_radii], dim=-1).unsqueeze(1)
                    goal_objects = torch.cat(
                        [goal_tensor, goal_radii], dim=-1).unsqueeze(1)
                    # start
                    start_rot_radii = torch.norm(start_tensor[:, :pos_dim] - cur_pose_tensor[:, :pos_dim],
                                                dim=-1).unsqueeze(0)
                    start_rot_objects = torch.cat(
                        [start_tensor, start_rot_radii], dim=-1).unsqueeze(1)

                    # Get policy output, form into action
                    pred_vec, pred_ori, object_forces = policy(current=current,
                                                            start=start_rot_objects,
                                                            goal=goal_objects, goal_rot=goal_rot_objects,
                                                            objects=objects_torch,
                                                            object_indices=object_idxs_tensor,
                                                            calc_rot=calc_rot,
                                                            calc_pos=calc_pos)
                    local_target_pos_world = cur_pose_tensor[0, 0:pos_dim] * Net2World
                    local_target_pos_world = local_target_pos_world + \
                        dstep * pred_vec[0, :pos_dim]
                    local_target_pos_world = local_target_pos_world.detach().cpu().numpy()
                    local_target_ori = decode_ori(
                        pred_ori.detach().cpu().numpy()).flatten()

            # Optionally view with ROS
            if args.view_ros:
                all_objects = [Object(pos=pose[0:POS_DIM], ori=pose[POS_DIM:],
                                    radius=radius) for pose, radius in
                            zip(object_poses_net, object_radii)]
                all_objects += [
                    Object(
                        pos=start_pose_net[0:POS_DIM], radius=Params.agent_radius, ori=start_pose_net[POS_DIM:]),
                    Object(
                        pos=goal_pose_net[0:POS_DIM], radius=Params.agent_radius, ori=goal_pose_net[POS_DIM:]),
                    Object(
                        pos=cur_pose_net[0:POS_DIM], radius=Params.agent_radius, ori=cur_pose_net[POS_DIM:])
                ]
                agent_traj = np.vstack(
                    [cur_pose_net, np.concatenate([Net2World * local_target_pos_world, cur_ori_quat])])
                if isinstance(object_forces, torch.Tensor):
                    object_forces = object_forces[0].detach().cpu().numpy()
                viz_3D_publisher.publish(objects=all_objects, agent_traj=agent_traj,
                                        expert_traj=None, object_colors=all_object_colors,
                                        object_forces=object_forces, force_colors_rgb=force_colors)

            # Apply low-pass filter to smooth out policy's sudden changes in orientation
            key_times = [0, 1]
            rotations = R.from_quat(
                np.vstack([cur_ori_quat, local_target_ori]))
            slerp = Slerp(key_times, rotations)
            alpha = 0.3   # alpha*cur_ori_quat + alpha*local_target_ori
            interp_rot = slerp([alpha])[0]

            # Clip target EE position to bounds
            local_target_pos_world = np.clip(local_target_pos_world, a_min=ee_min_pos_world, a_max=ee_max_pos_world)

            # Publish is_intervene
            is_intervene_pub.publish(Bool(is_intervene))
            # Publish target pose
            if not DEBUG:
                target_pose = np.concatenate([local_target_pos_world, interp_rot.as_quat()])
                pose_pub.publish(pose_to_msg(target_pose))
                # # Publish is_intervene
                # is_intervene_pub.publish(Bool(is_intervene))

            if DEBUG:
                cur_pos_world = local_target_pos_world
                cur_ori_quat = interp_rot.as_quat()

            rospy.sleep(0.1)

        print(f"Finished! Error {error} vs tol {tol}")