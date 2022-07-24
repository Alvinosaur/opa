"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from curses.ascii import CAN
import numpy as np
import os
import argparse
import ipdb
import json
import typing
from tqdm import tqdm
import copy
from pynput import keyboard

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32

import torch

from model import Policy, PolicyNetwork, pose_to_model_input, decode_ori
from train import random_seed_adaptation, process_single_full_traj, DEVICE
from data_params import Params
from exp_params import *
from exp_utils import *
from elastic_band import Object
from viz_3D import Viz3DROSPublisher, pose_to_msg, msg_to_pose
from exp1_cup_low_table_sim import robot_table_surface_projections
from data_generation import rand_quat

World2Net = 10.0
Net2World = 1 / World2Net
num_objects = 1 + 2  # + 2 for obstacles added later
ee_min_pos_world = np.array([0.23, -0.475, -0.1])
ee_max_pos_world = np.array([0.725, 0.55, 0.35])
inspection_radii = np.array([7.0])[:, np.newaxis]  # defined on net scale
goal_rot_radius = np.array([2.0])
goal_pos_world = np.array([0.4, -0.475, 0.1])
goal_ori_quat = np.array([0.707, 0.707, 0, 0])
# start_pos_world = np.array([0.4, 0.525, -0.1])
start_pos_world = None
HOME_POSE_NET = np.array([3.73, 1.0, 1.3, 0.707, 0.707, 0, 0])
BOX_MASS = 0.3
CAN_MASS = 0.6
start_poses = [
    # boxes
    np.array([0.2, 0.35, -0.131]),
    np.array([0.2, 0.465, -0.131]),
    np.array([0.22, 0.58, -0.131]),

    # cans
    np.array([0.333, 0.43, -0.16]),
    np.array([0.333, 0.545, -0.16]),
    np.array([0.43, 0.43, -0.16]),
    np.array([0.43, 0.545, -0.16]),
]
start_ori_quats = [
    # boxes
    np.array([0, 1., 0, 0]),
    np.array([0, 1., 0, 0]),
    np.array([0, 1., 0, 0]),

    # cans
    np.array([0.707, 0.707, 0, 0]),
    np.array([0.707, 0.707, 0, 0]),
    np.array([0.707, 0.707, 0, 0]),
    np.array([0.707, 0.707, 0, 0]),
]
extra_masses = [
    BOX_MASS,
    BOX_MASS,
    BOX_MASS, 

    CAN_MASS,
    CAN_MASS,
    CAN_MASS,
    CAN_MASS,
]
# import ipdb
# ipdb.set_trace()

# start_ori_quat = np.array([-0.75432945, -0.00351821, -0.64973774,
#  -0.09389131])
inspection_pos_world = np.array([0.7, 0.1, -0.1])
inspection_ori_quat = np.array([0, 0, 0, 1.0])
inspection_pos_world_2 = np.array([0.7, -0.3, 0.5])
inspection_ori_quat_2 = R.from_euler("xyz", [0, 0, -45], degrees=True).as_quat()
obstacles_pos_world = [
    np.array([0.484, 0.1, 0.0]),
    np.array([0.734, -0.3, 0.0]),
]
obstacles_ori_quat = [
    np.array([0,0,0,1.]),
    np.array([0,0,0,1.]),
]
obstacles_radii = np.array([2.0, 1.5])[:, np.newaxis]

# Need more adaptation steps because this example is more difficult
custom_num_pos_net_updates = 20
custom_num_rot_net_updates = 35

# Global info updated by callbacks
cur_pos_world, cur_ori_quat = None, None
perturb_pose_traj_world = []
is_intervene = False
need_update = False

DEBUG = False
if DEBUG:
    dstep = 0.05
    ros_delay = 0.1
else:
    dstep = 0.13
    ros_delay = 0.4


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
            perturb_pose_traj_world.append(
                np.concatenate([cur_pos_world, cur_ori_quat]))


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

def reach_start_pos(viz_3D_publisher, start_pose_net, goal_pose_net, object_poses_net, object_radii):
    global cur_pos_world, cur_ori_quat
    start_pos_world = Net2World * start_pose_net[:3]
    start_ori_quat = start_pose_net[3:]
    start_pose_world = np.concatenate([start_pos_world, start_ori_quat])
    if DEBUG:
        cur_pos_world = np.copy(start_pos_world)
        cur_ori_quat = np.copy(start_ori_quat)
        # cur_pos_world = np.copy(inspection_pos_world)
        # cur_ori_quat = np.copy(inspection_ori_quat)
        return 

    else:
        start_dist = np.linalg.norm(start_pos_world - cur_pos_world)
        pose_error = 1e10
        dEE_pos = 1e10
        dEE_pos_running_avg = RunningAverage(length=5, init_vals=1e10)
        prev_pos_world = np.copy(cur_pos_world)
        while not rospy.is_shutdown() and (
                cur_pos_world is None or (pose_error > 0.12 and dEE_pos_running_avg.avg > 1e-2 )):
            pos_vec = start_pos_world - cur_pos_world
            pos_mag = np.linalg.norm(pos_vec)
            target_pos_world = cur_pos_world + pos_vec * min(pos_mag, dstep) / pos_mag
            # target_pos_world = cur_pos_world + np.array([0, -0.1, 0.2])
            dist_to_start_ratio = min(pos_mag / (start_dist  + 1e-5), 1.0)
            target_ori_quat = interpolate_rotations(start_quat=cur_ori_quat, stop_quat=start_ori_quat, 
                                                    alpha=1-dist_to_start_ratio)
            pose_pub.publish(
                pose_to_msg(np.concatenate([target_pos_world, target_ori_quat]), frame=ROBOT_FRAME))
            is_intervene_pub.publish(False)

            # Publish objects
            object_colors = [
                (0, 255, 0),
            ]
            all_object_colors = object_colors + [
                Params.start_color_rgb,
                Params.goal_color_rgb,
                Params.agent_color_rgb,
            ]
            # force_colors = object_colors + [Params.goal_color_rgb]
            all_objects = [Object(pos=Net2World * pose[0:POS_DIM], ori=[0.7627784,  -0.00479786 , 0.6414479,   0.08179578],
                                      radius=Net2World * radius) for pose, radius in
                               zip(object_poses_net, object_radii)]
            all_objects += [
                Object(
                    pos=Net2World * start_pose_net[0:POS_DIM], radius=Net2World * Params.agent_radius, ori=start_pose_net[POS_DIM:]),
                Object(
                    pos=Net2World * goal_pose_net[0:POS_DIM], radius=Net2World * goal_rot_radius.item(), ori=goal_pose_net[POS_DIM:]),
                Object(
                    pos=cur_pos_world, radius=Net2World * Params.agent_radius, ori=cur_ori_quat)
            ]
            viz_3D_publisher.publish(objects=all_objects, object_colors=all_object_colors,)

            if cur_pos_world is None:
                print("Waiting to receive robot pos")
            else:
                cur_pose_world = np.concatenate([cur_pos_world, cur_ori_quat])
                pose_error = calc_pose_error(cur_pose_world, start_pose_world, rot_scale=0.1)
                dEE_pos = np.linalg.norm(cur_pos_world - prev_pos_world)
                dEE_pos_running_avg.update(dEE_pos)
                prev_pos_world = np.copy(cur_pos_world)
                pos_error = np.linalg.norm(cur_pos_world - start_pos_world)
                print("Waiting to reach start pos (%s) error: %.3f,  change: %.3f" %
                      (np.array2string(target_pos_world, precision=2), pos_error, dEE_pos_running_avg.avg))
            rospy.sleep(ros_delay)
        rospy.sleep(0.5)  # pause to let arm finish converging

        print("Final error: ", pose_error)
        print(cur_pos_world, start_pos_world)


if __name__ == "__main__":
    args = parse_arguments()
    argparse_dict = vars(args)
    rospy.init_node('exp2_OPA')

    # Robot EE pose
    rospy.Subscriber('/kinova/pose_tool_in_base_fk',
                     PoseStamped, robot_pose_cb, queue_size=1)

    # Target pose topic
    pose_pub = rospy.Publisher(
        "/kinova_demo/pose_cmd", PoseStamped, queue_size=10)
    is_intervene_pub = rospy.Publisher("/is_intervene", Bool, queue_size=10)
    gripper_pub = rospy.Publisher("/siemens_demo/gripper_cmd", Bool, queue_size=10)
    extra_mass_pub = rospy.Publisher("/gripper_extra_mass", Float32, queue_size=10)

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
        torch.load(
            os.path.join(Params.model_root, args.model_name, "model_%d.h5" % args.loaded_epoch),
            map_location=DEVICE))
    policy = Policy(network)

    # define scene objects and whether or not we care about their pos/ori
    # In exp3, don't know anything about the scanner or table. Initialize both
    # position and orientation features as None with requires_grad=True
    calc_rot, calc_pos = True, True
    train_rot, train_pos = True, True
    # NOTE: not training "object_types", purely object identifiers
    object_idxs = np.arange(num_objects)
    pos_obj_types = [Params.GOAL_IDX, None, None]
    # pos_obj_types = [Params.GOAL_IDX, Params.REPEL_IDX, Params.REPEL_IDX]
    # pos_obj_types = [Params.ATTRACT_IDX, None, None]
    pos_requires_grad = [False, True, False]
    # rot_obj_types = [Params.CARE_ROT_IDX, None, None]
    rot_obj_types = [None, None, None]
    rot_requires_grad = [True, False, False]
    # rot_offsets_debug = [
    #     torch.tensor([-0.731, -0.677,  0.063,  0.058], device=DEVICE),
    #     torch.tensor([0, 0, 0, 1], device=DEVICE),
    #     torch.tensor([0, 0, 0, 1], device=DEVICE),
    # ]
    rot_offsets_debug = None
    policy.init_new_objs(pos_obj_types=pos_obj_types, rot_obj_types=rot_obj_types,
                         pos_requires_grad=pos_requires_grad, rot_requires_grad=rot_requires_grad,
                         rot_offsets=rot_offsets_debug)
    obj_pos_feats = policy.obj_pos_feats

    # # DEBUG LOAD SAVED WEIGHTS
    # saved_weights = torch.load("exp2_saved_weights_iter_4.pth")
    # policy.update_obj_feats(**saved_weights)
    # print(saved_weights)

    # Define final goal pose (drop-off bin)
    goal_pose_net = np.concatenate([goal_pos_world * World2Net, goal_ori_quat])
    goal_pose_world = np.concatenate([goal_pos_world, goal_ori_quat])

    # Convert np arrays to torch tensors for model input
    agent_radius_tensor = torch.tensor(
        [Params.agent_radius], device=DEVICE).view(1, 1)
    goal_rot_radii = torch.from_numpy(goal_rot_radius).to(
        torch.float32).to(DEVICE).view(1, 1)
    object_idxs_tensor = torch.from_numpy(object_idxs).to(
        torch.long).to(DEVICE).unsqueeze(0)
    obstacles_radii_torch = torch.from_numpy(obstacles_radii).to(
                                torch.float32).to(DEVICE).view(-1, 1)
    inspection_radii_torch = torch.from_numpy(inspection_radii).to(
                                torch.float32).to(DEVICE).view(-1, 1)
    object_radii_torch = torch.vstack([inspection_radii_torch, obstacles_radii_torch])
    object_radii = np.vstack([inspection_radii, obstacles_radii])

    # Optionally view with ROS Rviz
    if args.view_ros:
        viz_3D_publisher = Viz3DROSPublisher(num_objects=num_objects, frame=ROBOT_FRAME)
        # TODO:
        object_colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
        ]
        all_object_colors = object_colors + [
            Params.start_color_rgb,
            Params.goal_color_rgb,
            Params.agent_color_rgb,
        ]
        force_colors = object_colors + [Params.goal_color_rgb]

    # Define pretrained feature bounds for random initialization
    pos_feat_min = torch.min(policy.policy_network.pos_pref_feat_train[Params.ATTRACT_IDX],
                             policy.policy_network.pos_pref_feat_train[Params.REPEL_IDX]).item()
    pos_feat_max = torch.max(policy.policy_network.pos_pref_feat_train[Params.ATTRACT_IDX],
                             policy.policy_network.pos_pref_feat_train[Params.REPEL_IDX]).item()

    rot_feat_min = torch.min(policy.policy_network.rot_pref_feat_train[Params.CARE_ROT_IDX],
                             policy.policy_network.rot_pref_feat_train[Params.IGNORE_ROT_IDX]).item()
    rot_feat_max = torch.max(policy.policy_network.rot_pref_feat_train[Params.CARE_ROT_IDX],
                             policy.policy_network.rot_pref_feat_train[Params.IGNORE_ROT_IDX]).item()
    pos_attract_feat = policy.policy_network.pos_pref_feat_train[Params.ATTRACT_IDX].detach(
    )

    it = 0
    pose_error_tol = 0.07
    max_pose_error_tol = 0.2  # too far to be considered converged and not moving
    del_pose_tol = 0.005  # over del_pose_interval iterations
    perturb_pose_traj_world = []
    override_pred_delay = False
    # [box1 intervene rot finish (don't reset to start), box2 watch, 
    # box3 (pretend, need to buy) add new obstacles show online repulsion from them, 
    # box4 watch final behavior
    # can1 intervene rot finish, can2 watch, can3 stand up and watch
    command_kinova_gripper(gripper_pub, cmd_open=True)
    num_exps = len(start_poses)
    # num_exps = 3
    for exp_iter in range(num_exps):  # +1 for original start pos
        # set extra mass of object to pick up
        extra_mass = extra_masses[exp_iter]

        # Set start robot pose
        start_pos_world = start_poses[exp_iter]
        start_ori_quat = start_ori_quats[exp_iter]
        start_pose_world = np.concatenate([start_pos_world, start_ori_quat])
        start_pose_net = np.concatenate(
            [start_pos_world * World2Net, start_ori_quat])
        start_tensor = torch.from_numpy(
            pose_to_model_input(start_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)

        if exp_iter == num_exps - 1:
            inspection_pos_world = inspection_pos_world_2
            inspection_ori_quat = inspection_ori_quat_2
        inspection_pos_net = World2Net * inspection_pos_world
        inspection_pose_net = np.concatenate(
            [inspection_pos_net, inspection_ori_quat], axis=-1)[np.newaxis]
        inspection_pose_tensor = torch.from_numpy(pose_to_model_input(
                inspection_pose_net)).to(torch.float32).to(DEVICE)

        obstacles_pos_net = World2Net * np.vstack(obstacles_pos_world)
        obstacles_pose_net = np.concatenate(
            [obstacles_pos_net, obstacles_ori_quat], axis=-1)
        obstacles_poses_tensor = torch.from_numpy(pose_to_model_input(
            obstacles_pose_net)).to(torch.float32).to(DEVICE)

        object_poses_net = np.vstack([inspection_pose_net, obstacles_pose_net])
        objects_tensor = torch.cat(
            [inspection_pose_tensor, obstacles_poses_tensor], dim=0)
        objects_torch = torch.cat(
            [objects_tensor, object_radii_torch], dim=-1).unsqueeze(0)

        # reach initial pose
        approach_pose_net = np.copy(start_pose_net)
        approach_pose_net[2] += 0.2 * World2Net
        reach_start_pos(viz_3D_publisher, HOME_POSE_NET, goal_pose_net, [], [])
        reach_start_pos(viz_3D_publisher, approach_pose_net, goal_pose_net, [], [])
        reach_start_pos(viz_3D_publisher, start_pose_net, goal_pose_net, [], [])
        # exit()
        command_kinova_gripper(gripper_pub, cmd_open=False)
        
        # once object is grabbed, set new mass
        for i in range(3): extra_mass_pub.publish(Float32(extra_mass))

        # move back up to approach pose
        reach_start_pos(viz_3D_publisher, approach_pose_net, goal_pose_net, [], [])

        # initialize target pose variables
        local_target_pos_world = np.copy(cur_pos_world)
        local_target_ori = np.copy(cur_ori_quat)

        pose_error = 1e10
        del_pose = 1e10
        del_pose_running_avg = RunningAverage(length=5, init_vals=1e10)

        prev_pose_world = None
        while (not rospy.is_shutdown() and pose_error > pose_error_tol and 
                (del_pose_running_avg.avg > del_pose_tol or pose_error > max_pose_error_tol)):
            cur_pose_world = np.concatenate([cur_pos_world, cur_ori_quat])
            cur_pos_net = cur_pos_world * World2Net
            cur_pose_net = np.concatenate([cur_pos_net, cur_ori_quat])
            pose_error = calc_pose_error(goal_pose_world, cur_pose_world, rot_scale=0)
            if prev_pose_world is not None:
                del_pose = calc_pose_error(prev_pose_world, cur_pose_world)
                del_pose_running_avg.update(del_pose)

            if need_update and not DEBUG:
                # DEBUG REMOVE
                for i in range(5):
                    pose_pub.publish(pose_to_msg(cur_pose_world, frame=ROBOT_FRAME))
                rospy.sleep(0.1)
                is_intervene = False
                is_intervene_pub.publish(is_intervene)
            
                assert len(
                    perturb_pose_traj_world) > 1, "Need intervention traj of > 1 steps"
                dist = np.linalg.norm(
                    perturb_pose_traj_world[-1][0:POS_DIM] - perturb_pose_traj_world[0][0:POS_DIM])
                # 1 step for start, 1 step for goal at least
                # T = max(2, int(np.ceil(dist / dstep)))
                T = 5
                perturb_pos_traj_world_interp = np.linspace(
                    start=perturb_pose_traj_world[0][0:POS_DIM], stop=perturb_pose_traj_world[-1][0:POS_DIM], num=T)
                final_perturb_ori = perturb_pose_traj_world[-1][POS_DIM:]

                perturb_ori_traj = np.copy(final_perturb_ori)[
                    np.newaxis, :].repeat(T, axis=0)
                perturb_pos_traj_net = perturb_pos_traj_world_interp * World2Net
                perturb_pose_traj_net = np.hstack(
                    [np.vstack(perturb_pos_traj_net), perturb_ori_traj])
                sample = (perturb_pose_traj_net, start_pose_net, goal_pose_net, goal_rot_radius,
                          object_poses_net[np.newaxis].repeat(T, axis=0),
                          object_radii[np.newaxis].repeat(T, axis=0),
                          object_idxs)
                processed_sample = process_single_full_traj(sample)

                # Update position
                random_seed_adaptation(policy, processed_sample, train_pos=True, train_rot=False, 
                            is_3D=True, num_objects=num_objects, loss_prop_tol=0.3, 
                            pos_feat_max=pos_feat_max, pos_feat_min=pos_feat_min,
                            rot_feat_max=rot_feat_max, rot_feat_min=rot_feat_min,
                            pos_requires_grad=pos_requires_grad)

                # Update rotation
                best_pos_feats, best_rot_feats, best_rot_offsets = (
                    random_seed_adaptation(policy, processed_sample, train_pos=False, train_rot=True, 
                            is_3D=True, num_objects=num_objects, loss_prop_tol=0.3, 
                            pos_feat_max=pos_feat_max, pos_feat_min=pos_feat_min,
                            rot_feat_max=rot_feat_max, rot_feat_min=rot_feat_min,
                            rot_requires_grad=rot_requires_grad))

                torch.save(
                    {
                        "obj_pos_feats": best_pos_feats,
                        "obj_rot_feats": best_rot_feats,
                        "obj_rot_offsets": best_rot_offsets
                    },
                    f"exp2_saved_weights_iter_{exp_iter}.pth"
                )

                # reset the intervention data
                perturb_pose_traj_world = []
                need_update = False
                override_pred_delay = True

                # reach back to the pose before p
                continue

                # break  # start new trial with updated weights

            elif it % 2 == 0 or override_pred_delay:
                override_pred_delay = False
                print("new target", it)
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
                    local_target_pos_world = cur_pose_tensor[0,
                                                             0:pos_dim] * Net2World
                    local_target_pos_world = local_target_pos_world + \
                        dstep * pred_vec[0, :pos_dim]
                    local_target_pos_world = local_target_pos_world.detach().cpu().numpy()
                    local_target_ori = decode_ori(
                        pred_ori.detach().cpu().numpy()).flatten()

            # Optionally view with ROS
            if args.view_ros:
                all_objects = [Object(pos=Net2World * pose[0:POS_DIM], ori=inspection_ori_quat,
                                      radius=Net2World * radius) for pose, radius in
                               zip(object_poses_net, object_radii)]
                all_objects += [
                    Object(
                        pos=Net2World * start_pose_net[0:POS_DIM], radius=Net2World * Params.agent_radius, ori=start_pose_net[POS_DIM:]),
                    Object(
                        pos=Net2World * goal_pose_net[0:POS_DIM], radius=Net2World * goal_rot_radius.item(), ori=goal_pose_net[POS_DIM:]),
                    Object(
                        pos=Net2World * cur_pose_net[0:POS_DIM], radius=Net2World * Params.agent_radius, ori=cur_pose_net[POS_DIM:])
                ]
                agent_traj = np.vstack(
                    [Net2World * cur_pose_net, np.concatenate([Net2World * local_target_pos_world, cur_ori_quat])])
                if isinstance(object_forces, torch.Tensor):
                    object_forces = object_forces[0].detach().cpu().numpy()

                viz_3D_publisher.publish(objects=all_objects, agent_traj=agent_traj,
                                         expert_traj=None, object_colors=all_object_colors,
                                         object_forces=object_forces, force_colors_rgb=force_colors)

            # Apply low-pass filter to smooth out policy's sudden changes in orientation
            interp_rot = interpolate_rotations(start_quat=cur_ori_quat, stop_quat=local_target_ori, alpha=1.0)

            # Clip target EE position to bounds
            local_target_pos_world = np.clip(
                local_target_pos_world, a_min=ee_min_pos_world, a_max=ee_max_pos_world)

            # Publish is_intervene
            is_intervene_pub.publish(Bool(is_intervene))
            # Publish target pose
            if not DEBUG:
                target_pose = np.concatenate(
                    [local_target_pos_world, interp_rot])
                pose_pub.publish(pose_to_msg(target_pose, frame=ROBOT_FRAME))
                # print("cur pose: ", np.concatenate([cur_pos_world, cur_ori_quat]))
                # print("target pose: ", target_pose)
                # print("target ori: ", interp_rot)
                # print()
                # # Publish is_intervene
                # is_intervene_pub.publish(Bool(is_intervene))

            if DEBUG:
                cur_pos_world = local_target_pos_world
                cur_ori_quat = interp_rot

            if it % 2 == 0:
                print("Pos error: ", np.linalg.norm(local_target_pos_world - cur_pos_world) )
                print("Ori error: ", np.linalg.norm(np.arccos(np.abs(cur_ori_quat @ local_target_ori))) )
                print("Dpose: ", del_pose_running_avg.avg)
                print()

            it += 1
            prev_pose_world = np.copy(cur_pose_world)
            rospy.sleep(0.3)

        print(f"Finished! Error {pose_error} vs tol {pose_error_tol}, \nderror {del_pose_running_avg.avg} vs tol {del_pose_tol}")
        print("Opening gripper to release item")
        rospy.sleep(0.3)
        command_kinova_gripper(gripper_pub, cmd_open=True)  

        # Once item released, set extra mass back to 0
        for i in range(3): extra_mass_pub.publish(Float32(0.0))
        rospy.sleep(0.3)

