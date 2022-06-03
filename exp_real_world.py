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

import rospy
from hri_tasks_msgs.msg import PoseSync

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import torch

from model import Policy, PolicyNetwork, pose_to_model_input, decode_ori
from train import perform_adaptation, DEVICE
from data_params import Params
from exp_params import *
from exp_utils import *
from elastic_band import Object
from viz_3D import Viz3DROSPublisher
from kinova_robot import KinovaRobot


Net2World = 1.0
World2Net = 1 / Net2World

# TODO: Custom experiment parameters
num_objects = 5
dstep = 0.075
object_radii = np.array([2.0] * num_objects)
goal_rot_radius = np.array([1.5])
goal_pos_world = np.array([5, 5, 5])
obstacles = []
# Need more adaptation steps because this example is more difficult
custom_num_pos_net_updates = 20
custom_num_rot_net_updates = 30

# Global info updated by callbacks
cur_joints, cur_pos_world, cur_ori = None, None, None
intervene_robot_poses = []
intervene_object_poses = [[] for _ in range(num_objects)]
object_pos_world = np.zeros((num_objects, 3))  # updated by callbacks
object_oris = np.array([0, 0, 0, 1.]).view(1, 1).repeat(num_objects, axis=0)
is_intervene = False
need_update = False


def is_intervene_cb(msg):
    if is_intervene == msg.is_intervene:
        return
    elif is_intervene:
        print("Intervene ended")
        need_update = True
    else:
        print("Intervene started")

    is_intervene = msg.is_intervene


def obj_pose_cb(msg):
    # TODO:
    if is_intervene:
        pass


def robot_pose_cb(msg):
    cur_pos_world = np.array(msg.pose.position)
    cur_ori = np.array(msg.pose.orientation)
    if is_intervene:
        intervene_robot_poses.append((cur_pos_world, cur_ori))


def robot_joints_cb(msg):
    cur_joints = np.array(msg.joints)


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


def main():
    global object_pos_world
    args = parse_arguments()
    argparse_dict = vars(args)

    # Setup ROS
    rospy.Subscriber('/robot_cartesian_pose', PoseSync, obj_pose_cb),

    # 5 Objects' poses
    rospy.Subscriber("/object0_pose_sync", PoseSync,
                     obj_pose_cb, queue_size=10)
    rospy.Subscriber("/object1_pose_sync", PoseSync,
                     obj_pose_cb, queue_size=10)
    rospy.Subscriber("/object2_pose_sync", PoseSync,
                     obj_pose_cb, queue_size=10)
    rospy.Subscriber("/object3_pose_sync", PoseSync,
                     obj_pose_cb, queue_size=10)
    rospy.Subscriber("/object4_pose_sync", PoseSync,
                     obj_pose_cb, queue_size=10)

    # Is-intervene Handler? subscriber, service, or action server?
    # TODO

    # Target joint topic specific to kinova
    joint_target_pub = rospy.Publisher(
        "/kinova_joint_ref", PoseSync, queue_size=10)

    robot = KinovaRobot()

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

    # Define start robot pose
    movable_joints = get_movable_joints(robot)
    set_joint_positions(robot, movable_joints, init_joints)
    start_pos_world, start_ori_quat = robot.get_EE_pose()
    start_pose_net = np.concatenate(
        [start_pos_world * World2Net, start_ori_quat])

    # Define final goal pose (drop-off bin)
    goal_ori_quat = np.copy(start_ori_quat)
    goal_pose_net = np.concatenate([goal_pos_world * World2Net, goal_ori_quat])

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

    # Optionally view with ROS Rviz
    if args.view_ros:
        viz_3D_publisher = Viz3DROSPublisher(num_objects=num_objects)
        # TODO:
        object_colors = [
            (0, 255, 0),
            (0, 0, 255),
        ]
        all_object_colors = object_colors + [
            Params.start_color_rgb,
            Params.goal_color_rgb,
            Params.agent_color_rgb,
        ]
        force_colors = object_colors + [Params.goal_color_rgb]

    it = 0
    tol = 0.1
    error = np.Inf
    while not rospy.is_shutdown() and error > tol:
        if cur_pos_world is None:
            rospy.sleep(0.1)
            continue

        # Get current object poses and convert to net input
        object_pos_net = World2Net * object_pos_world
        object_poses_net = np.concatenate(
            [object_pos_net, object_oris], axis=-1)
        object_poses_tensor = torch.from_numpy(pose_to_model_input(
            object_poses_net)).to(torch.float32).to(DEVICE)
        objects_torch = torch.cat(
            [object_poses_tensor, object_radii_torch], dim=-1).unsqueeze(0)

        perturb_pos_traj_sim = []

        it += 1
        cur_pos_net = Net2World * cur_pos_world
        cur_pose_net = np.concatenate([cur_pos_net, cur_ori])
        error = np.linalg.norm(goal_pos_world - cur_pos_world)

        if need_update:
            dist = np.linalg.norm(
                perturb_pos_traj_sim[-1] - perturb_pos_traj_sim[0])
            # 1 step for start, 1 step for goal at least
            T = max(2, int(np.ceil(dist / dstep)))
            perturb_pos_traj_sim = np.linspace(
                start=perturb_pos_traj_sim[0], stop=perturb_pos_traj_sim[-1], num=T)

            # Construct overall perturb traj and adaptation data
            perturb_pos_traj_net = Net2World * perturb_pos_traj_sim
            # NOTE: cannot track arbitrary orientation traj, perturbation
            #   only defines a final, target orientation
            perturb_ori_traj = np.copy(cur_ori)[
                np.newaxis, :].repeat(T, axis=0)
            perturb_pose_traj_net = np.hstack(
                [np.vstack(perturb_pos_traj_net), perturb_ori_traj])
            batch_data = [
                (perturb_pose_traj_net, start_pose_net, goal_pose_net, goal_rot_radius,
                    object_poses_net[np.newaxis].repeat(T, axis=0),
                    object_radii[np.newaxis].repeat(T, axis=0),
                    object_idxs)]

            # Update policy in position first, then orientation after with diff num iters
            perform_adaptation(policy=policy, batch_data=batch_data,
                               train_pos=True, train_rot=False,
                               n_adapt_iters=custom_num_pos_net_updates, dstep=dstep,
                               verbose=False, clip_params=True)
            # TODO: do we need to truncate traj to only be 2 timesteps for ori?
            #   that's what we did before, but may not be necessary
            T = 2
            batch_data = [
                (perturb_pose_traj_net[[0, -1], :], start_pose_net, goal_pose_net, goal_rot_radius,
                    object_poses_net[np.newaxis].repeat(T, axis=0),
                    object_radii[np.newaxis].repeat(T, axis=0),
                    object_idxs)]
            perform_adaptation(policy=policy, batch_data=batch_data,
                               train_pos=False, train_rot=True,
                               n_adapt_iters=custom_num_rot_net_updates, dstep=dstep,
                               verbose=False, clip_params=True)

            # reset the intervention data
            perturb_pos_traj_sim = []
            need_update = False
            break  # start new trial with updated weights

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
                local_target_pos_sim = cur_pose_tensor[0,
                                                       :pos_dim] / Net2World
                local_target_pos_sim = local_target_pos_sim + \
                    dstep * pred_vec[0, :pos_dim]
                local_target_pos_sim = local_target_pos_sim.detach().cpu().numpy()
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
                [cur_pose_net, np.concatenate([Net2World * local_target_pos_sim, cur_ori])])
            if isinstance(object_forces, torch.Tensor):
                object_forces = object_forces[0].detach().cpu().numpy()
            viz_3D_publisher.publish(objects=all_objects, agent_traj=agent_traj,
                                     expert_traj=None, object_colors=all_object_colors,
                                     object_forces=object_forces, force_colors_rgb=force_colors)

        # Apply low-pass filter to smooth out policy's sudden changes in orientation
        key_times = [0, 1]
        rotations = R.from_quat(
            np.vstack([cur_ori, local_target_ori]))
        slerp = Slerp(key_times, rotations)
        alpha = 0.3
        # alpha*cur_ori_quat + alpha*local_target_ori
        interp_rot = slerp([alpha])[0]
        target_pose = (local_target_pos_sim, interp_rot.as_quat())

        # Publish target pose
        cur_pose = (cur_pos_world, cur_ori)
        target_joints = robot.iterative_IK(
            thetas=cur_joints, xs=cur_pose, xd=target_pose)
        joint_target_pub.publish(joints_to_msg(target_joints))


if __name__ == '__main__':
    main()
