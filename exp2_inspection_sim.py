"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import numpy as np
import os
from pynput import keyboard
import logging
import argparse
import ipdb

from scipy.spatial.transform import Rotation as R

import torch

from pybullet_tools.kuka_primitives import get_tool_link, Attach, BodyGrasp

import pybullet_data

from model import Policy, PolicyNetwork, pose_to_model_input, decode_ori
from train import perform_adaptation, DEVICE
from data_params import Params
from exp_params import *
from exp_utils import *
from elastic_band import Object
from viz_3D import Viz3DROSPublisher

signal.signal(signal.SIGINT, sigint_handler)

pybullet_data_path = pybullet_data.getDataPath()
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger()
logging.getLogger('asyncio').setLevel(logging.WARNING)
# get_camera_pose()[0]
camera_pos = np.array([0.17462485, 1.29283311, 1.02121961])
# get_camera().target
camera_target = np.array([0.17468099296092987, -0.8952553272247314, 0.792593777179718])


def on_press(key):
    """
    Callback for keyboard events. Modifies global variables directly.

    :param key: pressed key
    :return:
    """
    global key_pressed, need_update
    global get_EE_pose, local_target_ori
    ROT_SPEED = np.deg2rad(10)

    if key == keyboard.Key.space:
        if len(perturb_traj_sim) > 0:
            key_pressed = False
            need_update = True
        else:
            key_pressed = not key_pressed
        return

    else:
        if key == keyboard.Key.ctrl:
            return

        key_pressed = True
        cur_pos, cur_ori_quat = get_EE_pose()  # world -> EE

        # initial state along trajectory
        if len(perturb_traj_sim) == 0:
            perturb_traj_sim.append([cur_pos, np.copy(cur_ori_quat)])

        # Apply rotation perturbations wrt world frame -> left multiply the offset
        try:
            # up down left right
            if key.char == "w":
                local_target_ori = (R.from_euler("y", ROT_SPEED, degrees=False) * R.from_quat(cur_ori_quat)).as_quat()
            elif key.char == "s":
                local_target_ori = (R.from_euler("y", -ROT_SPEED, degrees=False) * R.from_quat(cur_ori_quat)).as_quat()

            # flipped for y-axis due to rotated camera view
            elif key.char == "d":
                local_target_ori = (R.from_euler("z", ROT_SPEED, degrees=False) * R.from_quat(cur_ori_quat)).as_quat()
            elif key.char == "a":
                local_target_ori = (R.from_euler("z", -ROT_SPEED, degrees=False) * R.from_quat(cur_ori_quat)).as_quat()

        except:
            if key == keyboard.Key.up:
                local_target_ori = (R.from_euler("x", ROT_SPEED, degrees=False) * R.from_quat(cur_ori_quat)).as_quat()
            elif key == keyboard.Key.down:
                local_target_ori = (R.from_euler("x", -ROT_SPEED, degrees=False) * R.from_quat(cur_ori_quat)).as_quat()

        perturb_traj_sim.append([cur_pos, np.copy(local_target_ori)])


def load_scene():
    floor_ids = load_floor()
    robot = load_model(DRAKE_IIWA_URDF)

    table_pos = Point(x=0.17, y=0.5, z=-0.15)  # np.ndarray
    table = load_model(os.path.join(MODEL_DIRECTORY, "table_collision/table_skinny.urdf"),
                       pose=Pose(table_pos, Euler(0, 0, 0)),
                       is_abs_path=True)

    inspection_zone_path = os.path.join(MODEL_DIRECTORY, "tray/traybox_transparent.urdf")
    inspection_zone = load_model(inspection_zone_path, is_abs_path=True)
    tray_pos_sim = np.array([0.16, 0.4, 0.5])
    set_pose(inspection_zone, Pose(Point(x=tray_pos_sim[0],
                                         y=tray_pos_sim[1],
                                         z=stable_z(inspection_zone, table))))

    pan_path = os.path.join(MODEL_DIRECTORY, "dinnerware/pan_tefal.urdf")
    init_pan_pos = Point(x=-0.4, y=0.125, z=0.9)
    pan = load_model(pan_path,
                     pose=Pose(init_pan_pos, Euler(0, 0, 0)),
                     is_abs_path=True)

    return floor_ids, robot, table, inspection_zone, pan, tray_pos_sim, init_pan_pos


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store', type=str, help="trained model name")
    parser.add_argument('--loaded_epoch', action='store', type=int)
    parser.add_argument('--view_ros', action='store_true', help="visualize 3D scene with ROS Rviz")
    parser.add_argument('--user', action='store', type=str, default="some_user", help="user name to save results")
    parser.add_argument('--draw_saved', action='store_true')
    parser.add_argument('--trial', action='store', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    """
    The robot presents pans to you(QA inspector). Once the robot moves above the center
    of the inspection tray, you should perturb the robot/pan's orientation
    to your desired orientation. NOTE that the final position of the pan DOES NOT MATTER,
    so just focus on achieving the desired orientation, regardless of where the pan moves.

        w                                  ^
    a   s   d                           <  v  >

    w: +Ry (wrt pybullet world frame)
    s: -Ry
    a: -Rz
    d: +Rz
    ^: +Rx
    v: -Rx

    Once satisfied with the robot/pan's orientation, press space, and watch the robot's
    updated behavior in the next episode without intervening. This pattern
    will continue of 1. Apply perturbation 2. Observe updated behavior 3. Repeat.

    This scene contains only one actual object: the inspection tray where the
    model doesn't know to care about its orientation and what rotational offset
    is desired.
    """
    connect(use_gui=True)

    # Draw global x,y,z axes
    draw_global_system()

    # Initialize camera pose (our view)
    set_camera_pose(camera_pos, camera_target)

    args = parse_arguments()
    argparse_dict = vars(args)

    # create folder to store user results
    user_dir = os.path.join("user_trials", args.user, "exp2_inspection")
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    saved_traj_path = os.path.join(user_dir, f'trajectories.npy')

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

    # Define scene objects and whether or not we care about their pos/ori
    # In exp2, only a single object: the inspection tray, whose position preference
    # is predefined as "Attract". Rotation preference and offset are unknown
    calc_rot, calc_pos = True, True
    train_rot, train_pos = True, False
    num_objects = 1
    object_idxs = np.arange(num_objects)  # NOTE: not "object_types" from training, purely object identifiers
    pos_obj_types = [Params.ATTRACT_IDX]
    pos_requires_grad = [train_pos]
    rot_obj_types = [None]  # Rotation preference and offset unknown
    rot_requires_grad = [train_rot]
    policy.init_new_objs(pos_obj_types=pos_obj_types, rot_obj_types=rot_obj_types,
                         pos_requires_grad=pos_requires_grad, rot_requires_grad=rot_requires_grad)

    # Custom experiment parameters
    dstep = 0.075
    object_radii = np.array([2.0])  # TODO: tune this
    goal_rot_radius = np.array([0.5])
    goal_pos_sim = np.array([0.421, 0.124, 0.714])
    obstacles = []

    # Load scene
    floor_ids, robot, table, inspection_zone, pan, inspection_pos_sim, init_pan_pos_sim = load_scene()

    # Helper func to get robot pose in world frame
    tool_link = get_tool_link(robot)  # int
    global get_EE_pose
    get_EE_pose = lambda: (
        np.array(get_com_pose(robot, tool_link)[0]),
        np.array(get_com_pose(robot, tool_link)[1])
    )

    # Define fixed grasp with object
    grasp = BodyGrasp(pan, grasp_pose, approach_pose, robot, tool_link)

    # Three different pan orientations that user should demonstrate
    desired_pan_orientations = [
        R.from_euler("ZYX", [-90, -45, 0], degrees=True).as_euler("xyz"),
        R.from_quat([0.4799, 0.3908, -0.1120, 0.7774]).as_euler("xyz"),
        R.from_quat([-0.3441, -0.2477, -0.0641, 0.9033]).as_euler("xyz"),
    ]

    # Optionally draw previously saved trajectory
    if args.draw_saved:
        trial = args.trial
        loaded_pan_traj = np.load(saved_traj_path, allow_pickle=True)[trial]
        for i in range(len(loaded_pan_traj)):
            set_pose(pan, loaded_pan_traj[i])
            time.sleep(0.02)

    # Define start robot pose
    movable_joints = get_movable_joints(robot)
    set_joint_positions(robot, movable_joints, init_joints)
    start_pos_sim, start_ori_quat = get_EE_pose()
    start_pose_net = np.concatenate([Sim2Net * start_pos_sim, start_ori_quat])

    # Define final goal pose
    goal_ori_quat = np.copy(start_ori_quat)
    goal_pose_net = np.concatenate([Sim2Net * goal_pos_sim, goal_ori_quat])

    # Define inspection zone pose
    object_ori = start_ori_quat
    object_poses_net = np.concatenate([Sim2Net * inspection_pos_sim, object_ori])[np.newaxis, :]

    # Convert np arrays to torch tensors for model input
    start_tensor = torch.from_numpy(
        pose_to_model_input(start_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
    agent_radius_tensor = torch.tensor([Params.agent_radius], device=DEVICE).view(1, 1)
    goal_rot_radii = torch.from_numpy(goal_rot_radius).to(torch.float32).to(DEVICE).view(1, 1)
    goal_tensor = torch.from_numpy(pose_to_model_input(goal_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
    object_radii_torch = torch.from_numpy(object_radii).to(torch.float32).to(DEVICE).view(1, num_objects, 1)
    object_idxs_tensor = torch.from_numpy(object_idxs).to(torch.long).to(DEVICE).unsqueeze(0)
    object_poses_torch = torch.from_numpy(
        pose_to_model_input(object_poses_net)).to(torch.float32).to(DEVICE).unsqueeze(0)
    objects_torch = torch.cat([object_poses_torch, object_radii_torch], dim=-1)

    # Activate keybindings to listen for user keyboard input to perturb robot
    # and define global variables modified in the keyboard callback
    global key_pressed, need_update, local_target_ori, perturb_traj_sim
    key_pressed = False
    need_update = False
    local_target_ori = None
    perturb_traj_sim = []
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Optionally view with ROS Rviz
    if args.view_ros:
        viz_3D_publisher = Viz3DROSPublisher(num_objects=num_objects, bounds=(Params.lb_3D, Params.ub_3D))
        pan_color = [(255, 0, 0)]
        all_object_colors = pan_color + [
            Params.start_color_rgb,
            Params.goal_color_rgb,
            Params.agent_color_rgb,
        ]

    all_pan_trajectories = []
    for trial_idx in range(len(desired_pan_orientations)):
        # For convenience of user interaction, reset learned rotation features
        policy.init_new_objs(pos_obj_types=pos_obj_types, rot_obj_types=rot_obj_types,
                             pos_requires_grad=pos_requires_grad, rot_requires_grad=rot_requires_grad)

        # Interleave 1. human perturb and 2. visualize updated policy behavior
        for _ in range(2):
            perturb_traj_sim = []
            pan_trajectory = []

            # reset position of arm/pan
            set_pose(pan, Pose(init_pan_pos_sim))
            set_joint_positions(robot, movable_joints, init_joints)

            it = 0
            tol = 0.15
            error = np.Inf
            while error > tol or key_pressed or need_update:
                it += 1
                cur_pos_sim, cur_ori_quat = get_EE_pose()
                cur_pos_net = Sim2Net * cur_pos_sim
                cur_pose_net = np.concatenate([cur_pos_net, cur_ori_quat])
                error = np.linalg.norm(goal_pos_sim - cur_pos_sim)

                if need_update:
                    # Construct overall perturb traj and adaptation data
                    # NOTE: Ignore position trajectory, just use initial position before
                    #   intervention, but set desired orientation since human only
                    #   cares about desired orientation. This hack is only necessary
                    #   in simulation bc using keyboard control is difficult. In reality,
                    #   we can easily perturb orientation while maintaining position
                    T = 2
                    init_perturb_pos_net = Sim2Net * perturb_traj_sim[0][0]
                    perturb_pos_traj_net = np.vstack([init_perturb_pos_net, init_perturb_pos_net])
                    perturb_ori_traj = np.vstack([perturb_traj_sim[-1][1], perturb_traj_sim[-1][1]])
                    perturb_traj = np.hstack([perturb_pos_traj_net, perturb_ori_traj])
                    batch_data = [
                        (perturb_traj, start_pose_net, goal_pose_net, goal_rot_radius,
                         object_poses_net[np.newaxis].repeat(T, axis=0),
                         object_radii[np.newaxis].repeat(T, axis=0),
                         object_idxs)]

                    # Update policy
                    print("Care and Ignore feats: ",
                          policy.policy_network.rot_pref_feat_train[Params.CARE_ROT_IDX].detach().cpu().numpy(),
                          policy.policy_network.rot_pref_feat_train[Params.IGNORE_ROT_IDX].detach().cpu().numpy())
                    print("Old pref feats: ", torch.cat(policy.obj_rot_feats, dim=0).detach().cpu().numpy())
                    print("New rot offset: ", torch.stack(policy.obj_rot_offsets).detach().cpu().numpy())
                    perform_adaptation(policy=policy, batch_data=batch_data,
                                       train_pos=train_pos, train_rot=train_rot,
                                       n_adapt_iters=num_rot_net_updates, dstep=dstep,
                                       verbose=False, clip_params=True)
                    print("New pref feats: ", torch.cat(policy.obj_rot_feats, dim=0).detach().cpu().numpy())
                    print("New rot offset: ", torch.stack(policy.obj_rot_offsets).detach().cpu().numpy())

                    # reset the intervention data
                    perturb_traj_sim = []
                    need_update = False
                    break

                if key_pressed:
                    # User only sets desired orientation
                    local_target_pos_sim = np.copy(cur_pos_sim)

                else:
                    pan_trajectory.append(get_pose(pan))
                    with torch.no_grad():
                        # Define "object" inputs into policy
                        # current
                        cur_pose_tensor = torch.from_numpy(
                            pose_to_model_input(cur_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
                        current = torch.cat([cur_pose_tensor, agent_radius_tensor], dim=-1).unsqueeze(1)
                        # goal
                        goal_radii = goal_radius_scale * torch.norm(
                            goal_tensor[:, :pos_dim] - cur_pose_tensor[:, :pos_dim],
                            dim=-1).unsqueeze(0)
                        goal_rot_objects = torch.cat([goal_tensor, goal_rot_radii], dim=-1).unsqueeze(1)
                        goal_objects = torch.cat([goal_tensor, goal_radii], dim=-1).unsqueeze(1)
                        # start
                        start_rot_radii = torch.norm(start_tensor[:, :pos_dim] - cur_pose_tensor[:, :pos_dim],
                                                     dim=-1).unsqueeze(0)
                        start_rot_objects = torch.cat([start_tensor, start_rot_radii], dim=-1).unsqueeze(1)

                        # Get policy output, form into action
                        pred_vec, pred_ori, _ = policy(current=current,
                                                       start=start_rot_objects,
                                                       goal=goal_objects, goal_rot=goal_rot_objects,
                                                       objects=objects_torch,
                                                       object_indices=object_idxs_tensor,
                                                       calc_rot=calc_rot,
                                                       calc_pos=calc_pos)
                        local_target_pos_sim = cur_pose_tensor[0, :pos_dim] / Sim2Net
                        local_target_pos_sim = local_target_pos_sim + dstep * pred_vec[0, :pos_dim]
                        local_target_pos_sim = local_target_pos_sim.detach().cpu().numpy()
                        local_target_ori = decode_ori(pred_ori.detach().cpu().numpy()).flatten()

                # Optionally view with ROS
                if args.view_ros:
                    all_objects = [Object(pos=pose[0:POS_DIM], ori=pose[POS_DIM:],
                                          radius=radius) for pose, radius in
                                   zip(object_poses_net, object_radii)]
                    all_objects += [
                        Object(pos=start_pose_net[0:POS_DIM], radius=Params.agent_radius, ori=start_pose_net[POS_DIM:]),
                        Object(pos=goal_pose_net[0:POS_DIM], radius=Params.agent_radius, ori=goal_pose_net[POS_DIM:]),
                        Object(pos=cur_pose_net[0:POS_DIM], radius=Params.agent_radius, ori=cur_pose_net[POS_DIM:])
                    ]
                    agent_traj = np.vstack(
                        [cur_pose_net, np.concatenate([Sim2Net * local_target_pos_sim, cur_ori_quat])])
                    viz_3D_publisher.publish(objects=all_objects, agent_traj=agent_traj,
                                             expert_traj=None, object_colors=all_object_colors)

                target_pose = (local_target_pos_sim, local_target_ori)
                saved_world = WorldSaver()
                try:
                    command = plan_step(robot, tool_link, target_pose=target_pose,
                                        obstacles=obstacles, grasp=grasp,
                                        custom_limits=custom_limits, ik_kwargs=ik_kwargs)
                except Exception as e:
                    print("Failed to move because: {}".format(e))
                    ipdb.set_trace()
                    continue
                saved_world.restore()
                update_state()
                command.refine(num_steps=10).execute(time_step=0.01)

            all_pan_trajectories.append(pan_trajectory)

    if os.path.exists(saved_traj_path):
        input("You sure you want to overwrite the saved trajectory? ")

    np.save(saved_traj_path, all_pan_trajectories)

    disconnect()


if __name__ == '__main__':
    with HideOutput():
        main()
