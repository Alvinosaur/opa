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
import torch
import ipdb
import typing
from typing import *

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from pybullet_tools.kuka_primitives import get_tool_link, Attach, BodyGrasp
import pybullet_data

from model import Policy, PolicyNetwork, pose_to_model_input, decode_ori
from train import perform_adaptation, DEVICE
from data_params import Params
from exp_params import *
from exp_utils import *
from viz_3D import Viz3DROSPublisher
from elastic_band import Object

pybullet_data_path = pybullet_data.getDataPath()
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()
logging.getLogger('asyncio').setLevel(logging.WARNING)
# get_camera_pose()[0]
camera_pos = np.array([0.25489926, 1.31271563, 0.8570624])
# get_camera().target
camera_target = np.array([0.30520865321159363, -0.48658108711242676, 0.8581621050834656])

signal.signal(signal.SIGINT, sigint_handler)


def on_press(key):
    """
    Callback for keyboard events. Modifies global variables directly.

    :param key: pressed key
    :return:
    """
    global key_pressed, local_target_pos_sim, perturb_pos_traj_sim, \
        need_update, get_EE_pose
    global robot_trajectory, human_perturbed
    POS_SPEED = 0.075
    if key == keyboard.Key.space:
        if len(perturb_pos_traj_sim) > 0:
            key_pressed = False
            need_update = True
        else:
            key_pressed = not key_pressed

    else:
        if key == keyboard.Key.ctrl:
            return

        cur_pos_sim, cur_ori_quat = get_EE_pose()
        cur_pos_sim = np.array(cur_pos_sim)
        local_target_pos_sim = np.copy(cur_pos_sim)

        if len(perturb_pos_traj_sim) == 0:
            perturb_pos_traj_sim.append(np.copy(local_target_pos_sim))

        try:
            # up down left right
            if key.char == "w":
                local_target_pos_sim[2] += POS_SPEED
                key_pressed = True
            elif key.char == "s":
                local_target_pos_sim[2] -= POS_SPEED
                key_pressed = True

            # flipped for y-axis due to rotated camera view
            elif key.char == "a":
                local_target_pos_sim[0] += POS_SPEED
                key_pressed = True
            elif key.char == "d":
                local_target_pos_sim[0] -= POS_SPEED
                key_pressed = True

        except:
            if key == keyboard.Key.up:
                local_target_pos_sim[1] -= POS_SPEED
                key_pressed = True
            elif key == keyboard.Key.down:
                local_target_pos_sim[1] += POS_SPEED
                key_pressed = True

        if key_pressed:
            # record intervention trajectory
            robot_trajectory.append((np.copy(local_target_pos_sim), np.copy(cur_ori_quat)))
            human_perturbed.append(True)

            perturb_pos_traj_sim.append(np.copy(local_target_pos_sim))


def robot_table_surface_projections(table_bounds_sim: typing.List[Tuple],
                                    EE_pos_sim: np.ndarray, table_height_sim: float):
    """
    Projects agent position onto table surfaces within their bounds. Returns
    projected positions in Network space (multiplies position with Sim2Net)

    :param table_bounds_sim: table bounds in simulation space
    :param EE_pos_sim: agent position(s) in simulation space  (T x 3)
    :param table_height_sim: table height in simulation space
    :return: projected table poses in Network space (T x n_tables x 7)
    """
    if len(EE_pos_sim.shape) == 1:
        EE_pos_sim = EE_pos_sim[np.newaxis, :]  # add time dimension
    T = EE_pos_sim.shape[0]

    table_poses_projected_net = []
    for table_bound_sim in table_bounds_sim:
        # Table xy position is simply EE xy position clipped within table bounds
        table_pos_sim = np.copy(EE_pos_sim)
        table_pos_sim[:, 0] = np.clip(table_pos_sim[:, 0],
                                      a_min=table_bound_sim[0],
                                      a_max=table_bound_sim[1])
        table_pos_sim[:, 1] = np.clip(table_pos_sim[:, 1],
                                      a_min=table_bound_sim[2],
                                      a_max=table_bound_sim[3])

        # Table z position is table height
        table_pos_sim[:, 2] = table_height_sim

        # Convert to Network space and store
        table_pose_net = np.concatenate([Sim2Net * table_pos_sim,
                                         # orientation doesn't matter for this experiment
                                         some_ori[np.newaxis].repeat(T, axis=0)],
                                        axis=1)
        table_poses_projected_net.append(table_pose_net[:, np.newaxis, :])

    return np.concatenate(table_poses_projected_net, axis=1)


def load_scene():
    floor_ids = load_floor()
    robot = load_model(DRAKE_IIWA_URDF)
    shelf = load_model(os.path.join(pybullet_data_path, KIVA_SHELF_SDF),
                       is_abs_path=True)[0]
    set_pose(shelf, Pose(Point(x=-1, y=0.4, z=stable_z(shelf, floor_ids[0])),
                         Euler(0, 0, np.pi / 2)))

    # Load tables
    table_width = 0.4
    table_len = 1.2
    table1_pos = Point(x=0.3, y=0.5, z=0)
    # table_bounds: (minx, maxx, miny, maxy)
    table1_bounds = (table1_pos[0] - table_len / 2, table1_pos[0] + table_len / 2,
                     table1_pos[1] - table_width / 2, table1_pos[1] + table_width / 2)
    table1 = load_model(
        os.path.join(MODEL_DIRECTORY, "table_collision/table_skinny.urdf"),
        pose=Pose(table1_pos),
        is_abs_path=True)

    table2_pos = Point(x=0.3, y=-0.5, z=0)
    table2_bounds = (table2_pos[0] - table_len / 2, table2_pos[0] + table_len / 2,
                     table2_pos[1] - table_width / 2, table2_pos[1] + table_width / 2)
    table2 = load_model(
        os.path.join(MODEL_DIRECTORY, "table_collision/table_skinny.urdf"),
        pose=Pose(table2_pos),
        is_abs_path=True)

    table3_pos = Point(x=0.3 + 0.5, y=0, z=-0.001)
    table3_ori = Euler(0, 0, np.pi / 2)
    # rotated, so bounds change as well
    table3_bounds = (table3_pos[0] - table_width / 2, table3_pos[0] + table_width / 2,
                     table3_pos[1] - table_len / 2, table3_pos[1] + table_len / 2)
    table3 = load_model(
        os.path.join(MODEL_DIRECTORY, "table_collision/table_skinny.urdf"),
        pose=Pose(table3_pos, table3_ori),
        is_abs_path=True)

    table_positions_sim = [table1_pos, table2_pos, table3_pos]
    table_bounds = [table1_bounds, table2_bounds, table3_bounds]

    # Load source(robot-held) and goal(visualize goal) cups
    src_cup_path = os.path.join(pybullet_data_path, MODEL_DIRECTORY, "dinnerware/cup/cup.urdf")
    goal_cup_path = os.path.join(pybullet_data_path, MODEL_DIRECTORY, "dinnerware/cup/cup_goal.urdf")
    cup = load_model(src_cup_path, is_abs_path=True)
    if isinstance(cup, tuple):
        cup = cup[0]

    cup_goal = load_model(goal_cup_path, is_abs_path=True)
    if isinstance(cup_goal, tuple):
        cup_goal = cup_goal[0]

    return robot, shelf, table1, table2, table3, cup, cup_goal, table_positions_sim, table_bounds


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store', type=str, help="trained model name")
    parser.add_argument('--loaded_epoch', action='store', type=int)
    parser.add_argument('--view_ros', action='store_true', help="visualize 3D scene with ROS Rviz")
    parser.add_argument('--user', action='store', type=str, default="some_user", help="user name to save results")
    parser.add_argument('--is_baseline', action='store_true')
    parser.add_argument('--draw_saved', action='store_true')
    parser.add_argument('--trial', action='store', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    """
    The robot arm has already grabbed cup and is carrying it to person represented by
    a green/blue floating cup. The robot should ideally keep the cup low to the nearest table,
    and this preference can be expressed by pushing the robot down to the table. This
    can be done repeatedly pressing these keys to change position:
        w                                  ^
    a   s   d                           <  v  >

    w: up (+z (pybullet's coordinate system))
    s: down (-z)
    a: left (+x)
    d: right (-x)
    ^: forward (-y)
    v: backward (+y)

    You can press the Space bar at any time to stop/start the robot.

    Once satisfied with the robot's new position, press Space to release control
    back to the policy. The policy will adapt automatically from your intervention.
    There are three different episodes with different goals, and the robot's
    policy is transferred across each. This means that once you perturb the robot
    in episode 1, ideally the robot should maintain your preference for the
    subsequent episodes 2 and 3.

    This scene contains three actual objects: three different tables where the robot
    initially ignores them in position.
    You can also run with --is_baseline to see the baseline's behavior.
    """
    connect(use_gui=True)

    # Draw global x,y,z axes
    draw_global_system()

    # Initialize camera pose (our view)
    set_camera_pose(camera_pos, camera_target)

    args = parse_arguments()
    argparse_dict = vars(args)
    is_baseline = args.is_baseline  # use fixed, linear controller

    # create folder to store user results
    user_dir = os.path.join("user_trials", args.user, "exp1_cup_low_table")
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    is_baseline_str = "default" if is_baseline else "policy"
    saved_traj_path = os.path.join(user_dir, f'trajectories_{is_baseline_str}.npy')
    saved_human_perturbed_path = os.path.join(user_dir, f'human_perturbed_{is_baseline_str}.npy')

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
    # In exp1, robot doesn't know to stay close to table, so initialize pos_types as None
    # orientation isn't the focus of exp1, so ignore
    calc_rot, calc_pos = False, True
    train_rot, train_pos = False, True
    num_objects = 3
    object_idxs = np.arange(num_objects)  # NOTE: not training "object_types", purely object identifiers
    pos_obj_types = [None, None, None]
    pos_requires_grad = [True, True, True]  # For exp1 specifically, update only position preference features
    rot_obj_types = [Params.IGNORE_ROT_IDX, Params.IGNORE_ROT_IDX, Params.IGNORE_ROT_IDX]
    rot_requires_grad = [False, False, False]
    policy.init_new_objs(pos_obj_types=pos_obj_types, rot_obj_types=rot_obj_types,
                         pos_requires_grad=pos_requires_grad, rot_requires_grad=rot_requires_grad)

    # Custom experiment parameters
    dstep = 0.1
    table_height_sim = 3.5 / Sim2Net
    object_radii = np.array([1.0] * num_objects)
    goal_rot_radius = np.array([1.0])  # arbitrary, orientation doesn't matter in this task
    init_cup_pos = Point(x=-0.4, y=0.125, z=0.9)
    obstacles = []

    # Load scene
    robot, shelf, table1, table2, table3, cup, cup_goal, table_positions_sim, table_bounds_sim = load_scene()
    set_pose(cup, Pose(init_cup_pos))

    # Helper func to get robot pose in world frame
    tool_link = get_tool_link(robot)  # int
    global get_EE_pose
    get_EE_pose = lambda: (
        np.array(get_com_pose(robot, tool_link)[0]),
        np.array(get_com_pose(robot, tool_link)[1])
    )

    # Define fixed grasp with object
    grasp = BodyGrasp(cup, grasp_pose, approach_pose, robot, tool_link)

    # Define start robot pose
    movable_joints = get_movable_joints(robot)
    set_joint_positions(robot, movable_joints, init_joints)
    start_pos_sim, start_ori_quat = get_EE_pose()
    start_pose_net = np.concatenate([Sim2Net * start_pos_sim, start_ori_quat])

    global cur_pos_sim
    cur_pos_sim = np.copy(start_pos_sim)

    # Define goal poses (for three different trials of same environment)
    goal_configs = [
        ((R.from_euler("xyz", [0, 0, -np.pi / 2]) * R.from_quat(start_ori_quat)).as_quat(),
         np.array([0.25, 0.5, 1.0])),
        ((R.from_euler("xyz", [0, 0, -np.pi / 2]) * R.from_quat(start_ori_quat)).as_quat(),
         np.array([0.5, 0.5, 1.0])),
        ((R.from_euler("xyz", [0, 0, -5 * np.pi / 6]) * R.from_quat(start_ori_quat)).as_quat(),
         np.array([0.75, 0.2, 0.95])),
    ]
    if args.draw_saved:  # Optionally replay a specific trial/goal pose
        goal_configs = [goal_configs[args.trial]]

    # Optionally draw previous saved trajectory
    if args.draw_saved:
        trial = args.trial
        width = 7

        loaded_robot_traj = np.load(saved_traj_path, allow_pickle=True)[trial]
        loaded_human_perturbed = np.load(saved_human_perturbed_path, allow_pickle=True)[trial]

        for i in range(1, len(loaded_robot_traj)):
            color = RGBA(0.9, 0, 0, 1) if loaded_human_perturbed[i] else RGBA(0, 0.7, 0, 1)
            add_line(loaded_robot_traj[i - 1][0], loaded_robot_traj[i][0], color=color, width=width)

    # save trajectories for post debugging/viz
    all_robot_trajectories = []  # overall robot EE pose trajectory across all trials
    all_human_perturbed = []  # whether human perturbed at a given time step across all trials

    # activate keybindings to listen for user keyboard input to perturb robot
    # and define global variables modified in the keyboard callback
    global key_pressed, need_update
    key_pressed = False
    need_update = False
    global local_target_pos_sim, perturb_pos_traj_sim, robot_trajectory, human_perturbed
    local_target_pos_sim = None  # next immediate target position to track, user can set this directly
    perturb_pos_traj_sim = []  # store human perturb used for online adaptation
    robot_trajectory = []  # append to all_robot_trajectories after each trial
    human_perturbed = []
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    # Optionally view with ROS Rviz
    if args.view_ros:
        viz_3D_publisher = Viz3DROSPublisher(num_objects=num_objects, bounds=(Params.lb_3D, Params.ub_3D))
        table_colors = [
            (255, 0, 0),
            (0, 255, 0),  # randomly chosen
            (0, 0, 255),
        ]
        all_object_colors = table_colors + [
            Params.start_color_rgb,
            Params.goal_color_rgb,
            Params.agent_color_rgb,
        ]
        force_colors = table_colors + [Params.goal_color_rgb]

    # loop through all experiments and record results
    for trial_idx, (goal_ori_quat, goal_pos_sim) in enumerate(goal_configs):
        # reset robot pose and src cup
        set_joint_positions(robot, movable_joints, init_joints)
        set_pose(cup, Pose(init_cup_pos))
        cur_pos_sim = np.copy(start_pos_sim)
        prev_pos_sim = np.copy(start_pos_sim)

        # set new goal cup pose
        set_pose(cup_goal, Pose(goal_pos_sim))

        # delay briefly to let user understand new scene
        time.sleep(1)

        # reset saved trajectories and human perturb
        robot_trajectory = []
        human_perturbed = []
        perturb_pos_traj_sim = []

        # Define correctly-scaled goal pose, also to make motion kinematically feasible
        # for robot, have orientation change as position changes (otherwise
        # robot physically will not be able to reach goal pose if facing original orientation)
        goal_pose_net = np.concatenate([Sim2Net * goal_pos_sim, goal_ori_quat])
        key_times = [0, np.linalg.norm(start_pos_sim - goal_pos_sim)]
        rotations = R.from_quat(np.vstack([start_ori_quat, goal_ori_quat]))
        global_slerp = Slerp(key_times, rotations)

        # Convert np arrays to torch tensors for model input
        start_tensor = torch.from_numpy(
            pose_to_model_input(start_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
        agent_radius_tensor = torch.tensor([Params.agent_radius], device=DEVICE).view(1, 1)
        goal_rot_radii = torch.from_numpy(goal_rot_radius).to(DEVICE).view(1, 1)
        goal_tensor = torch.from_numpy(pose_to_model_input(goal_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
        object_radii_torch = torch.from_numpy(object_radii).to(torch.float32).to(DEVICE).view(num_objects, 1)
        object_idxs_tensor = torch.from_numpy(object_idxs).to(torch.long).to(DEVICE).unsqueeze(0)

        it = 0
        tol = 0.1
        error = np.Inf
        while error > tol:
            it += 1
            cur_pos_sim, cur_ori_quat = get_EE_pose()
            dir_with_mag = goal_pos_sim - cur_pos_sim
            error = np.linalg.norm(dir_with_mag)
            cur_pos_net = Sim2Net * cur_pos_sim
            cur_pose_net = np.concatenate([cur_pos_net, cur_ori_quat])

            # Kinematically feasible change in orientation so goal position can be reached
            tgt_time = np.linalg.norm(start_pos_sim - goal_pos_sim) - np.linalg.norm(goal_pos_sim - cur_pos_sim)
            tgt_time = np.clip(tgt_time, 0, key_times[1])
            local_target_ori = global_slerp([tgt_time])[0].as_quat()

            # Once human presses "Space bar", need_update->True to trigger policy adaptation
            if need_update and not is_baseline:
                # NOTE: fit line to the noisy perturbation trajectory, this
                #   assumption only valid if assuming perturbation shows *preferred behavior*
                #   which is a preferred pose. Cannot learn to imitate complex trajectory
                dist = np.linalg.norm(perturb_pos_traj_sim[-1] - perturb_pos_traj_sim[0])
                T = max(2, int(np.ceil(dist / dstep)))  # 1 step for start, 1 step for goal at least
                perturb_pos_traj_sim = np.linspace(
                    start=perturb_pos_traj_sim[0], stop=perturb_pos_traj_sim[-1], num=T)

                # Given desired EE positions from human perturb, need to calculate corresponding
                # projected object poses for flat tables
                table_poses_projected_net = robot_table_surface_projections(table_bounds_sim=table_bounds_sim,
                                                                            EE_pos_sim=perturb_pos_traj_sim,
                                                                            table_height_sim=table_height_sim)

                # Construct overall perturb traj and adaptation data
                perturb_pos_traj_net = Sim2Net * perturb_pos_traj_sim
                perturb_ori_traj = np.copy(cur_ori_quat)[np.newaxis, :].repeat(T, axis=0)
                perturb_traj = np.hstack([np.vstack(perturb_pos_traj_net), perturb_ori_traj])
                batch_data = [
                    (perturb_traj, start_pose_net, goal_pose_net, goal_rot_radius,
                     table_poses_projected_net, object_radii[np.newaxis].repeat(T, axis=0),
                     object_idxs)]

                # Update policy
                print("Attract and repel feats: ",
                      policy.policy_network.pos_pref_feat_train[Params.ATTRACT_IDX].detach().cpu().numpy(),
                      policy.policy_network.pos_pref_feat_train[Params.REPEL_IDX].detach().cpu().numpy())
                print("Old pref feats: ", torch.cat(policy.obj_pos_feats, dim=0).detach().cpu().numpy())
                perform_adaptation(policy=policy, batch_data=batch_data,
                                   train_pos=train_pos, train_rot=train_rot,
                                   n_adapt_iters=num_pos_net_updates, dstep=dstep,
                                   verbose=False, clip_params=True)
                print("New pref feats: ", torch.cat(policy.obj_pos_feats, dim=0).detach().cpu().numpy())

                # reset the intervention data
                perturb_pos_traj_sim = []
                need_update = False
                continue

            if key_pressed:
                # local_target_pos_sim/ori already set by user in keyboard callback
                pass

            else:
                robot_trajectory.append((cur_pos_sim, cur_ori_quat))
                human_perturbed.append(False)

                if is_baseline:  # direct lin interp to the goal
                    num_steps = np.ceil(error / dstep)
                    local_target_pos_sim = cur_pos_sim + dir_with_mag / num_steps

                else:  # policy, take one-step action
                    # Compute robot EE -> table projections
                    table_poses_projected_net = robot_table_surface_projections(
                        table_bounds_sim=table_bounds_sim, EE_pos_sim=cur_pos_sim,
                        table_height_sim=table_height_sim)[0]  # remove time dimension
                    table_poses_projected_tensor = torch.from_numpy(
                        pose_to_model_input(table_poses_projected_net)).to(torch.float32).to(
                        DEVICE)
                    objects_torch = torch.cat([table_poses_projected_tensor, object_radii_torch], dim=-1).unsqueeze(0)

                    with torch.no_grad():
                        # Define "object" inputs into policy
                        # current agent
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
                        pred_vec, pred_ori, object_forces = policy(current=current,
                                                                   start=start_rot_objects,
                                                                   goal=goal_objects, goal_rot=goal_rot_objects,
                                                                   objects=objects_torch,
                                                                   object_indices=object_idxs_tensor,
                                                                   calc_rot=calc_rot,
                                                                   calc_pos=calc_pos)
                        local_target_pos_sim = cur_pose_tensor[0, :pos_dim] / Sim2Net
                        try:
                            local_target_pos_sim = local_target_pos_sim + dstep * pred_vec[0, :pos_dim]
                            local_target_pos_sim = local_target_pos_sim.detach().cpu().numpy()
                        except TypeError:
                            # human intervened right here and local_target_pos_sim is now a numpy array
                            # to avoid modifying the human intervention, do not apply pred_vec
                            pass

            # Optionally view with ROS
            if args.view_ros:
                all_objects = [Object(pos=pose[0:POS_DIM], ori=pose[POS_DIM:],
                                      radius=radius) for pose, radius in
                               zip(table_poses_projected_net, object_radii)]
                all_objects += [
                    Object(pos=start_pose_net[0:POS_DIM], radius=Params.agent_radius, ori=start_pose_net[POS_DIM:]),
                    Object(pos=goal_pose_net[0:POS_DIM], radius=Params.agent_radius, ori=goal_pose_net[POS_DIM:]),
                    Object(pos=cur_pose_net[0:POS_DIM], radius=Params.agent_radius, ori=cur_pose_net[POS_DIM:])
                ]
                agent_traj = np.vstack([cur_pose_net, np.concatenate([Sim2Net * local_target_pos_sim, cur_ori_quat])])
                if isinstance(object_forces, torch.Tensor):
                    object_forces = object_forces[0].detach().cpu().numpy()
                viz_3D_publisher.publish(objects=all_objects, agent_traj=agent_traj,
                                         expert_traj=None, object_colors=all_object_colors,
                                         object_forces=object_forces, force_colors_rgb=force_colors)

            # Apply generated action
            target_pose = (local_target_pos_sim, local_target_ori)
            # During planning, IK sampled, so save original world state and restore after planning
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

        all_robot_trajectories.append(robot_trajectory)
        all_human_perturbed.append(human_perturbed)

    if os.path.exists(saved_traj_path):
        input("You sure you want to overwrite the saved trajectory? ")

    np.save(saved_traj_path, all_robot_trajectories)
    np.save(saved_human_perturbed_path, all_human_perturbed)

    disconnect()


if __name__ == '__main__':
    with HideOutput():
        main()
