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
from scipy.spatial.transform import Slerp

from pybullet_tools.kuka_primitives import get_tool_link, BodyGrasp
import pybullet_data

import torch

from model import Policy, PolicyNetwork, pose_to_model_input, decode_ori
from train import perform_adaptation, DEVICE
from data_params import Params
from exp_params import *
from exp_utils import *
from elastic_band import Object
from viz_3D import Viz3DROSPublisher

pybullet_data_path = pybullet_data.getDataPath()
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger()
logging.getLogger('asyncio').setLevel(logging.WARNING)
# get_camera_pose()[0]
camera_pos = np.array([0.22864523, 1.29155352, 0.96121949])
# get_camera().target
camera_target = np.array([0.34110042452812195, -0.4951653480529785, 0.7741619348526001])

signal.signal(signal.SIGINT, sigint_handler)


def on_press(key):
    """
    Callback for keyboard events. Modifies global variables directly.

    :param key: pressed key
    :return:
    """
    global key_pressed, need_update, is_perturb_pos
    global get_EE_pose, local_target_pos_sim, local_target_ori
    global perturb_pos_traj_sim
    ROT_SPEED = np.deg2rad(20)
    POS_SPEED = 0.075

    if key == keyboard.Key.shift:
        is_perturb_pos = not is_perturb_pos

    elif key == keyboard.Key.space:
        if len(perturb_pos_traj_sim) > 0:
            key_pressed = False
            need_update = True
        else:
            key_pressed = not key_pressed

        # NOTE: in real world implementation, this would be a separate thread/process
        #   so robot can still continue to move
        return

    else:
        if key == keyboard.Key.ctrl:
            return

        local_target_pos_sim, local_target_ori = get_EE_pose()

        # initial state along trajectory
        if len(perturb_pos_traj_sim) == 0:
            perturb_pos_traj_sim.append(np.copy(local_target_pos_sim))

        if is_perturb_pos:
            try:
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

            except Exception as e:
                if key == keyboard.Key.up:
                    local_target_pos_sim[1] -= POS_SPEED
                    key_pressed = True
                elif key == keyboard.Key.down:
                    local_target_pos_sim[1] += POS_SPEED
                    key_pressed = True

        else:
            # Apply rotation perturbations wrt world frame -> left multiply the offset
            try:
                if key.char == "w":
                    key_pressed = True
                    local_target_ori = (
                            R.from_euler("z", ROT_SPEED, degrees=False) * R.from_quat(local_target_ori)).as_quat()
                elif key.char == "s":
                    key_pressed = True
                    local_target_ori = (
                            R.from_euler("z", -ROT_SPEED, degrees=False) * R.from_quat(local_target_ori)).as_quat()

                # flipped for y-axis due to rotated camera view
                elif key.char == "d":
                    key_pressed = True
                    local_target_ori = (
                            R.from_euler("x", ROT_SPEED, degrees=False) * R.from_quat(local_target_ori)).as_quat()
                elif key.char == "a":
                    key_pressed = True
                    local_target_ori = (
                            R.from_euler("x", -ROT_SPEED, degrees=False) * R.from_quat(local_target_ori)).as_quat()

            except:
                if key == keyboard.Key.up:
                    key_pressed = True
                    local_target_ori = (
                            R.from_euler("y", -ROT_SPEED, degrees=False) * R.from_quat(local_target_ori)).as_quat()
                elif key == keyboard.Key.down:
                    key_pressed = True
                    local_target_ori = (
                            R.from_euler("y", ROT_SPEED, degrees=False) * R.from_quat(local_target_ori)).as_quat()

        if key_pressed:
            perturb_pos_traj_sim.append(np.copy(local_target_pos_sim))


def load_scene():
    floor_ids = load_floor()
    robot = load_model(DRAKE_IIWA_URDF)

    table_pos = Point(x=0.17, y=0.5, z=-0.05)
    table = load_model(os.path.join(MODEL_DIRECTORY, "table_collision/table_skinny.urdf"),
                       pose=Pose(table_pos, Euler(0, 0, 0)),
                       is_abs_path=True)

    bowl_path = os.path.join(MODEL_DIRECTORY, "dinnerware/plate.urdf")
    init_bowl_pos = Point(x=-0.35, y=0.125, z=0.9)
    bowl = load_model(bowl_path,
                      pose=Pose(init_bowl_pos, Euler(0, 0, 0)),
                      is_abs_path=True)

    # visualize goal location for interpretability
    goal_item_path = os.path.join(MODEL_DIRECTORY, "dinnerware/plate_goal.urdf")
    goal_bowl = load_model(goal_item_path,
                           is_abs_path=True)

    # Scanner
    scanner_path = os.path.join(pybullet_data_path, "tray/traybox.urdf")
    scanner = load_model(scanner_path, is_abs_path=True)

    return floor_ids, robot, table, bowl, goal_bowl, scanner, table_pos, init_bowl_pos


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store', type=str, help="trained model name")
    parser.add_argument('--loaded_epoch', action='store', type=int)
    parser.add_argument('--view_ros', action='store_true', help="visualize 3D scene with ROS Rviz")
    parser.add_argument('--user', action='store', type=str, default="some_user", help="user name to save results")
    parser.add_argument('--trial', action='store', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    """
    The robot carries bowls to a drop-off zone represented by a floating
    green bowl. You should perturb both position and orientation of the robot
    SLIGHTLY BEFORE the robot passes the scanner on its way to the goal. You
    should perturb so the robot moves closer to the black scanner and reorients the bowl
    so that its bottom faces the scanner. NOTE that the final position of the bowl **DOES** MATTER
    unlike exp2. Initially, your perturbations will be in POSITION mode. To perturb
    orientation, press Shift. You can toggle between these modes.

        w                                  ^
    a   s   d                           <  v  >

    POSITION:
    w: up (+z (pybullet's coordinate system))
    s: down (-z)
    a: left (+x)
    d: right (-x)
    ^: forward (-y)
    v: backward (+y)

    ORIENTATION:
    w: +Ry (wrt pybullet world frame)
    s: -Ry
    a: -Rz
    d: +Rz
    ^: +Rx
    v: -Rx

    Once satisfied with the robot/bowls's position and orientation, press space,
    and watch the robot's updated behavior in the next episode without intervening.
    After, three new scanner poses will be used, and you should judge whether
    the robot's behavior correctly updates to match the new scanner.

    Since our policy predicts target orientations without consider its current,
    there is no "smoothness" in change of orientation. We need to enforce this
    by applying a low-pass filter to the orientation prediction. This is done
    using SLERP interpolation between current and predicted quaternion.

    This scene contains only two actual objects: table and scanner whose
    position/rotation preferences and rotational offsets are unknown and adaptable,
    which is the most realistic scenario.
    """
    connect(use_gui=True)

    # Draw global x,y,z axes
    draw_global_system()

    # Initialize camera pose (our view)
    set_camera_pose(camera_pos, camera_target)

    args = parse_arguments()
    argparse_dict = vars(args)

    # create folder to store user results
    user_dir = os.path.join("user_trials", args.user, "exp3_scanning_item")
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

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
    num_objects = 2
    object_idxs = np.arange(num_objects)  # NOTE: not training "object_types", purely object identifiers
    pos_obj_types = [None] * num_objects
    pos_requires_grad = [train_pos] * num_objects
    rot_obj_types = [None] * num_objects
    rot_requires_grad = [train_rot] * num_objects
    policy.init_new_objs(pos_obj_types=pos_obj_types, rot_obj_types=rot_obj_types,
                         pos_requires_grad=pos_requires_grad, rot_requires_grad=rot_requires_grad)

    # Custom experiment parameters
    dstep = 0.075
    table_height_sim = 3.5 / Sim2Net
    object_radii = np.array([2.0] * num_objects)
    goal_rot_radius = np.array([1.5])
    goal_pos_sim = np.array([1.80553318e+00, 1.1, 4.57226727e+00]) / Sim2Net
    obstacles = []
    # Need more adaptation steps because this example is more difficult
    custom_num_pos_net_updates = 20
    custom_num_rot_net_updates = 30

    # Load scene
    floor_ids, robot, table, bowl, goal_bowl, scanner, table_pos_sim, init_bowl_pos_sim = load_scene()
    table_pos_sim[2] = table_height_sim  # set a proper height for the table

    # Helper func to get robot pose in world frame
    tool_link = get_tool_link(robot)  # int
    global get_EE_pose
    get_EE_pose = lambda: (
        np.array(get_com_pose(robot, tool_link)[0]),
        np.array(get_com_pose(robot, tool_link)[1])
    )

    # Define fixed grasp with object
    grasp = BodyGrasp(bowl, grasp_pose, approach_pose, robot, tool_link)

    # Define start robot pose
    movable_joints = get_movable_joints(robot)
    set_joint_positions(robot, movable_joints, init_joints)
    start_pos_sim, start_ori_quat = get_EE_pose()
    start_pose_net = np.concatenate([Sim2Net * start_pos_sim, start_ori_quat])

    global cur_pos_sim
    cur_pos_sim = np.copy(start_pos_sim)

    # Define final goal pose (drop-off bin)
    goal_ori_quat = np.copy(start_ori_quat)
    goal_pose_net = np.concatenate([Sim2Net * goal_pos_sim, goal_ori_quat])
    set_pose(goal_bowl, Pose(goal_pos_sim))

    # Define various scanner poses to test generalization to diff obj poses
    base_obj_ori = R.from_quat(np.array([0.34202014, 0., 0., 0.93969262]))
    scanner_poses_sim = [
        (np.array([-0.1, -0.1, 1.1]), base_obj_ori),  # Initial human demo/perturb
        (np.array([-0.1, -0.1, 1.1]), base_obj_ori),  # Check updated behavior on same scene

        # Test generalization to diff scanner poses
        (np.array([-0.2, -0.1, 1.1]), base_obj_ori * R.from_euler('xyz', [0, np.pi / 4, 0])),
        (np.array([0.0, -0.2, 0.9]), base_obj_ori * R.from_euler('xyz', [np.pi / 2, 0, 0])),
        (np.array([0.0, 0.3, 1.1]), base_obj_ori * R.from_euler('xyz', [np.pi + np.pi / 4, 0, 0])),
    ]

    # Convert np arrays to torch tensors for model input
    start_tensor = torch.from_numpy(
        pose_to_model_input(start_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
    agent_radius_tensor = torch.tensor([Params.agent_radius], device=DEVICE).view(1, 1)
    goal_rot_radii = torch.from_numpy(goal_rot_radius).to(torch.float32).to(DEVICE).view(1, 1)
    object_radii_torch = torch.from_numpy(object_radii).to(torch.float32).to(DEVICE).view(num_objects, 1)
    object_idxs_tensor = torch.from_numpy(object_idxs).to(torch.long).to(DEVICE).unsqueeze(0)

    # Activate keybindings to listen for user keyboard input to perturb robot
    # and define global variables modified in the keyboard callback
    global key_pressed, need_update
    key_pressed = False
    need_update = False
    global is_perturb_pos, local_target_ori, local_target_pos_sim, perturb_pos_traj_sim
    is_perturb_pos = True  # allow user to switch between pos/ori perturbation
    local_target_pos_sim = None  # next immediate target position to track, user can set this directly
    local_target_ori = None
    perturb_pos_traj_sim = []
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    # Optionally view with ROS Rviz
    if args.view_ros:
        viz_3D_publisher = Viz3DROSPublisher(num_objects=num_objects, bounds=(Params.lb_3D, Params.ub_3D))
        table_scanner_colors = [
            (0, 255, 0),
            (0, 0, 255),
        ]
        all_object_colors = table_scanner_colors + [
            Params.start_color_rgb,
            Params.goal_color_rgb,
            Params.agent_color_rgb,
        ]
        force_colors = table_scanner_colors + [Params.goal_color_rgb]

    for trial_idx, (new_scanner_pos_sim, new_scanner_ori) in enumerate(scanner_poses_sim):
        # set new scanner pose, define overall tensor for all objects(table, scanner) in scene
        set_pose(scanner, Pose(new_scanner_pos_sim, new_scanner_ori.as_euler('xyz')))
        object_pos_net = np.array([Sim2Net * table_pos_sim, Sim2Net * new_scanner_pos_sim])
        object_oris = np.array([some_ori, new_scanner_ori.as_quat()])
        object_poses_net = np.concatenate([object_pos_net, object_oris], axis=-1)
        object_poses_tensor = torch.from_numpy(pose_to_model_input(object_poses_net)).to(torch.float32).to(DEVICE)
        objects_torch = torch.cat([object_poses_tensor, object_radii_torch], dim=-1).unsqueeze(0)

        # reset robot pose and src bowl
        set_joint_positions(robot, movable_joints, init_joints)
        set_pose(bowl, Pose(init_bowl_pos_sim))
        cur_pos_sim = np.copy(start_pos_sim)

        perturb_pos_traj_sim = []

        it = 0
        tol = 0.1
        error = np.Inf
        while error > tol:
            it += 1
            cur_pos_sim, cur_ori_quat = get_EE_pose()
            cur_pos_net = Sim2Net * cur_pos_sim
            cur_pose_net = np.concatenate([cur_pos_net, cur_ori_quat])
            error = np.linalg.norm(goal_pos_sim - cur_pos_sim)

            if need_update:
                dist = np.linalg.norm(perturb_pos_traj_sim[-1] - perturb_pos_traj_sim[0])
                T = max(2, int(np.ceil(dist / dstep)))  # 1 step for start, 1 step for goal at least
                perturb_pos_traj_sim = np.linspace(
                    start=perturb_pos_traj_sim[0], stop=perturb_pos_traj_sim[-1], num=T)

                # Construct overall perturb traj and adaptation data
                perturb_pos_traj_net = Sim2Net * perturb_pos_traj_sim
                # NOTE: cannot track arbitrary orientation traj, perturbation
                #   only defines a final, target orientation
                perturb_ori_traj = np.copy(cur_ori_quat)[np.newaxis, :].repeat(T, axis=0)
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

            if key_pressed:
                # local_target_pos_sim/ori already set by user in keyboard callback
                pass

            else:
                with torch.no_grad():
                    # Define "object" inputs into policy
                    # current
                    cur_pose_tensor = torch.from_numpy(
                        pose_to_model_input(cur_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
                    current = torch.cat([cur_pose_tensor, agent_radius_tensor], dim=-1).unsqueeze(1)
                    # goal
                    goal_tensor = torch.from_numpy(
                        pose_to_model_input(goal_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
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
                if isinstance(object_forces, torch.Tensor):
                    object_forces = object_forces[0].detach().cpu().numpy()
                viz_3D_publisher.publish(objects=all_objects, agent_traj=agent_traj,
                                         expert_traj=None, object_colors=all_object_colors,
                                         object_forces=object_forces, force_colors_rgb=force_colors)

            # Apply low-pass filter to smooth out policy's sudden changes in orientation
            key_times = [0, 1]
            rotations = R.from_quat(np.vstack([cur_ori_quat, local_target_ori]))
            slerp = Slerp(key_times, rotations)
            alpha = 0.3
            interp_rot = slerp([alpha])[0]  # alpha*cur_ori_quat + alpha*local_target_ori
            target_pose = (local_target_pos_sim, interp_rot.as_quat())
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

    disconnect()


if __name__ == '__main__':
    with HideOutput():
        main()
