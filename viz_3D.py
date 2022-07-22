"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import rospy
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped, Point
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import os
import random
import argparse
from typing import *
import json
import ipdb

from elastic_band import Object
from model import Policy, PolicyNetwork, pose_to_model_input, decode_ori
from train import DEVICE
from data_params import Params
from exp_params import *
from exp_utils import pose_to_msg, msg_to_pose

seed = 444
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.set_printoptions(suppress=True)  # disable scientific notation
FORCE_VEC_SCALE = 1
POS_DIM = 3
ROT_DIM = 6


def publish_object_forces(pub: rospy.Publisher, cur_pos: np.ndarray,
                          object_forces: np.ndarray, color_rgbs: List[Tuple[float, float, float]],
                          frame):
    """
    Publish function to view position "forces" from each object/node
    on the agent.

    :param pub: ROS publisher of arrow markers
    :param cur_pos: Current agent position
    :param object_forces: Forces from each object/node
    :param color_rgbs: RGB colors for each object
    :return:
    """
    marker_array_msg = MarkerArray()

    vec_mags = np.linalg.norm(object_forces, axis=-1, keepdims=True)
    object_forces /= vec_mags
    obj_weights = vec_mags / vec_mags.sum(0)[:, np.newaxis]

    for i, vec in enumerate(object_forces):
        marker = Marker()
        marker.header.frame_id = frame
        marker.id = i
        marker.type = marker.ARROW
        marker.action = 0
        marker.color.r = color_rgbs[i][0]
        marker.color.g = color_rgbs[i][1]
        marker.color.b = color_rgbs[i][2]
        marker.color.a = 1.0
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        # Need two points to draw arrow: source and destination
        marker.points.append(Point(*cur_pos))
        marker.points.append(Point(*(cur_pos + FORCE_VEC_SCALE * obj_weights[i] * vec)))
        marker_array_msg.markers.append(marker)

    pub.publish(marker_array_msg)


def publish_objects(object_pub: rospy.Publisher, pose_pubs: List[rospy.Publisher],
                    objects: List[Object], obj_colors_rgb, frame):
    """
    Publish function to view objects/nodes as spheres.

    :param object_pub: ROS publisher of object markers
    :param pose_pubs: ROS pose publishers specific to each object/node
    :param objects: List of objects/nodes
    :param obj_colors_rgb: RGB colors for each object
    :return:
    """
    marker_array_msg = MarkerArray()
    for i, (obj, color_rgb, pose_pub) in enumerate(zip(objects, obj_colors_rgb, pose_pubs)):
        marker = Marker()
        marker.header.frame_id = frame
        marker.id = i
        marker.type = marker.SPHERE
        marker.action = 0
        marker.pose.position.x = obj.pos[0]
        marker.pose.position.y = obj.pos[1]
        marker.pose.position.z = obj.pos[2]
        marker.color.r = color_rgb[0]
        marker.color.g = color_rgb[1]
        marker.color.b = color_rgb[2]

        # make other objects and goal be partially transparent
        marker.color.a = 0.5
        marker.scale.x = 2 * obj.radius
        marker.scale.y = 2 * obj.radius
        marker.scale.z = 2 * obj.radius
        marker_array_msg.markers.append(marker)

        # Visualize 3D pose axes
        pose = pose_to_msg(np.concatenate([obj.pos, obj.ori]), frame)
        pose_pub.publish(pose)

    object_pub.publish(marker_array_msg)


def path_from_traj(traj: np.ndarray, frame) -> Path:
    """
    Convert a trajectory to a nav_msgs.msg.Path() message

    :param traj: Pose trajectory (N x pos_dim + rot_dim)
    :return: nav_msgs.msg.Path() message
    """
    path = Path()
    path.header.frame_id = frame
    assert traj.shape[1] == 7  # pos[3], ori[4]
    for i in range(traj.shape[0]):
        pose = pose_to_msg(traj[i], frame)
        path.poses.append(pose)

    return path


def make_boundary_message(lb: np.ndarray, ub: np.ndarray, frame) -> MarkerArray:
    """
    Make a series of Line marker messages to draw bounds of data

    :param lb: Lower bound of data
    :param ub: Upper bound of data
    :return: MarkerArray() message containing Line markers
    """

    # draw all edges of cube
    def make_boundary_marker(p0, p1, idx):
        marker = Marker()
        marker.header.frame_id = frame
        marker.id = idx
        marker.type = marker.LINE_LIST
        # marker.action = marker.ADD
        marker.scale.x = 0.01
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points.append(Point(p0[0], p0[1], p0[2]))
        marker.points.append(Point(p1[0], p1[1], p1[2]))
        return marker

    # perform three CCW rotations about the center of the XY box
    # assumes XY lower and upper bounds are the same (box)
    cx, cy, _ = (lb + ub) / 2
    theta = np.pi / 2
    M = np.linalg.inv(np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]]) @
                      np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]]) @
                      np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]]))
    xy_vecs = [np.array([[ub[0] - lb[0], 0, 0]])]
    for i in range(3):
        xy_vecs.append(xy_vecs[-1] @ M)
    xy_vecs = np.vstack(xy_vecs)
    xy_vecs = xy_vecs[:, 0:2]

    # horizontal edges
    point_pairs = []
    p0 = lb[0:2]
    for i in range(4):
        p1 = p0 + xy_vecs[i]
        point_pairs.append([np.append(p0, lb[2]), np.append(p1, lb[2])])
        point_pairs.append([np.append(p0, ub[2]), np.append(p1, ub[2])])
        p0 = np.copy(p1)

    # vertical edges
    point_pairs += [
        [np.array([lb[0], lb[1], lb[2]]), np.array([lb[0], lb[1], ub[2]])],
        [np.array([lb[0], ub[1], lb[2]]), np.array([lb[0], ub[1], ub[2]])],
        [np.array([ub[0], lb[1], lb[2]]), np.array([ub[0], lb[1], ub[2]])],
        [np.array([ub[0], ub[1], lb[2]]), np.array([ub[0], ub[1], ub[2]])],
    ]

    bounds_msg = MarkerArray()
    bounds_msg.markers = [make_boundary_marker(p0, p1, idx)
                          for idx, (p0, p1) in enumerate(point_pairs)]
    return bounds_msg


class Viz3DROSPublisher(object):
    """
    Helper to visualize any live 3D scene in Rviz.
    """

    def __init__(self, num_objects, bounds=None, frame="map"):
        self.frame = frame
        # Optionally draw a bounding cube around scene
        self.bounds_msg = make_boundary_message(bounds[0], bounds[1], frame) if bounds is not None else None

        # Start ROS node
        try:
            rospy.init_node("view_obstacles")
        except rospy.exceptions.ROSException:
            pass
        self.ros_rate = rospy.Rate(hz=10)

        # Define ROS Publishers
        self.bounds_pub = rospy.Publisher("/bounds", MarkerArray, queue_size=5)
        self.object_pub = rospy.Publisher('object_markers', MarkerArray, queue_size=10)
        self.object_forces_pub = rospy.Publisher('other_vecs', MarkerArray, queue_size=10)
        self.pose_pubs = [rospy.Publisher('obj_%d' % i, PoseStamped, queue_size=10) for i in range(num_objects)]
        self.pose_pubs += [
            rospy.Publisher('start', PoseStamped, queue_size=10),
            rospy.Publisher('goal', PoseStamped, queue_size=10),
            rospy.Publisher('agent', PoseStamped, queue_size=10),
        ]
        self.agent_path_pub = rospy.Publisher('agent_path', Path, queue_size=10)
        self.expert_path_pub = rospy.Publisher('expert_path', Path, queue_size=10)
        self.expert_pose_pub = rospy.Publisher('expert', PoseStamped, queue_size=10)

    def publish(self, objects: List[Object], object_colors,
                agent_traj: np.ndarray = None, expert_traj: np.ndarray = None,
                object_forces: np.ndarray = None, force_colors_rgb: List[Tuple] = None):
        """
        Repeatedly call this function to update visualization.
        """

        publish_objects(object_pub=self.object_pub, pose_pubs=self.pose_pubs,
                        obj_colors_rgb=object_colors, objects=objects, frame=self.frame)

        if agent_traj is not None:
            self.agent_path_pub.publish(path_from_traj(agent_traj, self.frame))

        if expert_traj is not None:
            # NOTE: expert_traj should dynamically change to be similar
            #   length as agent traj
            self.expert_path_pub.publish(path_from_traj(expert_traj, self.frame))
            self.expert_pose_pub.publish(pose_to_msg(expert_traj[0], self.frame))

        if self.bounds_msg is not None:
            self.bounds_pub.publish(self.bounds_msg)

        if object_forces is not None:
            publish_object_forces(self.object_forces_pub, cur_pos=agent_traj[0, 0:POS_DIM],
                                  object_forces=object_forces,
                                  color_rgbs=force_colors_rgb, frame=self.frame)

        try:
            self.ros_rate.sleep()
        except rospy.ROSInterruptException:
            pass


def viz_helper(policy: Policy, start, goal, goal_radius,
               object_poses, object_radii, object_types,
               objects: List[Object], expert_traj, viz_args,
               viz_3D_publisher: Viz3DROSPublisher):
    """
    Runs policy in closed-loop and visualizes its behavior.

    :param policy: Policy to run
    :param start: Start pose (7,)
    :param goal: Goal pose (7,)
    :param goal_radius: Goal radius
    :param object_poses: Object poses (n_objects x 7)
    :param object_radii: Object radii (n_objects,)
    :param object_types: Object types (n_objects,)
    :param objects: List of objects
    :param expert_traj: Expert trajectory (N x 7)
    :param viz_args: Extra visualization arguments
    :param viz_3D_publisher: Viz3DROSPublisher object
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
    agent_radius_tensor = torch.tensor([Params.agent_radius], device=DEVICE).view(1, 1)
    start_objects = torch.cat([start_tensor, agent_radius_tensor], dim=-1).unsqueeze(1)

    # Object
    object_poses_torch = torch.from_numpy(
        pose_to_model_input(object_poses)).to(torch.float32).to(DEVICE).unsqueeze(0)
    object_radii_torch = torch.from_numpy(object_radii).to(torch.float32).to(DEVICE).view(1, -1, 1)
    objects_torch = torch.cat([object_poses_torch, object_radii_torch], dim=-1)
    object_types_tensor = torch.from_numpy(object_types).to(torch.long).to(DEVICE).unsqueeze(0)

    # all_objects filled inside the loop because agent pose updates each timestep
    # [actual objects, start, goal, agent]
    all_object_radii = object_radii.tolist() + [
        Params.agent_radius,
        goal_radius,
        Params.agent_radius,
    ]
    all_objects = objects + [
        Object(pos=start[0:POS_DIM], radius=Params.agent_radius, ori=start[POS_DIM:]),  # start
        Object(pos=goal[0:POS_DIM], radius=goal_radius, ori=goal[POS_DIM:]),  # goal
    ]
    all_object_colors = Params.obj_colors_rgb[object_types].tolist() + viz_args["non_object_colors_rgb"]
    force_colors = Params.obj_colors_rgb[object_types].tolist() + [Params.goal_color_rgb]

    cur_pose = np.copy(start)
    global_wpt_idx = 1
    step = 0
    # Closed loop visualizing model rollout and executing only first action
    while (step < viz_args["max_num_steps"] and not rospy.is_shutdown() and
           np.linalg.norm(cur_pose[0:POS_DIM] - goal[0:POS_DIM]) > viz_args["dist_tol"]):
        cur_pose_tensor = torch.from_numpy(pose_to_model_input(cur_pose[np.newaxis])).to(torch.float32).to(DEVICE)
        object_forces_rollout = []
        rollout_traj = []
        for k in range(viz_args["rollout_horizon"]):
            # Define "object" inputs into policy
            # Current
            current = torch.cat([cur_pose_tensor, agent_radius_tensor], dim=-1).unsqueeze(1)
            # Goal
            goal_radii = torch.norm(goal_tensor[:, 0:POS_DIM] - cur_pose_tensor[:, 0:POS_DIM], dim=-1).unsqueeze(0)
            goal_rot_objects = torch.cat([goal_tensor, goal_rot_radii], dim=-1).unsqueeze(1)
            goal_objects = torch.cat([goal_tensor, goal_radii], dim=-1).unsqueeze(1)
            # Start
            start_rot_radii = torch.norm(start_tensor[:, 0:POS_DIM] - cur_pose_tensor[:, 0:POS_DIM],
                                         dim=-1).unsqueeze(0)
            start_rot_objects = torch.cat([start_tensor, start_rot_radii], dim=-1).unsqueeze(1)

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
            pred_pos = pred_pose[0:1, :POS_DIM]
            pred_ori = decode_ori(pred_pose[0:1, POS_DIM:])
            rollout_traj.append(np.hstack([pred_pos, pred_ori]))

        rollout_traj = np.vstack(rollout_traj)
        cur_pose = rollout_traj[0]
        if len(object_forces_rollout) > 0:
            object_forces = object_forces_rollout[0][0].detach().cpu().numpy()
        else:
            object_forces = None

        # Inlucde current agent as an "object" and visualize objects as spheres
        # [actual objects, start, goal, agent]
        all_objects_per_step = all_objects + [
            Object(pos=cur_pose[0:POS_DIM], radius=Params.agent_radius, ori=cur_pose[POS_DIM:])
        ]

        viz_3D_publisher.publish(objects=all_objects_per_step, agent_traj=rollout_traj,
                                 expert_traj=expert_traj[global_wpt_idx:], object_colors=all_object_colors,
                                 object_forces=object_forces, force_colors_rgb=force_colors)

        global_wpt_idx = np.argmin(np.linalg.norm(cur_pose[:POS_DIM] - expert_traj[:, :POS_DIM], axis=1)) + 1
        global_wpt_idx = min(global_wpt_idx, expert_traj.shape[0] - 1)
        step += 1


def viz_main(policy, test_data_root, calc_pos, calc_rot):
    viz_args = dict(
        rollout_horizon=20,
        max_num_steps=150,
        lb=Params.lb_2D,
        ub=Params.ub_2D,
        dist_tol=0.3,
        calc_pos=calc_pos,
        calc_rot=calc_rot,
        non_object_colors_rgb=[
            Params.start_color_rgb,
            Params.goal_color_rgb,
            Params.agent_color_rgb,
        ]
    )

    # NOTE: Extra modification to pose_pub needed to handle variable number of objects
    data = np.load(os.path.join(test_data_root, "traj_%d.npz" % 0), allow_pickle=True)
    num_objects = len(data["object_types"])
    viz_3D_publisher = Viz3DROSPublisher(num_objects=num_objects, bounds=(Params.lb_3D, Params.ub_3D))

    N = 10
    for i in range(N):
        # Load data sample
        rand_file_idx = np.random.choice(100)
        print("file idx: %d" % rand_file_idx)
        data = np.load(os.path.join(test_data_root, "traj_%d.npz" % rand_file_idx), allow_pickle=True)
        expert_traj = data["states"]
        goal_radius = data["goal_radius"].item()
        object_poses = np.copy(data["object_poses"])
        object_radii = np.copy(data["object_radii"])
        object_types = np.copy(data["object_types"])
        objects = [
            Object(pos=object_poses[i][0:POS_DIM], radius=object_radii[i],
                   # NOTE: R * Offset_R, order matters to apply object-relative offset!!
                   ori=(R.from_quat(object_poses[i][POS_DIM:]) *
                        R.from_quat(Params.ori_offsets_3D[obj_type])).as_quat()
                   )
            for i, obj_type in enumerate(object_types)
        ]

        viz_helper(policy, start=expert_traj[0], goal=expert_traj[-1],
                   goal_radius=goal_radius, object_poses=object_poses,
                   object_radii=object_radii, object_types=object_types,
                   objects=objects, expert_traj=expert_traj, viz_args=viz_args,
                   viz_3D_publisher=viz_3D_publisher)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store', type=str, help="trained model name")
    parser.add_argument('--data_name', action='store', type=str, help="name for data folder")
    parser.add_argument('--loaded_epoch', action='store', type=int, default=-1)
    parser.add_argument('--calc_pos', action='store_true', help="calculate position")
    parser.add_argument('--calc_rot', action='store_true', help="calculate rotation")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    assert args.model_name is not None
    assert args.data_name is not None

    # Load trained model arguments
    with open(os.path.join(Params.model_root, args.model_name, "train_args_pt_1.json"), "r") as f:
        train_args = json.load(f)
    test_data_root = f"{Params.data_root}/{args.data_name}_test/"

    # load model
    assert train_args["is_3D"], "viz_3D only for 3D models"
    network = PolicyNetwork(n_objects=Params.n_train_objects, pos_dim=POS_DIM, rot_dim=ROT_DIM,
                            pos_preference_dim=train_args['pos_preference_dim'],
                            rot_preference_dim=train_args['rot_preference_dim'],
                            hidden_dim=train_args['hidden_dim'],
                            device=DEVICE).to(DEVICE)
    network.load_state_dict(
        torch.load(os.path.join(Params.model_root, args.model_name, "model_%d.h5" % args.loaded_epoch)))
    policy = Policy(network)

    # TODO: Uncomment to tune Policy.pos_ignore_feat and Policy.rot_ignore_feat
    # network.pos_pref_feat_train[Params.ATTRACT_IDX].data.copy_(policy.pos_ignore_feat)
    # network.pos_pref_feat_train[Params.REPEL_IDX].data.copy_(policy.pos_ignore_feat)
    # network.rot_pref_feat_train[Params.CARE_ROT_IDX].data.copy_(policy.rot_ignore_feat)
    # network.rot_pref_feat_train[Params.IGNORE_ROT_IDX].data.copy_(policy.rot_ignore_feat)

    # TODO: Uncomment to compare learned vs actual rotational offsets in 3D
    # learned_offsets = network.rot_offsets_train / torch.norm(network.rot_offsets_train, dim=1, keepdim=True)
    # print(learned_offsets.detach().cpu().numpy())  # ignored rot offset is arbitrary
    # print(Params.ori_offsets_3D)
    # ipdb.set_trace()

    viz_main(policy, test_data_root, calc_pos=args.calc_pos, calc_rot=args.calc_rot)
