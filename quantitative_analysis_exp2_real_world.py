import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
plt.set_loglevel("info")

from model import Policy, PolicyNetwork, pose_to_model_input, decode_ori
from train import random_seed_adaptation, process_single_full_traj, DEVICE
from data_params import Params
from exp_params import *
from exp_utils import *
from elastic_band import Object
from viz_3D import Viz3DROSPublisher, pose_to_msg, msg_to_pose
from exp1_cup_low_table_sim import robot_table_surface_projections
from data_generation import rand_quat

from exp2_real_world import obstacles_pos_world, obstacles_radii, Net2World, inspection_pos_world, num_objects


saved_trial_folder = "hardware_demo_videos/trial8"
obstacle_pos = obstacles_pos_world[0]


def calc_min_dist_fn(pose_traj): return np.min(np.linalg.norm(
    pose_traj[:, 0:3] - obstacle_pos[np.newaxis], axis=-1))


# measure min distance from obstacle before and after position correction
pose_traj_before = np.load(os.path.join(
    saved_trial_folder, "ee_pose_traj_iter_0.npy"))
min_dist_before = calc_min_dist_fn(pose_traj_before)

pose_traj_during = np.load(os.path.join(
    saved_trial_folder, "ee_pose_traj_iter_1.npy"))
is_intervene_traj = np.load(os.path.join(
    saved_trial_folder, "is_intervene_traj1.npy"))

pose_traj_after = np.load(os.path.join(
    saved_trial_folder, "ee_pose_traj_iter_2.npy"))
min_dist_after = calc_min_dist_fn(pose_traj_after)

# Plot 2D top-down view of trajectories 1) before 2) during intervention 3) updated
plt.plot(pose_traj_before[:, 0], pose_traj_before[:,
         1], label="Trial 1", linewidth=3.0)
plt.scatter([pose_traj_before[0, 0], ], [pose_traj_before[0, 1], ])

for i in range(len(pose_traj_during) - 1):
    color = "red" if is_intervene_traj[i] else "green"
    label = "Trial 2 (Human)" if is_intervene_traj[i] else "Trial 2 (Robot)"
    plt.plot(pose_traj_during[i:i + 2, 0],
             pose_traj_during[i:i + 2, 1], label=label,
             color=color, linewidth=3.0)
plt.scatter([pose_traj_during[0, 0], ], [
            pose_traj_during[0, 1], ], color="green")

plt.plot(pose_traj_after[:, 0], pose_traj_after[:, 1],
         label="Trial 3", linewidth=3.0)
plt.scatter([pose_traj_after[0, 0], ], [pose_traj_after[0, 1], ])

plt.gca().add_patch(plt.Circle(
    obstacle_pos[0:2], obstacles_radii[0] * Net2World, alpha=0.4, label="Obstacle"))

print("min_dist_before: %.3f" % min_dist_before)
print("min_dist_after: %.3f" % min_dist_after)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.gca().set_aspect('equal')
plt.xticks([-1, -0.5, 0.0, 0.5, 1.0])
plt.yticks([-1, -0.5, 0.0, 0.5, 1.0])
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
# plt.show()

# measure error between demonstrated orientation and achieved orientation

# First get the demonstrated orientations, the final orientation of each perturbation traj
desired_box_ori = np.load(os.path.join(
    saved_trial_folder, "perturb_traj_iter_1_num_1.npy"))[-1, 3:]
desired_can_ori = np.load(os.path.join(
    saved_trial_folder, "perturb_traj_iter_4_num_0.npy"))[-1, 3:]

# Next get the orientation trajectories of physical robot and with simulated, perfect policy
robot_box_traj = np.load(os.path.join(
    saved_trial_folder, "ee_pose_traj_iter_2.npy"))
robot_can_traj = np.load(os.path.join(
    saved_trial_folder, "ee_pose_traj_iter_5.npy"))

sim_box_traj = np.load(os.path.join(
    saved_trial_folder, "ee_pose_traj_iter_2_sim.npy"))
sim_can_traj = np.load(os.path.join(
    saved_trial_folder, "ee_pose_traj_iter_5_sim.npy"))

# Calculate orientation errors
# calc_ori_error_fn = lambda q1, q2: np.arccos(np.abs(q1 @ q2))

# Returns angle of axis-angle represntation, overall giving angle between two
# orientations in radians
def calc_ori_error_fn(q1, q2, deg=False): return np.linalg.norm((
    R.from_quat(q1) * R.from_quat(q2).inv()).as_rotvec()) * (deg * 180 / np.pi + (1 - deg) * 1)

##################################################
# def get_nearest_orientation(pose_traj):
#     dists = np.linalg.norm(
#         pose_traj[:, 0:3] - inspection_pos_world[np.newaxis], axis=-1)
#     nearest_idx = np.argmin(dists)
#     return pose_traj[nearest_idx, 3:]

# nearest_robot_box_ori = get_nearest_orientation(robot_box_traj)
# nearest_robot_can_ori = get_nearest_orientation(robot_can_traj)

# nearest_sim_box_ori = get_nearest_orientation(sim_box_traj)
# nearest_sim_can_ori = get_nearest_orientation(sim_can_traj)

# robot_box_ori_error = calc_ori_error_fn(
#     nearest_robot_box_ori, desired_box_ori, deg=True)
# robot_can_ori_error = calc_ori_error_fn(
#     nearest_robot_can_ori, desired_can_ori, deg=True)

# sim_box_ori_error = calc_ori_error_fn(
#     nearest_sim_box_ori, desired_box_ori, deg=True)
# sim_can_ori_error = calc_ori_error_fn(
#     nearest_sim_can_ori, desired_can_ori, deg=True)
##################################################

##################################################
robot_box_ori_error = min([
    calc_ori_error_fn(robot_box_traj[i][3:], desired_box_ori, deg=True)
    for i in range(len(robot_box_traj))])
robot_can_ori_error = min([
    calc_ori_error_fn(robot_can_traj[i][3:], desired_can_ori, deg=True)
    for i in range(len(robot_can_traj))])

sim_box_ori_error = min([
    calc_ori_error_fn(sim_box_traj[i][3:], desired_box_ori, deg=True)
    for i in range(len(sim_box_traj))])
sim_can_ori_error = min([
    calc_ori_error_fn(sim_can_traj[i][3:], desired_can_ori, deg=True)
    for i in range(len(sim_can_traj))])
##################################################

print("robot vs sim:")
print(robot_box_ori_error, sim_box_ori_error)
print(robot_can_ori_error, sim_can_ori_error)

# TODO: Show rviz video to really judge what these distances mean
