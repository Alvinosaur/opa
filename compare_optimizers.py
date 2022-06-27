import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from freq_analysis import fftPlot
from data_params import Params
from viz_2D import draw
from elastic_band import Object

root = "eval_adaptation_results"
pos_folders = ["Adam_pos_fixed_init_detached_steps",
               "learn2learn_group_pos_fixed_init_detached_steps",
               "RLS(alpha_0.5_lmbda_0.9)_pos_fixed_init_detached_steps"
               ]

rot_folders = ["Adam_rot_fixed_init_detached_steps",
               "learn2learn_group_rot_fixed_init_detached_steps",
               "RLS(alpha_0.5_lmbda_0.9)_rot_fixed_init_detached_steps"
               ]

all_pos_loss = []
for folder in pos_folders:
    loss = np.load(os.path.join(root, folder, "loss.npy"))
    all_pos_loss.append(loss[np.newaxis])

all_pos_loss = np.concatenate(all_pos_loss, axis=0)
lowest_loss_label = np.argmin(all_pos_loss, axis=0)

fig, ax = plt.subplots()
# Frequency Analysis
case = 0
train_rot = False
samples_files = [
    "sampled_file_idxs_pos.npy",
    "sampled_file_idxs_rot.npy",
    "sampled_file_idxs_rot_ignore.npy",
]
datasets = ["pos_2D_test", "rot_2D_test", "rot_2D_test"]
sampled_file_idxs = np.load(f"eval_adaptation_results/{samples_files[case]}")
for sample_i, file_idx in enumerate(sampled_file_idxs):
    if sample_i <= 0:
        continue
    data = np.load(os.path.join("data", datasets[case], "traj_%d.npz" %
                                file_idx), allow_pickle=True)
    object_types = data["object_types"]
    print("is_rot", data["is_rot"].item())
    print("file %d idx: %d" % (sample_i, file_idx))
    expert_traj = data["states"]

    ######### Debug Plot #########
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
    objects = [
        Object(pos=object_poses[i][0:2], radius=object_radii[i],
               ori=object_poses[i][-1] + theta_offsets[i]) for i in range(len(object_types))
    ]
    # draw(ax, start_pose=start, goal_pose=goal,
    #      goal_radius=goal_radius, agent_radius=Params.agent_radius,
    #      object_types=object_types, offset_objects=objects,
    #      pred_traj=expert_traj, expert_traj=expert_traj,
    #      show_rot=train_rot, hold=True)

    ######## FFT Analysis #########
    expert_traj_xy = expert_traj[:, :2]

    # Use PCA to extract principal axis, align trajectory to principal axis
    pca = PCA(n_components=1)
    pca.fit(expert_traj_xy)
    major_axis = pca.components_[0]
    midpoint = np.mean(expert_traj_xy, axis=0)
    ang = np.arctan2(major_axis[1], major_axis[0])
    R = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang), np.cos(ang)]])

    major_axis_start = midpoint + major_axis * \
        np.linalg.norm(expert_traj_xy[0] - midpoint)
    major_axis_end = midpoint - major_axis * \
        np.linalg.norm(expert_traj_xy[-1] - midpoint)
    draw(ax, start_pose=start, goal_pose=goal,
         goal_radius=goal_radius, agent_radius=Params.agent_radius,
         object_types=object_types, offset_objects=objects,
         pred_traj=np.vstack([major_axis_start[np.newaxis],
                              major_axis_end[np.newaxis],
                              ]),
         expert_traj=expert_traj,
         show_rot=train_rot, hold=True)
    # normalize to be zero-centered and undo rotation of major axis
    traj_aligned = np.dot(expert_traj_xy - midpoint, R)

    plt.clf()
    plt.plot(traj_aligned[:, 0], traj_aligned[:, 1], "k-")
    plt.show()
    plt.tight_layout()
    xmag, xfreq = fftPlot(traj_aligned[:, 1], dt=1, title="Overall FFT")
    fig, ax = plt.subplots()
    # xmag, xfreq = fftPlot(expert_traj_xy[:, 0], dt=1, title="X FFT")
    # ymag, yfreq = fftPlot(expert_traj_xy[:, 1], dt=1, title="Y FFT")
