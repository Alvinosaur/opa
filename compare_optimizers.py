import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

root = "eval_adaptation_results"
pos_folders = ["Adam_pos_fixed_init_detached_steps",
               "learn2learn_group_pos_fixed_init_detached_steps",
               "RLS(alpha_0.5_lmbda_0.9)_pos_fixed_init_detached_steps"
               ]

rot_folders = ["Adam_rot_fixed_init_detached_steps",
               "learn2learn_group_rot_fixed_init_detached_steps",
               "RLS(alpha_0.5_lmbda_0.9)_rot_fixed_init_detached_steps"
               ]
class_colors = np.array(["blue", "green", "red"])
class_labels = np.array(["Adam", "LSTM", "RLS"])
# Load loss data from the optimizers
all_pos_loss = []
for folder in pos_folders:
    loss = np.load(os.path.join(root, folder, "loss.npy"))
    final_loss = loss[:, -1]
    all_pos_loss.append(final_loss[np.newaxis])

all_pos_loss = np.concatenate(all_pos_loss, axis=0)
lowest_loss_label = np.argmin(all_pos_loss, axis=0)

# TODO: figure out what non-uniform FFT libraries do, make sense of their output fourier transform
# Load Properties to describe each data sample
# Frequency and amplitude of the trajectories
all_pos_freq = []
all_pos_amp = []
auc_per_sample = np.load(os.path.join(root, "auc_per_sample_pos_2D_test.npy"))
max_mag_per_sample = np.load(os.path.join(
    root, "max_mag_per_sample_pos_2D_test.npy"))
variance_per_sample = np.load(os.path.join(
    root, "variance_per_sample_pos_2D_test.npy"))
inputs = [
    [auc, max_mag, var] for auc, max_mag, var in zip(
        auc_per_sample, max_mag_per_sample, variance_per_sample)
]
inputs = np.array(inputs)

# Plot all samples as scatter points with each class labeled with a different color
fig, axes = plt.subplots(3, 1, figsize=(5, 10))  #
axes[0].scatter(inputs[:, 0], inputs[:, 1], c=class_colors[lowest_loss_label])
axes[0].set_title("Max Magnitude vs AUC")  # Y vs X
axes[1].scatter(inputs[:, 0], inputs[:, 2], c=class_colors[lowest_loss_label])
axes[1].set_title("Variance vs AUC")
scatter = axes[2].scatter(inputs[:, 1], inputs[:, 2],
                          c=class_colors[lowest_loss_label])
axes[2].set_title("Variance vs Max Magnitude")
# Manual legend
handles = [mpatches.Patch(color=color, label=label) for color, label in
           zip(class_colors, class_labels)]
plt.legend(handles=handles)
plt.savefig("pos_best_optim_scatter.png", dpi=300)
# plt.show()


# Classification Attempts
# SVM
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(inputs[:, :-1], lowest_loss_label)


# Attempt 2: learn simple MLP with attention over the different input features
