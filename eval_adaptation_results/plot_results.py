import os
import ipdb
import numpy as np
import matplotlib.pyplot as plt
import re


model = "RLS"

if model == "LSTM":
    model_name = "learn2learn_group_diff_init"
    model_title = "Learn2Learn_diff_init"
elif model == "RLS":
    alpha = 0.5
    lmbda = 0.9
    model_name = "RLS(alpha_%.1f_lmbda_%.1f)" % (alpha, lmbda)
    model_title = model_name
elif model == "Adam":
    model_name = model_title = model
elif model == "SGD":
    lr = 0.1
    momentum = 0.9
    model_name = "SGD(lr_%.1f_momentum_%.1f)" % (lr, momentum)
    model_title = model_name
else:
    raise NotImplementedError


num_steps = 32  # [0, ..., 31]
# num_steps = 22
# num_samples = 10  # only show first 10 samples for clarity

# data_name = "pos"
# data_name = "rot"
data_name = "rot_ignore"

# data_type = "pos attract"
# data_type = "pos repel"
data_type = "rot pref"
# data_type = "rot offset"

root_folder = "/home/alvin/research/intelligent_control_lab/human_robot_interaction/opa"
# root_folder = "/home/ashek/research/hri/opa"

folder = "%s/eval_adaptation_results/%s_%s" % (
    root_folder, model_name, data_name)
fname = os.path.join(folder, "qualitative_output.txt")
with open(fname, "r") as f:
    text = f.read()

pattern = r"iter (\d+) loss: (\d+\.\d+)"
matches = re.findall(pattern, text)
iter_losses = np.array([(int(i), float(loss)) for i, loss in matches])
max_iters = int(iter_losses[:, 0].max())

# matches = [(i, v) for i, v in matches if v > 0.02]
sum_losses = np.zeros(max_iters + 1)
counts = np.zeros(max_iters + 1)
for i, loss in iter_losses:
    sum_losses[int(i)] += loss
    counts[int(i)] += 1

avg_losses = sum_losses / counts

plt.plot(avg_losses, color="black")
plt.title(model_title + " Loss vs Iter")
plt.xlabel("Update Iteration")
plt.ylabel("Loss")
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(4, 4, forward=True)
plt.savefig(os.path.join(
    folder, "%s_loss_vs_iter.png" % (model_title)), dpi=100)
# plt.show()
plt.clf()

###############################################################################
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(10, 5),
                               gridspec_kw={'height_ratios': [3, 1]})  # type: ignore
pattern = r"\-Original Grad: ([-+]?\d+.\d+), \-lr \* Pred Grad:\s+([-+]?\d+.\d+), New P: ([-+]?\d+.\d+)"
matches = re.findall(pattern, text)
orig_grads = [float(i) for i, _, _ in matches]
pred_grads = [float(i) for _, i, _ in matches]

if data_name in ["rot", "rot_ignore"]:
    orig_grad_pref = np.array(orig_grads[0::2])
    pred_grad_pref = np.array(pred_grads[0::2])
    if data_type == "rot offset":
        orig_grad_offset = np.array(orig_grads[1::2])
        pred_grad_offset = np.array(pred_grads[1::2])
else:
    if data_type == "pos repel":
        orig_grad_pref = np.array(orig_grads[0::2])
        pred_grad_pref = np.array(pred_grads[0::2])
    else:
        orig_grad_pref = np.array(orig_grads[1::2])
        pred_grad_pref = np.array(pred_grads[1::2])

# ipdb.set_trace()

pattern = r"Target params: \[(.*)\]"
target_param_strs = re.findall(pattern, text)
target_params = np.vstack([np.fromstring(s, dtype=float, sep=", ")
                           for s in target_param_strs])

pattern = r"Actual params: \[(.*)\]"
actual_param_strs = re.findall(pattern, text)
actual_params = np.vstack([np.fromstring(s, dtype=float, sep=", ")
                           for s in actual_param_strs])

# look at a specific example
matches = matches[:]

vlines = np.arange(0, len(actual_params) + 1, num_steps)
grad_i = 0
for sample_i in range(target_params.shape[0]):
    start_step, end_step = vlines[sample_i], vlines[sample_i + 1]
    ts = np.arange(start_step, end_step)

    if data_name in ["rot", "rot_ignore"]:
        if data_type == "rot pref":
            target_pref = target_params[sample_i, 0]
            ax0.hlines(y=target_pref, xmin=start_step, xmax=end_step,
                       linestyles="dashed", label="Target Pref", color="tab:green")
        else:
            try:
                target_offset = target_params[sample_i, 1]
                ax0.hlines(y=target_offset, xmin=start_step, xmax=end_step,
                           linestyles="dashed", label="Target Offset", color="tab:red")
            except:
                # This only happens when plotting rot offset of samples that involve the Rotation Ignored object which has no target offset
                pass
    else:
        if data_type == "pos repel":
            target_pref = target_params[sample_i, 0]
            ax0.hlines(y=target_pref, xmin=start_step, xmax=end_step,
                       linestyles="dashed", label="Target Pref", color="tab:green")
        else:
            target_pref = target_params[sample_i, 1]
            ax0.hlines(y=target_pref, xmin=start_step, xmax=end_step,
                       linestyles="dashed", label="Target Pref", color="tab:green")

    cur_params = actual_params[start_step:end_step]
    if data_name in ["rot", "rot_ignore"]:
        if data_type != "rot offset":
            actual_pref = cur_params[:, 0]
            ax0.plot(ts, actual_pref, label="Actual Pref", color="tab:green")
        else:
            actual_offset = cur_params[:, 1]
            ax0.plot(ts, actual_offset, label="Actual Offset", color="tab:red")
    else:
        if data_type == "pos repel":
            actual_pref = cur_params[:, 0]
            ax0.plot(ts, actual_pref, label="Actual Pref", color="tab:green")
        else:
            actual_pref = cur_params[:, 1]
            ax0.plot(ts, actual_pref, label="Actual Pref", color="tab:green")

    if data_type != "rot offset":
        # no grad for last timestep/param
        ax0.plot(ts[:-1], orig_grad_pref[grad_i:grad_i + num_steps - 1],
                 label="-1 * Orig Grad Pref", color="tab:blue", linestyle="dashed")
        ax0.plot(ts[:-1], pred_grad_pref[grad_i:grad_i + num_steps - 1],
                 label="-1 * Pred Grad Pref", color="tab:blue")
    else:
        ax0.plot(ts[:-1], orig_grad_offset[grad_i:grad_i + num_steps - 1],
                 label="-1 * Orig Grad Rot Offset", color="tab:orange", linestyle="dashed")
        ax0.plot(ts[:-1], pred_grad_offset[grad_i:grad_i + num_steps - 1],
                 label="-1 * Pred Grad Rot Offset", color="tab:orange")

    ax1.plot(ts[:-1], iter_losses[grad_i:grad_i +
             num_steps - 1, 1], color="black")

    grad_i += num_steps - 1

for x in vlines:
    ax0.axvline(x=x, linestyle="dashed", color="black", alpha=0.3)

# Show legend labels without duplicates
handles, labels = ax0.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax0.legend(by_label.values(), by_label.keys(), loc="lower left")

ax1.set_ylabel("Loss")
fig.suptitle(model_title + " " + data_type)
ax0.set_ylim([-2, 2])
plt.tight_layout()
plt.savefig(os.path.join(folder, "%s_params_and_grad.png" %
            data_type), dpi=100)
# plt.show()
