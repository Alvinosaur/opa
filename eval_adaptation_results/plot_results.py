import os
import ipdb
import numpy as np
import matplotlib.pyplot as plt
import re


folder = "/home/ashek/research/hri/opa/eval_adaptation_results/RLS(alpha_0.5_lmbda_0.4)_pos"
fname = os.path.join(folder, "qualitative_output.txt")
# applied_lr = 1e-1  # LSTM lr
applied_lr = 0.5  # RLS Alpha
# applied_lr = 1  # Adam, NOTE: we directly measure new_p - old_p, lr already applied
num_steps = 31  # [0, ..., 30]
# num_steps = 11
# target_value = -1.07  # ROT IGNORE
target_value = 1.33  # ROT CARE
# target_value = np.pi/2  # ROT OFFSET
# target_value = 1.18  # POS REPEL
# target_value = 0.27  # POS ATTRACT
with open(fname, "r") as f:
    text = f.read()

pattern = r"iter (\d+) loss: (\d+\.\d+)"
matches = re.findall(pattern, text)
matches = [(int(i), float(loss)) for i, loss in matches]
max_iters = max(matches, key=lambda x: x[0])[0]

# matches = [(i, v) for i, v in matches if v > 0.02]
losses = np.zeros(max_iters + 1)
for i, loss in matches:
    losses[i] = loss

plt.plot(losses)
plt.title("Test Time Adaptation Loss vs Update Iter for Pos")
plt.xlabel("Update Iteration")
plt.ylabel("Loss")
plt.savefig(os.path.join(
    folder, "adaptation_loss_vs_update_iter.png"))

pattern = r"Original Grad: ([-+]?\d+.\d+), LSTM Grad: ([-+]?\d+.\d+), New P: ([-+]?\d+.\d+)"
matches = re.findall(pattern, text)
matches = [(float(i), float(j), float(k)) for i, j, k in matches]

# only look at every other output to get the 2nd param
start = 0  # 1: 1, 3, 5, ... or 0: 0, 2, 4, ...
matches = matches[start::2]

# look at a specific example
matches = matches[:]

vlines = np.arange(0, len(matches)+1, num_steps)

plt.plot([-v[0] for v in matches], label="-1 * Original Grad")
plt.plot([-applied_lr * v[1] for v in matches], label="-lr * Pred Grad")
plt.plot([v[2] for v in matches], label="New P")
plt.hlines(y=target_value, xmin=0, xmax=len(matches),
           linestyles="dashed", label="Target P")
for x in vlines:
    plt.axvline(x=x, linestyle="dashed", color="black", alpha=0.5)
plt.legend()
plt.title("Rot Pref")
plt.show()
