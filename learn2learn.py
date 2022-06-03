import os
import numpy as np
import copy
import argparse
import json
from pytest import param
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset import Dataset
from data_params import Params
from model import PolicyNetwork, Policy
from train import load_batch, cuda, DEVICE
from online_adaptation import perform_adaptation_learn2learn


def detach_var(v):
    var = Variable(v.data, requires_grad=True)
    var.retain_grad()
    return var


class GradIsNoneError(Exception):
    pass


class LearnedOptimizer(nn.Module):
    """
    Unlike the original Learn2Learn that uses a shared learned optimizer
    for all model weights, we train a specific learned optimizer for each
    of the specific types of adaptable parameters of OPA. These include
    learned position and rotation preferences as well as rotation offsets.
    """

    def __init__(self, device, hidden_dim=20, param_dim=1,
                 max_steps=5, opt_lr=1e-3, tgt_lr=1e-1, training=False):
        """
        :param n_params: number of parameters to optimize
        :param device: device to use
        :param preproc: whether to use preprocessing
        :param hidden_dim: size of hidden state
        :param preproc_factor: factor to use for preprocessing
        :param max_steps: maximum number of steps to optimize for
        :param opt_lr: learning rate for meta-optimizer
        :param lr: learning rate for the target params to optimize
        """
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.tgt_lr = tgt_lr
        self.training = training
        self.param_dim = param_dim
        self.recurs = nn.LSTMCell(param_dim, hidden_dim)
        self.recurs2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, param_dim)

        # filled in on the first call to step()
        self.param_size = None
        self.hidden_states = self.hidden_states_temp = None
        self.cell_states = self.cell_states_temp = None

        # Optimization Info
        self.max_steps = max_steps
        self.loss_to_optimize = 0.0
        self.num_steps = 0.0

        # "Meta Optimizer" to optimize this learned optimizer's own weights
        self.meta_opt = torch.optim.Adam(self.parameters(), lr=opt_lr)

    def forward(self, inp, p_indices):
        assert self.hidden_states is not None, "Must call step() first!"
        hidden0, cell0 = self.recurs(
            inp, (self.hidden_states[0][p_indices], self.cell_states[0][p_indices]))
        hidden1, cell1 = self.recurs2(
            hidden0, (self.hidden_states[1][p_indices], self.cell_states[1][p_indices]))

        self.hidden_states_temp[0][p_indices] = hidden0
        self.cell_states_temp[0][p_indices] = cell0
        self.hidden_states_temp[1][p_indices] = hidden1
        self.cell_states_temp[1][p_indices] = cell1

        return self.output(hidden1)

    def reset_lstm(self):
        self.param_size = None  # Will automatically reset hidden states on the next call

    def step(self, new_loss, params, verbose=False, retain_graph_override=False):
        if len(params) == 0:
            return [], False

        if self.param_size is None:
            # Initialize hidden and cell states
            self.param_size = sum(p.numel() for p in params)
            zero_init = [
                Variable(torch.zeros(self.param_size, self.hidden_dim, device=self.device)) for _ in range(2)]
            self.hidden_states = copy.deepcopy(zero_init)
            self.hidden_states_temp = copy.deepcopy(zero_init)
            self.cell_states = copy.deepcopy(zero_init)
            self.cell_states_temp = copy.deepcopy(zero_init)

        new_loss.backward(retain_graph=self.training or retain_graph_override)
        self.loss_to_optimize = self.loss_to_optimize + new_loss
        self.num_steps += 1

        offset = 0
        new_params = []
        for pi, p in enumerate(params):
            cur_sz = p.numel()
            # p_indices: [offset:offset + cur_sz]
            if self.param_dim == 1:
                p_indices = torch.arange(
                    offset, offset + cur_sz, device=DEVICE)
            else:
                p_indices = torch.arange(pi, pi+1, device=DEVICE)

            # Get original gradients and feed as input to learned optimizer
            try:
                if p.grad is None:
                    raise GradIsNoneError()
                gradients = detach_var(p.grad)
                if self.param_dim == 1:
                    gradients = gradients.view(cur_sz, 1)  # separate batches
                else:
                    gradients = gradients.view(1, cur_sz)

                updates = self.forward(gradients, p_indices)
            except GradIsNoneError:
                # No gradients??
                # This is possible when an object type/idx doesn't appear in a scene, so it isn't involved in computation... should only occur during training of LearnedOptimizer
                gradients = updates = torch.zeros(cur_sz, 1, device=DEVICE)

            # Apply learned update
            new_p = p - self.tgt_lr * updates.view(*p.size())

            # Ensure computation graph is preserved
            new_p.retain_grad()
            new_params.append(new_p)

            offset += cur_sz
            if verbose:
                print("Original Grad: %.3f, LSTM Grad: %.3f, New P: %.3f" % (
                    -gradients.item(), -updates.item(), new_p.item()))

        # prevent computation graph from being too large
        if self.num_steps >= self.max_steps:
            need_detach = True
            self.num_steps = 0
            if self.training:
                self.meta_opt.zero_grad()
                self.loss_to_optimize.backward()
                self.meta_opt.step()

            # reset computation graph
            self.loss_to_optimize = 0.0
            torch.cuda.empty_cache()

            self.hidden_states = [detach_var(v)
                                  for v in self.hidden_states_temp]
            self.cell_states = [detach_var(v) for v in self.cell_states_temp]

        else:
            need_detach = False
            self.hidden_states = self.hidden_states_temp
            self.cell_states = self.cell_states_temp

        return new_params, need_detach

    @staticmethod
    def load_model(folder, device, epoch=3):
        args_path = os.path.join(folder, "train_args.json")
        with open(args_path, "r") as f:
            learn2learn_args = json.load(f)
        learned_opt = LearnedOptimizer(device=device, max_steps=1,
                                       tgt_lr=learn2learn_args['tgt_lr'],
                                       opt_lr=learn2learn_args['opt_lr'],
                                       hidden_dim=learn2learn_args['hidden_dim'],)
        learned_opt.load_state_dict(
            torch.load(os.path.join(folder, f"learned_opt_{epoch}.h5")))
        learned_opt.to(device)
        learned_opt.eval()

        return learned_opt


class LearnedOptimizerGroup(object):
    def __init__(self, pos_opt_path, rot_opt_path, rot_offset_opt_path, device):
        self.pos_opt = LearnedOptimizer.load_model(pos_opt_path, device=device)
        self.rot_opt = LearnedOptimizer.load_model(rot_opt_path, device=device)
        self.rot_offset_opt = LearnedOptimizer.load_model(
            rot_offset_opt_path, device=device)

    def reset_lstm(self):
        self.pos_opt.reset_lstm()
        self.rot_opt.reset_lstm()
        self.rot_offset_opt.reset_lstm()

    def step(self, new_loss, params, param_types, verbose=False):
        # separate params into pos, rot, and rot_offset
        pos_params, rot_params, rot_offset_params = [], [], []
        pos_param_idxs, rot_param_idxs, rot_offset_param_idxs = [], [], []
        for pi, (p, t) in enumerate(zip(params, param_types)):
            if t == Policy.POS_FEAT:
                pos_params.append(p)
                pos_param_idxs.append(pi)
            elif t == Policy.ROT_FEAT:
                rot_params.append(p)
                rot_param_idxs.append(pi)
            elif t == Policy.ROT_OFFSET:
                rot_offset_params.append(p)
                rot_offset_param_idxs.append(pi)
            else:
                raise ValueError("Unknown param type")

        pos_params, need_detach_pos = self.pos_opt.step(
            new_loss, pos_params, verbose=verbose, retain_graph_override=True)
        rot_params, need_detach_rot = self.rot_opt.step(
            new_loss, rot_params, verbose=verbose, retain_graph_override=True)
        rot_offset_params, need_detach_rot_offset = self.rot_offset_opt.step(
            new_loss, rot_offset_params, verbose=verbose, retain_graph_override=False)

        # combine params back into list
        new_params = [None] * len(params)
        for pi, p in zip(pos_param_idxs, pos_params):
            new_params[pi] = p
        for pi, p in zip(rot_param_idxs, rot_params):
            new_params[pi] = p
        for pi, p in zip(rot_offset_param_idxs, rot_offset_params):
            new_params[pi] = p

        return new_params, (need_detach_pos or need_detach_rot or need_detach_rot_offset)


def train_helper(policy: Policy, learned_opt: LearnedOptimizer, batch_data, train_pos, train_rot, adapt_kwargs):
    (traj, goal_radius, obj_poses, obj_radii, obj_types) = batch_data[0]
    # num_objects = len(obj_types)  # fails when rot data can contain "care" or "ignore" objects, but only one object in a given scene
    num_objects = 2  # (Attract, Repel pos) or (Care, Ignore rot)
    assert len(obj_types) <= num_objects, "Hardcoded 'num_objects' is wrong!"

    # Reset learned object features for objects
    pos_obj_types = [None] * num_objects  # None means no pos preference
    pos_requires_grad = [train_pos] * num_objects

    if train_rot:
        if np.random.random() < 0.0:
            # Learn offsets
            rot_obj_types = [Params.IGNORE_ROT_IDX, Params.CARE_ROT_IDX]
            rot_requires_grad = [False] * num_objects
            rot_offset_requires_grad = [True] * num_objects
            rot_offsets = [None] * num_objects
        else:
            # Learn pref feats
            rot_obj_types = [None] * num_objects
            rot_requires_grad = [True] * num_objects
            rot_offset_requires_grad = [False] * \
                num_objects  # CUSTOM EXPERIMENT!!!
            rot_offsets = torch.from_numpy(
                Params.ori_offsets_2D).float().to(DEVICE)
            if len(rot_offsets.shape) == 1:
                rot_offsets = rot_offsets.unsqueeze(-1)
    else:
        rot_obj_types = [None] * num_objects
        rot_requires_grad = [False] * num_objects
        rot_offsets = None
        rot_offset_requires_grad = None

    policy.init_new_objs(pos_obj_types=pos_obj_types,
                         rot_obj_types=rot_obj_types,
                         rot_offsets=rot_offsets,
                         pos_requires_grad=pos_requires_grad, rot_requires_grad=rot_requires_grad,
                         rot_offset_requires_grad=rot_offset_requires_grad,
                         use_rand_init=True)

    losses, _ = perform_adaptation_learn2learn(policy, learned_opt, batch_data,
                                               train_pos=train_pos, train_rot=train_rot,
                                               **adapt_kwargs)
    return losses


def plot_losses(fig, ax_pos, ax_rot, all_pos_losses, all_rot_losses, figname):
    ax_pos.set_title("Position Loss vs Adaptation Step")
    for i in range(len(all_pos_losses)):
        ax_pos.plot(all_pos_losses[i], label="Epoch %d" % (i+1))
    ax_pos.legend()

    ax_rot.set_title("Rotation Loss vs Adaptation Step")
    for i in range(len(all_rot_losses)):
        ax_rot.plot(all_rot_losses[i], label="Epoch %d" % (i+1))
    ax_rot.legend()

    ax_pos.set_xticks(np.arange(0, len(all_pos_losses[0])))
    ax_rot.set_xticks(np.arange(0, len(all_rot_losses[0])))
    ax_pos.set_yticks(np.linspace(0, np.max(all_pos_losses), 5))
    ax_rot.set_yticks(np.linspace(0, np.max(all_rot_losses), 5))

    fig.supxlabel("Adaptation Step")
    fig.supylabel("Loss")
    fig.suptitle("Losses")
    plt.tight_layout()
    plt.savefig(figname)

    ax_pos.clear()
    ax_rot.clear()


def train(policy: Policy, learned_opt: LearnedOptimizer, train_args, saved_root: str,
          is_3D: bool, adapt_kwargs: dict):
    # can reduce if CPU RAM is not enough, just pre-loads data to reduce file reads
    buffer_size = 3000
    str_3D = "3D" if is_3D else "2D"
    train_pos_dataset = Dataset(
        root=f"data/pos_{str_3D}_train", buffer_size=buffer_size)
    train_rot_dataset = Dataset(
        root=f"data/rot_{str_3D}_train", buffer_size=buffer_size)

    batch_size = train_args.batch_size
    num_epochs = 7
    epochs_per_save = 1

    # Update over both pos and rot data one-by-one
    min_len = min(len(train_pos_dataset), len(train_rot_dataset))
    train_pos_indices = np.arange(min_len)
    train_rot_indices = np.arange(min_len)
    num_train_batches = min_len // batch_size

    fig, (ax_pos, ax_rot) = plt.subplots(1, 2, figsize=(10, 5))
    all_pos_losses = []
    all_rot_losses = []
    pbar = tqdm(total=num_epochs * num_train_batches)
    for epoch in range(num_epochs):
        np.random.shuffle(train_pos_indices)
        np.random.shuffle(train_rot_indices)
        epoch_pos_losses = np.zeros(adapt_kwargs["n_adapt_iters"])
        epoch_rot_losses = np.zeros(adapt_kwargs["n_adapt_iters"])

        # Train
        policy.policy_network.eval()
        learned_opt.train()

        for b in (range(num_train_batches)):
            # Position parameter adaptation
            # pos_batch_indices = train_pos_indices[b *
            #                                       batch_size:(b + 1) * batch_size]
            # pos_batch_data = load_batch(train_pos_dataset, pos_batch_indices)
            # pos_losses = train_helper(
            #     policy, learned_opt, pos_batch_data, train_pos=True, train_rot=False, adapt_kwargs=adapt_kwargs)
            # epoch_pos_losses += pos_losses

            # Rotation parameter adaptation
            rot_batch_indices = train_rot_indices[b *
                                                  batch_size:(b + 1) * batch_size]
            rot_batch_data = load_batch(train_rot_dataset, rot_batch_indices)
            rot_losses = train_helper(
                policy, learned_opt, rot_batch_data, train_pos=False, train_rot=True, adapt_kwargs=adapt_kwargs)
            epoch_rot_losses += rot_losses

            pbar.update(1)

            if b % 10 == 0:
                print(epoch_pos_losses / (b + 1))
                print(epoch_rot_losses / (b + 1))

            if b == 1 and epoch == 0:
                # Optionally save copy of all relevant scripts/files for reproducibility
                os.mkdir(saved_root)
                shutil.copy("model.py", os.path.join(saved_root, "model.py"))
                shutil.copy("data_params.py", os.path.join(
                    saved_root, "data_params.py"))
                shutil.copy("train.py", os.path.join(saved_root, "train.py"))
                shutil.copy("learn2learn.py", os.path.join(
                    saved_root, "learn2learn.py"))
                shutil.copy("online_adaptation.py", os.path.join(
                    saved_root, "online_adaptation.py"))

                # Save train args
                with open(os.path.join(saved_root, f"train_args.json"), "w") as outfile:
                    json.dump(vars(train_args), outfile, indent=4)

        avg_epoch_pos_losses = epoch_pos_losses / num_train_batches
        avg_epoch_rot_losses = epoch_rot_losses / num_train_batches
        print("Avg pos loss vs updates")
        print(avg_epoch_pos_losses)
        print("Avg rot loss vs updates")
        print(avg_epoch_rot_losses)
        all_pos_losses.append(avg_epoch_pos_losses)
        all_rot_losses.append(avg_epoch_rot_losses)

        if epoch % epochs_per_save == 0:
            torch.save(learned_opt.state_dict(), os.path.join(
                saved_root, "learned_opt_%d.h5" % (epoch + 1)))
            plot_losses(fig, ax_pos, ax_rot, all_pos_losses, all_rot_losses,
                        figname=os.path.join(saved_root, "adaptor_loss.png"))
            np.savez(os.path.join(saved_root, "losses"),
                     all_rot_losses=all_rot_losses, all_pos_losses=all_pos_losses)

    # Save final model
    torch.save(learned_opt.state_dict(), os.path.join(
        saved_root, "learned_opt_%d.h5" % (epoch + 1)))
    plot_losses(fig, ax_pos, ax_rot, all_pos_losses, all_rot_losses,
                figname=os.path.join(saved_root, "adaptor_loss.png"))
    np.savez(os.path.join(saved_root, "losses"),
             all_rot_losses=all_rot_losses, all_pos_losses=all_pos_losses)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_name', action='store',
                        type=str, help="learned opt save name", required=True)
    parser.add_argument('--model_name', action='store',
                        type=str, help="trained model name", required=True)
    parser.add_argument('--loaded_epoch', action='store', type=int,
                        default=0, help="training epoch of pretrained model to load from")

    # Default hyperparameters used for paper experiments, no need to tune to reproduce
    parser.add_argument('--lr', action='store', type=float,
                        default=1e-3, help="learning rate")
    parser.add_argument('--batch_size', action='store', type=int, default=16)

    # Learned Optimizer Arguments
    # hidden_dim, max_steps, tgt_lr, opt_lr
    parser.add_argument('--hidden_dim', action='store', type=int, default=32)
    parser.add_argument('--max_steps', action='store', type=int, default=5,
                        help="max number of consecutive gradient steps to take under same computation graph. Same as K for K-shot adaptation.")
    parser.add_argument('--tgt_lr', action='store', type=float, default=1e-1,
                        help="learning rate for the optimization target(policy params)")
    parser.add_argument('--opt_lr', action='store', type=float, default=1e-3,
                        help="learning rate for the optimizer")
    return parser.parse_args()


if __name__ == "__main__":
    opt_args = parse_arguments()
    with open(os.path.join(Params.model_root, opt_args.model_name, "train_args_pt_1.json"), "r") as f:
        model_args = json.load(f)

    is_3D = model_args["is_3D"]
    if is_3D:
        print("Training 3D Learn2Learn!")
        pos_dim = 3
        rot_dim = 6  # R * [x-axis, y-axis]
    else:
        print("Training 2D Learn2Learn!")
        pos_dim = 2
        rot_dim = 2  # cos(theta), sin(theta)

    # Define model
    network = PolicyNetwork(n_objects=Params.n_train_objects, pos_dim=pos_dim, rot_dim=rot_dim,
                            pos_preference_dim=model_args['pos_preference_dim'],
                            rot_preference_dim=model_args['rot_preference_dim'],
                            hidden_dim=model_args['hidden_dim'],
                            device=DEVICE).to(DEVICE)
    network.load_state_dict(
        torch.load(os.path.join(Params.model_root, opt_args.model_name, "model_%d.h5" % opt_args.loaded_epoch)))
    network.to(DEVICE)
    policy = Policy(network)

    learned_opt = LearnedOptimizer(
        device=DEVICE, max_steps=opt_args.max_steps, hidden_dim=opt_args.hidden_dim)  # param_dim=4
    learned_opt.to(DEVICE)

    # Set K-shot adaptation K = max_steps, meaning that learned optimizer only performs its own gradient update once per sample
    n_adapt_iters = opt_args.max_steps
    # dist of each rollout step of policy
    dstep = Params.dstep_3D if is_3D else Params.dstep_2D
    adapt_kwargs = dict(n_adapt_iters=n_adapt_iters,
                        dstep=dstep, clip_params=False)

    # Run training
    # torch.autograd.set_detect_anomaly(True)
    train(policy, learned_opt,
          saved_root=os.path.join(Params.model_root, opt_args.opt_name),
          train_args=opt_args,
          is_3D=is_3D,
          adapt_kwargs=adapt_kwargs)
