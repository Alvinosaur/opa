from ast import Assert
import os
from sqlite3 import adapt
import numpy as np
import copy
import argparse
import json
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


def w(v):
    if cuda:
        return v.cuda()
    return v


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

    def __init__(self, device, preproc=False, hidden_dim=20, preproc_factor=10.0,
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
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_dim)
        else:
            self.recurs = nn.LSTMCell(1, hidden_dim)
        self.recurs2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)

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
        if self.preproc:
            # Implement preproc described in Appendix A

            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = inp.data
            inp2 = torch.zeros(inp.size()[0], 2, device=self.device)
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (
                torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (
                float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = Variable(inp2)

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

    def step(self, new_loss, params, verbose=False):
        if self.param_size is None:
            # Initialize hidden and cell states
            self.param_size = sum(p.numel() for p in params)
            zero_init = [
                w(Variable(torch.zeros(self.param_size, self.hidden_dim, device=self.device))) for _ in range(2)]
            self.hidden_states = copy.deepcopy(zero_init)
            self.hidden_states_temp = copy.deepcopy(zero_init)
            self.cell_states = copy.deepcopy(zero_init)
            self.cell_states_temp = copy.deepcopy(zero_init)

        new_loss.backward(retain_graph=self.training)
        self.loss_to_optimize = self.loss_to_optimize + new_loss
        self.num_steps += 1

        offset = 0
        new_params = []
        for p in params:
            cur_sz = p.numel()
            # p_indices: [offset:offset + cur_sz]
            p_indices = torch.arange(offset, offset + cur_sz, device=DEVICE)

            # Get original gradients and feed as input to learned optimizer
            try:
                if p.grad is None:
                    raise GradIsNoneError()
                gradients = detach_var(p.grad.view(cur_sz, 1))
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
                    -gradients.item(), updates.item(), new_p.item()))

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


def train_helper(policy: Policy, learned_opt: LearnedOptimizer, batch_data, train_pos, train_rot, adapt_kwargs):
    (traj, goal_radius, obj_poses, obj_radii, obj_types) = batch_data[0]
    # num_objects = len(obj_types)  # fails when rot data can contain "care" or "ignore" objects, but only one object in a given scene
    num_objects = 2  # (Attract, Repel pos) or (Care, Ignore rot)
    assert len(obj_types) <= num_objects, "Hardcoded 'num_objects' is wrong!"

    # Reset learned object features for objects
    pos_obj_types = [None] * num_objects  # None means no pos preference
    pos_requires_grad = [train_pos] * num_objects
    rot_obj_types = [None] * num_objects  # None means no rot preference
    rot_requires_grad = [train_rot] * num_objects
    policy.init_new_objs(pos_obj_types=pos_obj_types, rot_obj_types=rot_obj_types,
                         pos_requires_grad=pos_requires_grad, rot_requires_grad=rot_requires_grad)

    # Reset hidden states of learned_opt
    learned_opt.reset_lstm()

    losses, _ = perform_adaptation_learn2learn(policy, learned_opt, batch_data,
                                               train_pos=train_pos, train_rot=train_rot,
                                               **adapt_kwargs)
    return losses


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
    num_epochs = 5
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
        batch_pos_losses = np.zeros(adapt_kwargs["n_adapt_iters"])
        batch_rot_losses = np.zeros(adapt_kwargs["n_adapt_iters"])

        # Train
        policy.policy_network.eval()
        learned_opt.train()

        for b in (range(num_train_batches)):
            # Position parameter adaptation
            pos_indices = train_pos_indices[b *
                                            batch_size:(b + 1) * batch_size]
            pos_batch_data = load_batch(train_pos_dataset, pos_indices)
            pos_losses = train_helper(
                policy, learned_opt, pos_batch_data, train_pos=True, train_rot=False, adapt_kwargs=adapt_kwargs)
            batch_pos_losses += pos_losses

            # Rotation parameter adaptation
            # rot_indices = train_rot_indices[b *
            #                                 batch_size:(b + 1) * batch_size]
            # rot_batch_data = load_batch(train_rot_dataset, rot_indices)
            # rot_losses = train_helper(
            #     policy, learned_opt, rot_batch_data, train_pos=False, train_rot=True, adapt_kwargs=adapt_kwargs)
            # batch_rot_losses += rot_losses
            pbar.update(1)

        avg_batch_pos_losses = batch_pos_losses / batch_size
        avg_batch_rot_losses = batch_rot_losses / batch_size
        all_pos_losses.append(avg_batch_pos_losses)
        all_rot_losses.append(avg_batch_rot_losses)

        if epoch == 0:
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

        if epoch % epochs_per_save == 0:
            torch.save(learned_opt.state_dict(), os.path.join(
                saved_root, "learned_opt_%d.h5" % (epoch + 1)))

            ax_pos.set_title("Position Loss Per Adaptation Step")
            for i in range(epoch+1):
                ax_pos.plot(all_pos_losses[i], label="Epoch %d" % epoch)
            ax_pos.legend()

            ax_rot.set_title("Rotation Loss Per Adaptation Step")
            for i in range(epoch+1):
                ax_rot.plot(all_rot_losses[i], label="Epoch %d" % epoch)
            ax_rot.legend()

            plt.savefig(os.path.join(saved_root, "adaptor_loss.png"))
            plt.clf()

    # Save final model
    torch.save(learned_opt.state_dict(), os.path.join(
        saved_root, "learned_opt_%d.h5" % (epoch + 1)))

    ax_pos.set_title("Position Loss Per Adaptation Step")
    for i in range(epoch+1):
        ax_pos.plot(all_pos_losses[i], label="Epoch %d" % epoch)
    ax_pos.legend()

    ax_rot.set_title("Rotation Loss Per Adaptation Step")
    for i in range(epoch+1):
        ax_rot.plot(all_rot_losses[i], label="Epoch %d" % epoch)
    ax_rot.legend()

    plt.savefig(os.path.join(saved_root, "adaptor_loss.png"))
    plt.clf()

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
    policy = Policy(network)

    learned_opt = LearnedOptimizer(
        device=DEVICE, max_steps=opt_args.max_steps, hidden_dim=opt_args.hidden_dim)
    learned_opt.to(DEVICE)

    # Set K-shot adaptation K = max_steps, meaning that learned optimizer only performs its own gradient update once per sample
    n_adapt_iters = opt_args.max_steps
    # dist of each rollout step of policy
    dstep = Params.dstep_3D if is_3D else Params.dstep_2D
    adapt_kwargs = dict(n_adapt_iters=n_adapt_iters, dstep=dstep,)

    # Run training
    # torch.autograd.set_detect_anomaly(True)
    train(policy, learned_opt,
          saved_root=os.path.join(Params.model_root, opt_args.opt_name),
          train_args=opt_args,
          is_3D=is_3D,
          adapt_kwargs=adapt_kwargs)
