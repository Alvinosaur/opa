from typing import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

import torch

from data_params import Params
from model import Policy, PolicyNetwork
from recursive_least_squares.rls import RLS
from train import load_batch, DEVICE
from online_adaptation import perform_adaptation, perform_adaptation_rls, perform_adaptation_learn2learn
from learn2learn import LearnedOptimizer

from dataset import Dataset


def evaluate_helper(policy: Policy, adaptation_func, batch_data, train_pos, train_rot, n_adapt_iters, dstep, log_file):
    (traj, goal_radius, obj_poses, obj_radii, obj_types) = batch_data[0]
    # num_objects = len(obj_types)  # fails when rot data can contain "care" or "ignore" objects, but only one object in a given scene
    num_objects = 2  # (Attract, Repel pos) or (Care, Ignore rot)
    assert len(obj_types) <= num_objects, "Hardcoded 'num_objects' is wrong!"

    # Reset learned object features for objects
    pos_obj_types = [None] * num_objects  # None means no pos preference
    rot_obj_types = [None] * num_objects
    rot_offsets = [None] * num_objects
    pos_requires_grad = [train_pos] * num_objects
    rot_requires_grad = [train_rot] * num_objects
    rot_offset_requires_grad = [train_rot] * num_objects

    policy.init_new_objs(pos_obj_types=pos_obj_types,
                         rot_obj_types=rot_obj_types,
                         rot_offsets=rot_offsets,
                         pos_requires_grad=pos_requires_grad, rot_requires_grad=rot_requires_grad,
                         rot_offset_requires_grad=rot_offset_requires_grad,
                         use_rand_init=False)

    losses, _ = adaptation_func(policy=policy, batch_data=batch_data,
                                train_pos=train_pos, train_rot=train_rot,
                                n_adapt_iters=n_adapt_iters, dstep=dstep,
                                clip_params=False, verbose=True,
                                log_file=log_file)
    return losses


def evaluate_adaptation(policy: Policy, adaptation_func, dstep,
                        pos_dataset, rot_dataset,
                        n_adapt_iters, results_fname):
    """
    Visually evalutes 2D policy with varying number of adaptation steps, where
    adaptation is performed directly on expert data.

    :param policy: Policy to evaluate
    :param test_data_root: Root directory of test data
    :param train_pos: Whether to train position
    :param train_rot: Whether to train rotation
    :return:
    """
    # Update over both pos and rot data one-by-one
    batch_size = 16
    # min_len = min(len(pos_dataset), len(rot_dataset))
    min_len = 64  # TODO: temporary
    pos_indices = np.arange(min_len)
    rot_indices = np.arange(min_len)
    num_batches = min_len // batch_size

    # Total loss
    sum_pos_losses = np.zeros(n_adapt_iters)
    sum_rot_losses = np.zeros(n_adapt_iters)

    log_file = open(f"{results_fname}.txt", "w")

    pbar = tqdm(total=num_batches)
    for b in range(num_batches):
        # Position parameter adaptation
        batch_pos_indices = pos_indices[b *
                                        batch_size:(b + 1) * batch_size]
        pos_batch_data = load_batch(pos_dataset, batch_pos_indices)
        pos_losses = evaluate_helper(policy, adaptation_func, pos_batch_data,
                                     train_pos=True, train_rot=False, n_adapt_iters=n_adapt_iters, dstep=dstep,
                                     log_file=log_file)
        sum_pos_losses += np.array(pos_losses) * len(batch_pos_indices)

        # Rotation parameter adaptation
        batch_rot_indices = rot_indices[b *
                                        batch_size:(b + 1) * batch_size]
        rot_batch_data = load_batch(rot_dataset, batch_rot_indices)
        rot_losses = evaluate_helper(policy, adaptation_func, rot_batch_data,
                                     train_pos=False, train_rot=True, n_adapt_iters=n_adapt_iters, dstep=dstep,
                                     log_file=log_file)
        sum_rot_losses += np.array(rot_losses) * len(batch_rot_indices)
        pbar.update(1)

        # TODO: add functionality to enable 3 different learned optimizers
        # one for each parameter group
        # import ipdb
        # ipdb.set_trace()

    avg_pos_losses = sum_pos_losses / min_len
    avg_rot_losses = sum_rot_losses / min_len

    np.savez(results_fname,
             avg_pos_losses=avg_pos_losses, avg_rot_losses=avg_rot_losses)


def run_evaluation():
    save_dir = "eval_adaptation_results"
    model_name = "policy_2D"
    # model_name = "policy_3D"
    loaded_epoch = 100
    n_adapt_iters = 30

    with open(os.path.join(Params.model_root, model_name, "train_args_pt_1.json"), "r") as f:
        model_args = json.load(f)

    is_3D = model_args["is_3D"]
    if is_3D:
        pos_dim = 3
        rot_dim = 6  # R * [x-axis, y-axis]
    else:
        pos_dim = 2
        rot_dim = 2  # cos(theta), sin(theta)

    # Define model
    network = PolicyNetwork(n_objects=Params.n_train_objects, pos_dim=pos_dim, rot_dim=rot_dim,
                            pos_preference_dim=model_args['pos_preference_dim'],
                            rot_preference_dim=model_args['rot_preference_dim'],
                            hidden_dim=model_args['hidden_dim'],
                            device=DEVICE).to(DEVICE)
    network.load_state_dict(
        torch.load(os.path.join(Params.model_root, model_name, "model_%d.h5" % loaded_epoch)))
    network.to(DEVICE)
    policy = Policy(network)

    buffer_size = 3000
    str_3D = "3D" if is_3D else "2D"
    # NOTE: Use TEST dataset to evaluate
    pos_dataset = Dataset(
        root=f"data/pos_{str_3D}_test", buffer_size=buffer_size)
    rot_dataset = Dataset(
        root=f"data/rot_{str_3D}_test", buffer_size=buffer_size)
    dstep = Params.dstep_3D if is_3D else Params.dstep_2D

    # # ########### RLS###########
    # rls = RLS(alpha=0.5, lmbda=0.3)
    # rls_name = f"eval_RLS_alpha_{rls.alpha}_lmbda_{rls.lmbda}"
    # adaptation_func = lambda *args, **kwargs: perform_adaptation_rls(
    #     rls=rls, *args, **kwargs)
    # results_fname = os.path.join(save_dir, rls_name)
    # evaluate_adaptation(policy, adaptation_func, dstep,
    #                     pos_dataset, rot_dataset,
    #                     n_adapt_iters, results_fname)

    # # ########### Adam ###########
    # lr = 1e-1
    # results_fname = os.path.join(save_dir, "Adam")
    # adaptation_func = lambda *args, **kwargs: perform_adaptation(
    #     Optimizer=torch.optim.Adam, lr=lr, *args, **kwargs)
    # evaluate_adaptation(policy, adaptation_func, dstep,
    #                     pos_dataset, rot_dataset,
    #                     n_adapt_iters, results_fname)

    # # ########### SGD ###########
    # lr = 0.3
    # results_fname = os.path.join(save_dir, "SGD")
    # adaptation_func = lambda *args, **kwargs: perform_adaptation(
    #     Optimizer=torch.optim.SGD, lr=lr, *args, **kwargs)
    # evaluate_adaptation(policy, adaptation_func, dstep,
    #                     pos_dataset, rot_dataset,
    #                     n_adapt_iters, results_fname)

    # ########### (SEPARATE) Learn2Learn ###########
    learn2learn_name = "learned_opt_pos_and_rot_rand_choice"
    results_fname = os.path.join(save_dir, learn2learn_name)
    args_path = os.path.join(
        Params.model_root, learn2learn_name, "train_args.json")
    with open(args_path, "r") as f:
        learn2learn_args = json.load(f)
    learned_opt = LearnedOptimizer(device=DEVICE, max_steps=1,
                                   tgt_lr=learn2learn_args['tgt_lr'],
                                   opt_lr=learn2learn_args['opt_lr'],
                                   hidden_dim=learn2learn_args['hidden_dim'],)
    learned_opt.load_state_dict(
        torch.load(os.path.join(Params.model_root, learn2learn_name, "learned_opt_3.h5")))
    learned_opt.to(DEVICE)
    learned_opt.eval()

    adaptation_func = lambda *args, **kwargs: perform_adaptation_learn2learn(
        learned_opt=learned_opt, *args, **kwargs)
    evaluate_adaptation(policy, adaptation_func, dstep,
                        pos_dataset, rot_dataset,
                        n_adapt_iters, results_fname)

    # # ########### (SHARED) Learn2Learn ###########
    # learn2learn_name = "learned_opt_pos_and_rot_rand_choice"
    # results_fname = os.path.join(save_dir, learn2learn_name)
    # args_path = os.path.join(
    #     Params.model_root, learn2learn_name, "train_args.json")
    # with open(args_path, "r") as f:
    #     learn2learn_args = json.load(f)
    # learned_opt = LearnedOptimizer(device=DEVICE, max_steps=1,
    #                                tgt_lr=learn2learn_args['tgt_lr'],
    #                                opt_lr=learn2learn_args['opt_lr'],
    #                                hidden_dim=learn2learn_args['hidden_dim'],)
    # learned_opt.load_state_dict(
    #     torch.load(os.path.join(Params.model_root, learn2learn_name, "learned_opt_3.h5")))
    # learned_opt.to(DEVICE)
    # learned_opt.eval()

    # adaptation_func = lambda *args, **kwargs: perform_adaptation_learn2learn(
    #     learned_opt=learned_opt, *args, **kwargs)
    # evaluate_adaptation(policy, adaptation_func, dstep,
    #                     pos_dataset, rot_dataset,
    #                     n_adapt_iters, results_fname)


if __name__ == "__main__":
    run_evaluation()
