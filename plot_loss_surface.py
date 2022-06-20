from typing import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

import torch

from data_params import Params
from model import Policy, PolicyNetwork
from train import load_batch, process_batch_data, DEVICE
from online_adaptation import adaptation_loss

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


def eval_performance(policy: Policy, dataset, dstep, calc_pos, calc_rot):
    """
    Visually evalutes 2D policy with varying number of adaptation steps, where
    adaptation is performed directly on expert data.

    :param policy: Policy to evaluate
    :param test_data_root: Root directory of test data
    :return:
    """
    # Update over both pos and rot data one-by-one
    batch_size = 32
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    # num_batches = len(dataset) // batch_size
    num_batches = 20

    total_loss = 0.0
    for b in range(num_batches):
        # Position parameter adaptation
        batch_indices = indices[b * batch_size:(b + 1) * batch_size]
        batch_data = load_batch(dataset, batch_indices)
        batch_data_processed = process_batch_data(
            batch_data, train_rot=None, n_samples=None, is_full_traj=True)
        with torch.no_grad():
            loss, _ = adaptation_loss(policy.policy_network, batch_data_processed, dstep,
                                      train_pos=calc_pos, train_rot=calc_rot,
                                      goal_pos_radius_scale=1.0)

        total_loss += loss * len(batch_indices)

    return total_loss / len(dataset)


def run_evaluation():
    model_name = "policy_2D"
    loaded_epoch = 70
    dstep = Params.dstep_2D

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

    pos_repel_feat = policy.policy_network.pos_pref_feat_train[Params.REPEL_IDX].detach(
    )
    pos_attract_feat = policy.policy_network.pos_pref_feat_train[Params.ATTRACT_IDX].detach(
    )
    rot_care_feat = policy.policy_network.rot_pref_feat_train[Params.CARE_ROT_IDX].detach(
    )
    rot_ignore_feat = policy.policy_network.rot_pref_feat_train[Params.IGNORE_ROT_IDX].detach(
    )
    rot_offset_start = policy.policy_network.rot_offsets_train[Params.START_IDX].detach(
    )

    # Position
    num_steps = 100
    pos_feats_linspace = torch.linspace(
        pos_attract_feat.item() - 0.5, pos_repel_feat.item() + 0.5, num_steps).to(DEVICE)

    param_loss_data = [[None] * num_steps for _ in range(num_steps)]
    pbar = tqdm(total=num_steps * num_steps)
    for i in range(num_steps):
        for j in range(num_steps):
            # Reset the attract and repel features separately
            obj_pos_feats = [pos_feats_linspace[i].view(
                1), pos_feats_linspace[j].view(1)]
            obj_rot_feats = [rot_care_feat] * 2
            obj_rot_offsets = [rot_offset_start] * 2  # alias fine, pure eval
            policy.update_obj_feats(
                obj_pos_feats, obj_rot_feats, obj_rot_offsets, same_var=False)

            pos_loss = eval_performance(policy=policy, dataset=pos_dataset,
                                        dstep=dstep, calc_pos=True, calc_rot=False)
            param_loss_data[i][j] = (
                pos_feats_linspace[i].item(), pos_feats_linspace[j].item(), pos_loss.item())
            pbar.update(1)

    np.save("param_loss_data_pos.npy", param_loss_data)


if __name__ == "__main__":
    run_evaluation()
