from socket import IP_DROP_MEMBERSHIP
from typing import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

import torch

from data_params import Params
from model import Policy, PolicyNetwork, decode_ori
from train import load_batch, process_batch_data, batch_inner_loop, DEVICE
from online_adaptation import adaptation_loss
from loss_plot_helpers import plot_2d_contour
from dataset import Dataset
from viz_2D import draw
from elastic_band import Object


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


def eval_performance_pos(policy: Policy, dataset, dstep, use_dot_prod=False):
    """
    Visually evalutes 2D policy with varying number of adaptation steps, where
    adaptation is performed directly on expert data.

    :param policy: Policy to evaluate
    :param test_data_root: Root directory of test data
    :return:
    """
    # Update over both pos and rot data one-by-one
    assert not dataset[0]["is_rot"]
    batch_size = 32
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    # num_batches = len(dataset) // batch_size
    num_batches = 20

    total_loss = 0.0
    for b in range(num_batches):
        batch_indices = indices[b * batch_size:(b + 1) * batch_size]
        batch_data = load_batch(dataset, batch_indices)
        if use_dot_prod:
            n_samples = 64
            with torch.no_grad():
                loss, _ = batch_inner_loop(policy.policy_network, batch_data=batch_data,
                                           n_samples=n_samples,
                                           train_rot=False,
                                           is_3D=False, is_training=False)
        else:
            batch_data_processed = process_batch_data(
                batch_data, train_rot=None, n_samples=None, is_full_traj=True)
            with torch.no_grad():
                loss, _ = adaptation_loss(policy.policy_network, batch_data_processed, dstep,
                                          train_pos=True, train_rot=False,
                                          goal_pos_radius_scale=1.0)

        total_loss += loss * len(batch_indices)

    return total_loss / len(dataset)


def eval_performance_rot(policy: Policy, dataset, dstep, care_rot):
    """
    Visually evalutes 2D policy with varying number of adaptation steps, where
    adaptation is performed directly on expert data.

    :param policy: Policy to evaluate
    :param test_data_root: Root directory of test data
    :return:
    """
    # Update over both pos and rot data one-by-one
    assert dataset[0]["is_rot"]
    batch_size = 32
    object_types = [dataset[i]['object_types'].item()
                    for i in range(len(dataset))]
    if care_rot:
        indices = [i for i in range(len(dataset))
                   if object_types[i] == Params.CARE_ROT_IDX]
    else:
        indices = [i for i in range(len(dataset))
                   if object_types[i] == Params.IGNORE_ROT_IDX]
    np.random.shuffle(indices)
    num_batches = min(20, int(np.ceil(len(indices) / batch_size)))

    total_loss = 0.0
    for b in range(num_batches):
        batch_indices = indices[b * batch_size:(b + 1) * batch_size]
        batch_data = load_batch(dataset, batch_indices)
        batch_data_processed = process_batch_data(
            batch_data, train_rot=None, n_samples=None, is_full_traj=True)
        with torch.no_grad():
            loss, pred_traj = adaptation_loss(policy.policy_network, batch_data_processed, dstep,
                                              train_pos=False, train_rot=True,
                                              goal_pos_radius_scale=1.0)

            (start_tensors, current_inputs, goal_tensors, goal_rot_inputs, object_inputs, obj_idx_tensors, _, _,
             traj_tensors) = batch_data_processed

            import ipdb
            ipdb.set_trace()
            start = start_tensors.cpu().numpy()[0]
            start = np.concatenate(
                [start[:2], decode_ori(start[2:]).flatten()])
            current = current_inputs.cpu().numpy()[0]
            goal = goal_tensors.cpu().numpy()[0]
            goal = np.concatenate(
                [goal[:2], decode_ori(goal[2:]).flatten()])
            object_types = obj_idx_tensors.cpu().numpy()[0]
            pred_traj = np.hstack([
                pred_traj.cpu().numpy()[0, :, :2],
                decode_ori(pred_traj.cpu().numpy()[0, :, 2:])
            ])
            expert_traj = np.hstack([
                traj_tensors.cpu().numpy()[0, :, :2],
                decode_ori(traj_tensors.cpu().numpy()[0, :, 2:])
            ])

            import ipdb
            ipdb.set_trace()

            object_poses = object_inputs[0, 0, :, :4].cpu().numpy()
            object_radii = object_inputs[0, 0, :, -1].cpu().numpy()
            theta_offsets = Params.ori_offsets_2D[object_types]
            objects = [
                Object(pos=object_poses[i][0:2], radius=object_radii[i],
                       ori=object_poses[i][-1] + theta_offsets[i]) for i in range(len(object_types))
            ]

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            draw(ax, start_pose=start, goal_pose=goal,
                 goal_radius=Params.agent_radius, agent_radius=Params.agent_radius,
                 object_types=object_types, offset_objects=objects,
                 pred_traj=pred_traj, expert_traj=expert_traj,
                 title=f"test", show_rot=True, hold=True)

        total_loss += loss * len(batch_indices)

    return total_loss / len(dataset)


def run_evaluation():
    model_name = "policy_2D"
    loaded_epoch = 100
    num_steps = 70
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

    # # Position
    # pos_feats_linspace = torch.linspace(
    #     pos_attract_feat.item() - 0.5, pos_repel_feat.item() + 0.5, num_steps).to(DEVICE).view(-1, 1)

    # param_loss_data = [[None] * num_steps for _ in range(num_steps)]
    # pbar = tqdm(total=num_steps * num_steps)
    # for i in range(num_steps):
    #     for j in range(num_steps):
    #         # Reset the attract and repel features separately
    #         obj_pos_feats = [pos_feats_linspace[i],
    #                          pos_feats_linspace[j]]  # Repel, Attract objs
    #         obj_rot_feats = [rot_care_feat] * 2
    #         obj_rot_offsets = [rot_offset_start] * 2  # alias fine, pure eval
    #         policy.update_obj_feats(
    #             obj_pos_feats, obj_rot_feats, obj_rot_offsets, same_var=False)

    #         pos_loss = eval_performance_pos(policy=policy, dataset=pos_dataset,
    #                                         dstep=dstep, use_dot_prod=True)
    #         param_loss_data[i][j] = (
    #             pos_feats_linspace[i].item(), pos_feats_linspace[j].item(), pos_loss.item())
    #         pbar.update(1)

    # np.save("param_loss_data_pos_dot_prod.npy", param_loss_data)

    # Rotation
    print(rot_ignore_feat, rot_care_feat)
    num_steps = 2
    rot_feats_linspace = torch.linspace(
        rot_ignore_feat.item(), rot_care_feat.item(), num_steps).to(DEVICE).view(-1, 1)
    rot_offsets_linspace = torch.linspace(
        0, 3 * np.pi / 2, num_steps).to(DEVICE).view(-1, 1)

    param_loss_data = [[None] * num_steps for _ in range(num_steps)]
    # pbar = tqdm(total=num_steps * num_steps)
    for i in range(num_steps):
        for j in range(num_steps):
            # Reset the attract and repel features separately
            obj_pos_feats = [pos_attract_feat]
            obj_rot_feats = [rot_feats_linspace[i]]
            # alias fine, pure eval
            # obj_rot_offsets = [rot_offsets_linspace[j]]
            obj_rot_offsets = [torch.tensor(np.pi / 2).view(-1, 1).to(DEVICE)]
            policy.update_obj_feats(
                obj_pos_feats, obj_rot_feats, obj_rot_offsets, same_var=False)

            print(obj_rot_feats, obj_rot_offsets)
            rot_loss_care = eval_performance_rot(policy=policy, dataset=rot_dataset,
                                                 dstep=dstep,
                                                 care_rot=True)
            rot_loss_ignore = eval_performance_rot(policy=policy, dataset=rot_dataset,
                                                   dstep=dstep,
                                                   care_rot=False)
            param_loss_data[i][j] = (
                rot_feats_linspace[i].item(), rot_offsets_linspace[j].item(), rot_loss_care.item(), rot_loss_ignore.item())
            # pbar.update(1)

    np.save("param_loss_data_rot.npy", param_loss_data)


def plot_evaluation():
    # # Position
    # loss_data_path = "param_loss_data_pos_cos_sim.npy"
    # data = np.load(loss_data_path)
    # y = np.array([data[i][0][0] for i in range(len(data))])  # repel
    # x = np.array([data[0][j][1] for j in range(len(data[0]))])   # attract
    # Z = [[None] * len(data) for _ in range(len(data))]
    # for i in range(len(data)):
    #     for j in range(len(data[0])):
    #         Z[i][j] = data[i][j][2]
    # Z = np.array(Z)

    # plot_2d_contour(x, y, Z, "pos_loss_cos_sim", vmin=0.1,
    #                 vmax=10, vlevel=0.5, show=True,
    #                 xlabel='Attract Feature', ylabel='Repel Feature')

    # Rotation
    loss_data_path = "param_loss_data_rot.npy"
    data = np.load(loss_data_path)
    y = np.array([data[i][0][0] for i in range(len(data))])  # rot feat
    x = np.array([data[0][j][1] for j in range(len(data[0]))])  # rot offset
    rot_loss_care = [[None] * len(data) for _ in range(len(data))]
    rot_loss_ignore = [[None] * len(data) for _ in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data[0])):
            rot_loss_care[i][j] = data[i][j][2]
            rot_loss_ignore[i][j] = data[i][j][3]
    rot_loss_care = np.array(rot_loss_care)
    rot_loss_ignore = np.array(rot_loss_ignore)

    import ipdb
    ipdb.set_trace()

    plot_2d_contour(x, y, rot_loss_care, "rot_loss_care", vmin=0.1,
                    vmax=10, vlevel=0.5, show=True,
                    xlabel='Rot Pref', ylabel='Rot Offset')

    plot_2d_contour(x, y, rot_loss_ignore, "rot_loss_ignore", vmin=0.1,
                    vmax=10, vlevel=0.5, show=True,
                    xlabel='Rot Pref', ylabel='Rot Offset')


if __name__ == "__main__":
    run_evaluation()
    # plot_evaluation()
