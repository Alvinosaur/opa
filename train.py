"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import random
import argparse
from tqdm import tqdm
from typing import *
import json

import torch

from dataset import Dataset
from data_params import Params
from model import PolicyNetwork, encode_ori_3D, encode_ori_2D, pose_to_model_input

seed = 444
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

cuda = torch.cuda.is_available()
DEVICE = "cuda:0" if cuda else "cpu"
if cuda:
    print("CUDA GPU!")
else:
    print("CPU!")


def load_batch(dataset: Dataset, indices: Iterable[int]):
    """
    Loads a batch of data specified by sample indices.
    :param dataset: Dataset object
    :param indices: list of indices
    :return: List of tuples containing:
        traj: trajectory
        goal_radii: goal radius
        obj_poses: object poses
        obj_radii: object radii
        obj_types: object indices/types
    """
    batch_trajs = []
    for idx in indices:
        traj = dataset[idx]["states"]
        obj_poses = np.copy(dataset[idx]["object_poses"])
        obj_radii = np.copy(dataset[idx]["object_radii"])[:, np.newaxis]
        obj_types = np.copy(dataset[idx]["object_types"].astype(int))
        goal_radius = np.array([dataset[idx]["goal_radius"]])
        batch_trajs.append(
            (traj, goal_radius, obj_poses, obj_radii, obj_types))
    return batch_trajs


def sample_starts_tgts(traj: np.ndarray, goals_r: np.ndarray,
                       objs: np.ndarray, objs_r: np.ndarray, obj_idxs: np.ndarray,
                       n_samples: int, train_rot: bool):
    """
    Given a trajectory, randomly sample individual pairs of states (x_t, x_{t+1}) to
    define ground truth action a_t*. Given state x_t, goal, and objects, model should
    predict a_t.

    :param traj: trajectory (T, pos_dim+ori_dim)
    :param goals_r: goal radius
    :param objs: object poses (n_objs, pos_dim+ori_dim)
    :param objs_r: object radii (n_objs, 1)
    :param obj_idxs: object indices (n_objs,)
    :param n_samples: (N) number of samples to draw
    :param train_rot: whether to train rotation. If False, only train position
    :return: Tuple:
        currents: sampled current poses (N x pos_dim)
        goals_rep: same final goal, but repeated (N x pos_dim)
        goals_r_rep: goal radii, but repeated (N x 1)
        target_ori: target orientation that model should predict
        targets: target poses (N x pos_dim+ori_dim)
        objs_rep: objects, repeated (N x n_objs x pos_dim+ori_dim)
        objs_r_rep: object radii, repeated (N x n_objs x 1)
        obj_idxs_rep: object indices, repeated (N x n_objs x 1)
    """
    # Ensure data types are correct
    traj = traj.astype(np.float32)
    goals_r = goals_r.astype(np.float32)
    objs = objs.astype(np.float32)
    objs_r = objs_r.astype(np.float32)
    obj_idxs = obj_idxs.astype(int)

    # prediction horizon is 1 timestep
    pred_horizon_rep = np.full(shape=(n_samples, 1), fill_value=1)
    T = traj.shape[0]

    assert traj.shape[1] in [3, 7]
    is_3D = traj.shape[1] == 3 + 4  # x, y, z, qw, qx, qy, qz
    pos_dim = 3 if is_3D else 2

    if train_rot:
        # For training rotation, separate timesteps with zero and nonzero change in orientation
        dtheta = np.sum(
            np.abs(traj[1:, pos_dim:] - traj[:-1, pos_dim:]), axis=-1)
        nonzero_indices = np.where(np.logical_not(
            np.isclose(dtheta, 0, atol=1e-5)))[0]
        zero_indices = np.where(np.isclose(dtheta, 0, atol=1e-5))[0]

        # Ensure a certain proportion of samples include non-zero change in orientation
        #   this is important because most timesteps have zero change, which
        #   can cause model to simply collapse to predicting zero change always.
        nonzero_prop = 0.4
        n_samples = min(T, n_samples)
        n_nonzero = int(nonzero_prop * n_samples)
        n_zero = n_samples - n_nonzero
        nonzero_samples = np.random.choice(nonzero_indices, size=n_nonzero)
        try:
            zero_samples = np.random.choice(zero_indices, size=n_zero)
        except:
            zero_samples = np.random.choice(T - 1, n_zero)

        # Define timesteps for (current, target/future)
        cur_indices = np.concatenate([nonzero_samples, zero_samples])
        future_indices = cur_indices + 1  # always 1 step ahead

    else:  # train_pos
        # For position, uniformly sample across trajectory
        cur_indices = np.random.randint(low=0, high=T - 1, size=n_samples)
        future_indices = cur_indices + np.min(
            np.concatenate(
                [T - (cur_indices[:, np.newaxis] + 1), pred_horizon_rep], axis=-1),
            axis=-1)

    # Calculate
    # TODO: Can add noise to current pose while predicting same target
    currents = traj[cur_indices]
    vec = traj[future_indices, 0:pos_dim] - currents[:, 0:pos_dim]
    target_trans = vec / np.linalg.norm(vec, axis=-1)[:, np.newaxis]

    if is_3D:
        target_ori = encode_ori_3D(traj[future_indices, pos_dim:])
    else:
        target_ori = encode_ori_2D(traj[future_indices, -1])

    goals_rep = traj[-1][np.newaxis, :].repeat(n_samples, axis=0)
    goals_r_rep = goals_r[np.newaxis, :].repeat(n_samples, axis=0)
    starts_rep = traj[0][np.newaxis, :].repeat(n_samples, axis=0)
    objs_rep = objs[np.newaxis, :, :].repeat(n_samples, axis=0)
    objs_r_rep = objs_r[np.newaxis, :, :].repeat(n_samples, axis=0)
    obj_idxs_rep = obj_idxs[np.newaxis, :].repeat(n_samples, axis=0)

    return (starts_rep, currents,
            goals_rep, goals_r_rep,
            target_trans, target_ori,
            objs_rep, objs_r_rep, obj_idxs_rep)


def process_batch_data(batch_data, train_rot, n_samples, is_full_traj=False):
    """
    Given List batch samples/trajectories, randomly sample n_samples from each
    and concatenate all model inputs/outputs into batch tensors.

    :param batch_data: List of samples/trajectories
    :param train_rot: whether to train rotation. If False, only train position
    :param n_samples: number of samples to take from each trajectory
    :return: Tuple of:
        start_rot_inputs: rotation-network starting poses in model input form (B*n_samples, 1, input_dim)
        current_inputs: sampled current poses in model input form (B*n_samples, 1, input_dim)
        goal_inputs: goal poses in model input form (B*n_samples, 1, input_dim)
        goal_rot_inputs: rotation-network goal poses in model input form (B*n_samples, 1, input_dim)
        object_inputs: object poses in model input form (B*n_samples, num_objects, input_dim)
        obj_idx_tensors: object indices/types (B*n_samples, 1, input_dim)
        all_tgt_ori_tensors: target orientations (B*n_samples, 2(2D) or 6(3D))
        all_tgt_trans_tensors: target translations (B*n_samples, 2(2D) or 3(3D))
    """
    all_starts = []  # starting pose of trajectories
    all_currents = []  # randomly sampled current poses x_t
    all_goals = []  # final pose of trajectories
    all_goals_r = []  # goal radius of trajectories
    all_tgt_ori = []  # target orientation actions
    all_tgt_trans = []  # target translation actions
    all_objs = []  # objects present for each sample
    all_objs_r = []  # object's radii
    all_obj_idxs = []  # object's indices/types
    all_trajs = []

    # Loop through batch trajectories and sample n_samples from each
    #  and concatenate all model inputs/outputs into batch tensors
    min_T = min([len(sample[0]) for sample in batch_data])
    for sample in batch_data:
        if is_full_traj:
            (traj, goals_r, objs, objs_r, obj_idxs) = sample

            # truncate trajectory to min_T with random start and end
            start_idx = np.random.randint(
                low=0, high=len(traj) - min_T + 1)

            starts = traj[0]  # original start before truncation
            goals = traj[-1]
            currents = traj[start_idx]
            traj = traj[start_idx:start_idx + min_T]

            starts = starts[np.newaxis]
            currents = currents[np.newaxis]
            traj = traj[np.newaxis]
            goals = goals[np.newaxis]
            goals_r = np.array([goals_r])
            obj_idxs = obj_idxs[np.newaxis]

            # Repeat T times if object poses/radii are static
            if len(objs.shape) == 2:  # num_objects x pose_dim
                objs = objs[np.newaxis].repeat(min_T, axis=0)
                objs_r = objs_r[np.newaxis].repeat(min_T, axis=0)

            objs = objs[np.newaxis]
            objs_r = objs_r[np.newaxis]
        else:
            (traj, goals_r, objs, objs_r, obj_idxs) = sample
            # pos_noise = np.random.normal(loc=0, scale=pos_noise_std, size=(n_samples, pos_dim))
            (starts, currents, goals, goals_r, tgt_trans, tgt_ori, objs, objs_r, obj_idxs) = (
                sample_starts_tgts(traj=traj, goals_r=goals_r,
                                   objs=objs, objs_r=objs_r,
                                   obj_idxs=obj_idxs, n_samples=n_samples,
                                   train_rot=train_rot))

        all_starts.append(starts)
        all_currents.append(currents)
        all_goals.append(goals)
        all_goals_r.append(goals_r)
        all_objs.append(objs)
        all_objs_r.append(objs_r)
        all_obj_idxs.append(obj_idxs)

        if is_full_traj:
            all_trajs.append(traj)
        else:
            all_tgt_ori.append(tgt_ori)
            all_tgt_trans.append(tgt_trans)

    # Convert to tensors
    start_tensors = torch.from_numpy(pose_to_model_input(
        np.vstack(all_starts))).to(torch.float32).to(DEVICE)
    current_tensors = torch.from_numpy(pose_to_model_input(
        np.vstack(all_currents))).to(torch.float32).to(DEVICE)
    goal_tensors = torch.from_numpy(pose_to_model_input(
        np.vstack(all_goals))).to(torch.float32).to(DEVICE)
    obj_r_tensors = torch.from_numpy(
        np.vstack(all_objs_r)).to(torch.float32).to(DEVICE)
    obj_idx_tensors = torch.from_numpy(
        np.vstack(all_obj_idxs)).to(torch.long).to(DEVICE)

    # Convert to expected model input form: [pose, radius]
    # Object
    # flatten during conversion then reshape back to original shape
    all_objs = np.vstack(all_objs)
    if is_full_traj:
        B, T, n_objects, _ = all_objs.shape
        all_objs = pose_to_model_input(all_objs.reshape(
            B * T * n_objects, -1)).reshape(B, T, n_objects, -1)
    else:
        B, n_objects, _ = all_objs.shape
        all_objs = pose_to_model_input(all_objs.reshape(
            [B * n_objects, -1])).reshape([B, n_objects, -1])

    obj_tensors = torch.from_numpy(all_objs).to(torch.float32).to(DEVICE)
    object_inputs = torch.cat([obj_tensors, obj_r_tensors], dim=-1)

    # Current agent pose in the expected input form
    agent_radii = torch.tensor(
        [Params.agent_radius], device=DEVICE).view(1, 1).repeat(B, 1)
    current_inputs = torch.cat(
        [current_tensors, agent_radii], dim=-1).unsqueeze(1)

    # Goal for model's rotational relation network
    goal_rot_radii = torch.from_numpy(
        np.vstack(all_goals_r)).to(torch.float32).to(DEVICE)
    goal_rot_inputs = torch.cat(
        [goal_tensors, goal_rot_radii], dim=-1).unsqueeze(1)

    traj_tensors = None
    all_tgt_ori_tensors = None
    all_tgt_trans_tensors = None
    if is_full_traj:
        # Trajectory
        all_trajs = np.vstack(all_trajs)
        all_trajs = pose_to_model_input(
            all_trajs.reshape(B * T, -1)).reshape(B, T, -1)
        traj_tensors = torch.from_numpy(all_trajs).to(torch.float32).to(DEVICE)

    else:
        all_tgt_ori_tensors = torch.from_numpy(
            np.vstack(all_tgt_ori)).to(torch.float32).to(DEVICE)
        all_tgt_trans_tensors = torch.from_numpy(
            np.vstack(all_tgt_trans)).to(torch.float32).to(DEVICE)

    return (start_tensors, current_inputs,
            goal_tensors, goal_rot_inputs,
            object_inputs, obj_idx_tensors,
            all_tgt_ori_tensors, all_tgt_trans_tensors, traj_tensors)


def batch_inner_loop(model: PolicyNetwork, batch_data: List[Tuple],
                     is_3D: bool,
                     train_rot: bool,
                     n_samples=8, is_training=True):
    """


    :param model: Model to train
    :param batch_data: List of batch samples/trajectories
    :param is_3D: Whether the model is 3D
    :param train_rot: Whether to train the rotational relation network.
                      If False, only train the position network
    :param n_samples: Number of samples to take from each trajectory
    :return:
    """
    pos_dim = 3 if is_3D else 2
    (start_tensors, current_inputs,
     goal_tensors, goal_rot_inputs,
     object_inputs, obj_type_tensors,
     all_tgt_ori_tensors, all_tgt_trans_tensors, _) = process_batch_data(batch_data, n_samples=n_samples,
                                                                         train_rot=train_rot)

    # Start for model's rotational relation network
    start_rot_radii = torch.norm(
        start_tensors[:, 0:pos_dim] - current_inputs[:, 0, 0:pos_dim], dim=-1).unsqueeze(-1)
    start_rot_inputs = torch.cat(
        [start_tensors, start_rot_radii], dim=-1).unsqueeze(1)

    # Goal Pos
    goal_radii = torch.norm(
        goal_tensors[:, 0:pos_dim] - current_inputs[:, 0, 0:pos_dim], dim=-1).unsqueeze(-1)
    goal_inputs = torch.cat([goal_tensors, goal_radii], dim=-1).unsqueeze(1)

    output_vec, output_ori, pos_effects = model(current=current_inputs,
                                                start=start_rot_inputs,
                                                goal=goal_inputs, goal_rot=goal_rot_inputs,
                                                objects=object_inputs,
                                                object_indices=obj_type_tensors,
                                                calc_pos=not train_rot,
                                                calc_rot=train_rot,
                                                is_training=is_training)

    pos_loss = torch.tensor([0.0], device=DEVICE)
    theta_loss = torch.tensor([0.0], device=DEVICE)
    if train_rot:
        if is_3D:
            output_x = output_ori[:, 0:3]
            output_y = output_ori[:, 3:6]
            tgt_x = all_tgt_ori_tensors[:, 0:3]
            tgt_y = all_tgt_ori_tensors[:, 3:6]
            # misalignment between pred and ground truth x and y axes of rotation matrix
            theta_loss = ((1 - torch.bmm(output_x.unsqueeze(1), tgt_x.unsqueeze(-1))).mean() +
                          (1 - torch.bmm(output_y.unsqueeze(1), tgt_y.unsqueeze(-1))).mean())
        else:
            # misalignment between pred and ground truth translation direction
            theta_loss = (1 - torch.bmm(output_ori.unsqueeze(1),
                          all_tgt_ori_tensors.unsqueeze(-1))).mean()
    else:  # train_pos
        pos_loss = (1 - torch.bmm(output_vec.unsqueeze(1),
                    all_tgt_trans_tensors.unsqueeze(-1))).mean()

    return pos_loss, theta_loss


def train(model: PolicyNetwork, train_args: argparse.Namespace, saved_root: str,
          loaded_epoch=0):
    """
    Train the model on the given data for specified number of epochs

    :param model: Model to train
    :param train_args: Training arguments
    :param saved_root: Root directory to save model to
    :param loaded_epoch: Epoch to start training from if loading pretrained model.
                         This is necessary when training pos and rot networks separately.
    :return:
    """
    buffer_size = 3000  # can reduce if CPU RAM is not enough, just pre-loads data to reduce file reads
    assert train_args.data_name
    test_dataset = Dataset(
        root=f"data/{train_args.data_name}_test", buffer_size=buffer_size)
    train_dataset = Dataset(
        root=f"data/{train_args.data_name}_train", buffer_size=buffer_size)

    batch_size = train_args.batch_size
    n_samples = train_args.n_samples
    train_rot = train_args.is_rot
    is_3D = train_args.is_3D

    # Verify using the right datasets
    assert train_rot == test_dataset[0]['is_rot'].item()
    assert train_rot == train_dataset[0]['is_rot'].item()
    assert is_3D == test_dataset[0]["is_3D"].item()
    assert is_3D == train_dataset[0]["is_3D"].item()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_args.lr)
    if train_rot:  # NOTE: Train rot OR pos, not both simultaneously because data is different
        num_epochs = 50
    else:  # train_pos=True
        num_epochs = 50
    epochs_per_save = 10
    epochs_per_test = 10

    train_indices = np.arange(len(train_dataset))
    test_indices = np.arange(0, len(test_dataset))
    num_train_batches = len(train_indices) // batch_size
    num_test_batches = len(test_indices) // batch_size
    all_train_losses = []
    all_train_pos_losses = []
    all_train_ori_losses = []
    all_test_losses = []
    all_test_pos_losses = []
    all_test_ori_losses = []

    # epochs start at loaded_epoch. If loaded_epoch == 0, the normal
    # file-saving and folder-checking will happen
    # but if loaded_epoch > 0, then skip file-saving and avoid overwriting
    # previously saved models
    if loaded_epoch > 0:
        loss_name = "loss_phase_2"
        args_name = "train_args_pt_2"
    else:
        loss_name = "loss"
        args_name = "train_args_pt_1"

    for epoch in tqdm(range(loaded_epoch, num_epochs + loaded_epoch)):
        np.random.shuffle(train_indices)
        avg_train_loss = 0.0
        avg_train_pos_loss = 0.0
        avg_train_ori_loss = 0.0

        # Train
        model.train()
        for b in (range(num_train_batches)):
            indices = train_indices[b * batch_size:(b + 1) * batch_size]
            batch_data = load_batch(train_dataset, indices)
            train_pos_loss, train_ori_loss = batch_inner_loop(model, batch_data=batch_data,
                                                              n_samples=n_samples,
                                                              train_rot=train_rot,
                                                              is_3D=is_3D)

            batch_train_loss = train_ori_loss if train_rot else train_pos_loss
            optimizer.zero_grad()
            batch_train_loss.backward()
            optimizer.step()

            avg_train_loss += batch_train_loss.item()
            avg_train_pos_loss += train_pos_loss.item()
            avg_train_ori_loss += train_ori_loss.item()

        all_train_losses.append(avg_train_loss / num_train_batches)
        all_train_pos_losses.append(avg_train_pos_loss / num_train_batches)
        all_train_ori_losses.append(avg_train_ori_loss / num_train_batches)

        if epoch % epochs_per_test == 0:
            avg_test_loss = 0.0
            avg_test_pos_loss = 0.0
            avg_test_ori_loss = 0.0

            # Test
            model.eval()
            for b in range(num_test_batches):
                indices = test_indices[b * batch_size:(b + 1) * batch_size]
                batch_data = load_batch(test_dataset, indices)

                with torch.no_grad():
                    test_pos_loss, test_ori_loss = batch_inner_loop(model, batch_data=batch_data,
                                                                    n_samples=n_samples,
                                                                    train_rot=train_rot,
                                                                    is_3D=is_3D)

                batch_test_loss = test_ori_loss if train_rot else test_pos_loss
                avg_test_loss += batch_test_loss.item()
                avg_test_pos_loss += test_pos_loss.item()
                avg_test_ori_loss += test_ori_loss.item()

            all_test_losses.append(avg_test_loss / num_test_batches)
            all_test_pos_losses.append(avg_test_pos_loss / num_test_batches)
            all_test_ori_losses.append(avg_test_ori_loss / num_test_batches)

        if epoch == 0:
            # Optionally save copy of all relevant scripts/files for reproducibility
            os.mkdir(saved_root)
            shutil.copy("model.py", os.path.join(saved_root, "model.py"))
            shutil.copy("data_params.py", os.path.join(
                saved_root, "data_params.py"))
            shutil.copy("train.py", os.path.join(saved_root, "train.py"))

            # Save train args
            with open(os.path.join(saved_root, f"{args_name}.json"), "w") as outfile:
                json.dump(vars(train_args), outfile, indent=4)

        if epoch % epochs_per_save == 0:
            torch.save(model.state_dict(), os.path.join(
                saved_root, "model_%d.h5" % epoch))

        if epoch % epochs_per_test == 0:
            # Plot losses
            print("train: %.4f, test: %.4f" %
                  (all_train_losses[-1], all_test_losses[-1]))
            test_timesteps = np.arange(
                0, len(all_test_losses)) * epochs_per_test
            plt.plot(all_train_losses, label="train loss")
            plt.plot(all_train_pos_losses, label="train pos loss")
            plt.plot(all_train_ori_losses, label="train theta loss")
            plt.plot(test_timesteps, all_test_losses, label="test loss")
            plt.plot(test_timesteps, all_test_pos_losses,
                     label="test pos loss")
            plt.plot(test_timesteps, all_test_ori_losses,
                     label="test theta loss")
            plt.legend()
            plt.savefig(os.path.join(saved_root, loss_name + ".png"))
            plt.clf()

            # Print learned rotational offset features vs true rotational offsets
            # for [ignored, care about, start, goal] objects
            if train_rot:
                if rot_dim == 2:
                    cur_offsets = (model.rot_offsets_train.detach(
                    ).cpu().numpy().flatten() * 180 / np.pi) % 360
                    actual_offsets = np.rad2deg(
                        Params.ori_offsets_2D).astype(int).tolist()
                    print(f"Learned ({cur_offsets.astype(int)}), "
                          f"actual ({actual_offsets}) theta offsets ")
                else:
                    cur_offsets = model.rot_offsets_train.detach().cpu().numpy()
                    cur_offsets = cur_offsets / \
                        np.linalg.norm(cur_offsets, axis=-1, keepdims=True)
                    actual_offsets = Params.ori_offsets_3D
                    print(f"Learned ({cur_offsets})\n"
                          f"actual ({actual_offsets}) quaternion offsets ")

    # Save final model
    print("train: %.4f, test: %.4f" %
          (all_train_losses[-1], all_test_losses[-1]))
    torch.save(model.state_dict(), os.path.join(
        saved_root, "model_%d.h5" % (epoch + 1)))

    test_timesteps = np.arange(0, len(all_test_losses)) * epochs_per_test
    plt.plot(all_train_losses, label="train loss")
    plt.plot(all_train_pos_losses, label="train pos loss")
    plt.plot(all_train_ori_losses, label="train theta loss")
    plt.plot(test_timesteps, all_test_losses, label="test loss")
    plt.plot(test_timesteps, all_test_pos_losses, label="test pos loss")
    plt.plot(test_timesteps, all_test_ori_losses, label="test theta loss")
    plt.legend()
    plt.savefig(os.path.join(saved_root, loss_name + ".png"))

    np.savez(os.path.join(saved_root, loss_name), train=all_train_losses,
             train_pos=all_train_pos_losses, train_theta=all_train_ori_losses,
             test=all_test_losses, test_pos=all_test_pos_losses, test_theta=all_test_ori_losses)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store',
                        type=str, help="trained model name")
    parser.add_argument('--data_name', action='store',
                        type=str, help="data folder name")
    parser.add_argument('--loaded_epoch', action='store', type=int,
                        default=0, help="training epoch of pretrained model to load from")
    parser.add_argument('--is_rot', action='store_true',
                        help="whether to train rotation network")
    parser.add_argument('--is_3D', action='store_true',
                        help="whether to use 3D data")

    # Default hyperparameters used for paper experiments, no need to tune to reproduce
    parser.add_argument('--lr', action='store', type=float,
                        default=1e-3, help="learning rate")
    parser.add_argument('--batch_size', action='store', type=int, default=16)
    parser.add_argument('--n_samples', action='store', type=int, default=32,
                        help="number of wpt samples to draw from each trajectory sample")
    parser.add_argument('--hidden_dim', action='store', type=int, default=64)
    parser.add_argument('--pos_preference_dim',
                        action='store', type=int, default=1)
    parser.add_argument('--rot_preference_dim',
                        action='store', type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    train_args = parse_arguments()
    if train_args.is_3D:
        print("Training 3D!")
    else:
        print("Training 2D!")
    if train_args.is_rot:
        print("Training rotation!")
    else:
        print("Training position!")

    if not os.path.exists(Params.model_root):
        os.makedirs(Params.model_root)

    if train_args.is_3D:
        pos_dim = 3
        rot_dim = 6  # R * [x-axis, y-axis]
    else:
        pos_dim = 2
        rot_dim = 2  # cos(theta), sin(theta)

    # Define model
    model = PolicyNetwork(n_objects=Params.n_train_objects, pos_dim=pos_dim, rot_dim=rot_dim,
                          pos_preference_dim=train_args.pos_preference_dim,
                          rot_preference_dim=train_args.rot_preference_dim,
                          hidden_dim=train_args.hidden_dim,
                          device=DEVICE).to(DEVICE)

    # initialize features specifically such that attract feature close to goal feature
    # to allow for interpolation after training
    model.pos_pref_feat_train[Params.REPEL_IDX].data.fill_(1.0)
    model.pos_pref_feat_train[Params.ATTRACT_IDX].data.fill_(0.3)
    model.pos_pref_feat_train[Params.GOAL_IDX].data.fill_(0.0)

    # model.rot_pref_feat_train[Params.IGNORE_ROT_IDX].data.fill_(-1.0)
    # model.rot_pref_feat_train[Params.CARE_ROT_IDX].data.fill_(1.0)
    # model.rot_pref_feat_train[Params.START_IDX].data.fill_(1.0)
    # model.rot_pref_feat_train[Params.GOAL_IDX].data.fill_(1.0)

    model.rot_pref_feat_train[Params.IGNORE_ROT_IDX].data.fill_(5.0)
    model.rot_pref_feat_train[Params.CARE_ROT_IDX].data.fill_(-5.0)
    model.rot_pref_feat_train[Params.START_IDX].data.fill_(-10.0)
    model.rot_pref_feat_train[Params.GOAL_IDX].data.fill_(3.0)

    # Load pretrained model
    # NOTE: This is necessary when training position and rotation networks
    #  separately in two phases, specifically phase 2
    model_name = train_args.model_name
    loaded_epoch = train_args.loaded_epoch
    if loaded_epoch > 0:
        model_file_name = f"{model_name}/model_{loaded_epoch}.h5"
        print("Loading %s" % model_file_name)
        model.load_state_dict(torch.load(
            os.path.join(Params.model_root, model_file_name)))

    # Run training
    train(model, saved_root=os.path.join(Params.model_root, model_name),
          train_args=train_args,
          loaded_epoch=loaded_epoch)
