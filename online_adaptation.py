from typing import *
import numpy as np
from tqdm import tqdm

import torch

from data_params import Params
from model import Policy, PolicyNetwork, encode_ori_3D, encode_ori_2D, pose_to_model_input
from recursive_least_squares.rls import RLS
from train import process_batch_data, DEVICE


def write_log(log_file, string):
    print(string)
    if log_file is not None:
        log_file.write(string+'\n')
        log_file.flush()


def params2str(params):
    params = np.array([float(v) for v in params])
    return np.array2string(params, precision=4, separator=', ')


def model_rollout(goal_tensors, current_inputs, start_tensors, goal_rot_inputs,
                  object_inputs, obj_idx_tensors,
                  intervention_traj,
                  model: PolicyNetwork,
                  agent_radii, out_T, goal_pos_radius_scale, dstep,
                  train_pos, train_rot, pos_dim):
    """
    Rollout model starting from current tensor for out_T timesteps. Recursively
    modifies current pose with predicted action to preserve gradient pipeline.
    NOTE: Assumes future object poses are known! Either objects are static,
        or their trajectories are assumed predictable.

    :param goal_tensors: goal poses  (B x pose_dim)
    :param current_inputs: current poses (B x 1 x pose_dim)
    :param start_tensors: start poses (B x pose_dim)
    :param goal_rot_inputs: goal rotation inputs (B x pose_dim)
    :param object_inputs: object poses (B x T x n_objs x pose_dim+1)
    :param obj_idx_tensors: object indices/types (B x n_objs)
    :param intervention_traj: human intervention/expert trajectory (B x T x pose_dim)
    :param model: model/network to rollout
    :param agent_radii: agent radii (B x 1)
    :param out_T: number of timesteps to rollout
    :param goal_pos_radius_scale: scale for goal position radius
    :param dstep: distance stepsize for executing predicted translation direction
    :param train_pos: whether to train position
    :param train_rot: whether to train rotation
    :return: predicted trajectory as tensor (B x out_T x pos_dim+ori_dim)
    """
    pred_traj = []
    rot_dim = 6 if pos_dim == 3 else 2
    for k in range(out_T):
        goal_radii = goal_pos_radius_scale * torch.norm(goal_tensors[:, :pos_dim] - current_inputs[:, 0, :pos_dim],
                                                        dim=-1).unsqueeze(-1)
        goal_inputs = torch.cat(
            [goal_tensors, goal_radii], dim=-1).unsqueeze(1)

        start_rot_radii = torch.norm(start_tensors[:, :pos_dim] - current_inputs[:, 0, :pos_dim],
                                     dim=-1).unsqueeze(-1)
        start_rot_inputs = torch.cat(
            [start_tensors, start_rot_radii], dim=-1).unsqueeze(1)

        pred_vec, pred_ori, other_vecs = model(current=current_inputs,
                                               start=start_rot_inputs,
                                               goal=goal_inputs, goal_rot=goal_rot_inputs,
                                               objects=object_inputs[:,
                                                                     k, :, :],
                                               object_indices=obj_idx_tensors,
                                               calc_rot=train_rot,
                                               calc_pos=train_pos,
                                               is_training=False)

        current_inputs = current_inputs.clone()
        if train_pos:
            # NOTE: gradient still flows to earlier timesteps
            current_inputs[:, 0, :pos_dim] = current_inputs[:, 0,
                                                            :pos_dim] + pred_vec * dstep
        else:
            # use human intervention position trajectory
            try:
                current_inputs[:, 0, :pos_dim] = intervention_traj[:,
                                                                   k, :pos_dim]
            except:
                break

        if train_rot:
            # B x 4 so copy over the ori vec
            current_inputs[:, 0, pos_dim:pos_dim + rot_dim] = pred_ori

        # NOTE: does torch.clone keep gradient? YES
        # B x 1 x pose_dim
        pred_traj.append(current_inputs[:, :, :-1])

    pred_traj = torch.cat(pred_traj, dim=1)  # B x T x pose_dim
    return pred_traj


def adaptation_loss(model: PolicyNetwork, batch_data_processed, dstep,
                    train_pos=True, train_rot=False,
                    goal_pos_radius_scale=1.0):
    """
    Calculates loss for online adaptation to human intervention.
    Unlike batch_inner_loop, which randomly samples timesteps from trajectory,
    this function uses ALL timesteps in trajectory. Also, this function
    computes loss by rolling out agent prediction recursively for multiple timesteps.
    This is important because gradients from the final timestep's loss still
    flow back to earlier timesteps as well.

    :param model: Model to train
    :param batch_data: Batch of data to adapt to
    :param dstep: distance to translate by each timestep using agent's predicted direction vector
    :param train_pos: whether to train position
           NOTE: both position and rotation can be trained simultaneously for adaptation
    :param train_rot: whether to train rotation
    :param goal_pos_radius_scale: rather than define goal radius as exactly
                the distance to agent, scale down by some factor
    :return: overall loss (torch.Tensor), rollout trajectory (B x T x pos_dim+rot_dim)(torch.Tensor)
    """
    (start_tensors, current_inputs, goal_tensors, goal_rot_inputs, object_inputs, obj_idx_tensors, _, _,
     traj_tensors) = batch_data_processed
    B, T = traj_tensors.shape[0:2]
    input_dim = traj_tensors.shape[-1]
    pos_dim = 3 if input_dim == 9 else 2
    out_T = T - 1
    agent_radii = torch.tensor(
        [Params.agent_radius], device=DEVICE).view(1, 1).repeat(B, 1)

    # Perform agent prediction rollout
    pred_traj = model_rollout(goal_tensors=goal_tensors, current_inputs=current_inputs,
                              start_tensors=start_tensors, goal_rot_inputs=goal_rot_inputs,
                              object_inputs=object_inputs, obj_idx_tensors=obj_idx_tensors,
                              intervention_traj=traj_tensors,
                              model=model,
                              agent_radii=agent_radii, out_T=out_T,
                              goal_pos_radius_scale=goal_pos_radius_scale, dstep=dstep,
                              train_pos=train_pos, train_rot=train_rot, pos_dim=pos_dim)

    # Compute loss
    loss = torch.tensor([0.0], device=DEVICE)
    if train_pos:
        # Upon post-analysis, we get more stable adaptation when using a different
        # loss function at adaptation time for position:
        # Calculate distance with each object for both human intervention and agent rollout
        # and measure difference in object distances
        gt_dists = torch.norm(traj_tensors[:, 1:, :pos_dim] -
                              object_inputs[:, 1:, 0, :pos_dim], dim=-1, keepdim=True)
        pred_dists = torch.norm(pred_traj[:, :, :pos_dim] -
                                object_inputs[:, 1:, 0, :pos_dim], dim=-1, keepdim=True)
        pos_loss = torch.abs(gt_dists - pred_dists).mean()
        loss = loss + pos_loss

    if train_rot:
        output_ori = pred_traj[:, :out_T, pos_dim:]
        target_ori = traj_tensors[:, :out_T, pos_dim:]
        if pos_dim == 2:
            theta_loss = (1 - torch.einsum("bik,bik->bi",
                          output_ori, target_ori)).mean()
        else:
            output_x = output_ori[:, :, 0:3]
            output_y = output_ori[:, :, 3:6]
            tgt_x = target_ori[:, :, 0:3]
            tgt_y = target_ori[:, :, 3:6]
            theta_loss = ((1 - torch.einsum("bik,bik->bi",
                          output_x, tgt_x)).mean() +
                          (1 - torch.einsum("bik,bik->bi",
                                            output_y, tgt_y)).mean())

        loss = loss + theta_loss

    return loss, pred_traj


def perform_adaptation(policy: Policy, batch_data: List[Tuple],
                       train_pos: bool, train_rot: bool,
                       n_adapt_iters: int, dstep: float,
                       verbose=False, clip_params=True,
                       ret_trajs=False,
                       Optimizer=torch.optim.Adam, lr=1e-1,
                       log_file=None):
    """
    Adapts the policy to the batch of human intervention data.
    :param policy: Policy to adapt
    :param batch_data: List of tuples of (human intervention data, agent rollout data)
    :param train_pos: Whether to train the policy on position
    :param train_rot: Whether to train the policy on rotation
    :param n_adapt_iters: Number of iterations to perform adaptation
    :param dstep: Step size for gradient descent
    :param verbose: Whether to print out adaptation progress
    :param clip_params: Whether to clip the policy parameters
    :return:
        losses: List of losses for each adaptation iteration
        pred_traj: Predicted trajectory for final adaptation iteration
    """
    batch_data_processed = process_batch_data(
        batch_data, train_rot=None, n_samples=None, is_full_traj=True)

    # Only adapt parameters that aren't frozen
    adaptable_parameters = []
    for p in policy.adaptable_parameters:
        if p.requires_grad:
            adaptable_parameters.append(p)
    if verbose:
        write_log(
            log_file, f"Actual params: {params2str(adaptable_parameters)}")

    optimizer = Optimizer(adaptable_parameters, lr=lr)
    losses = []
    pred_trajs = []
    if n_adapt_iters == 0:
        # Don't perform any update, but return loss and predicted trajectory
        with torch.no_grad():
            loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                              batch_data_processed=batch_data_processed,
                                              train_pos=train_pos, train_rot=train_rot,
                                              dstep=dstep)
        return [loss.item()], [pred_traj.detach().cpu().numpy()]

    for iteration in (range(n_adapt_iters)):
        loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                          batch_data_processed=batch_data_processed,
                                          train_pos=train_pos, train_rot=train_rot,
                                          dstep=dstep)
        optimizer.zero_grad()
        loss.backward()
        if verbose:
            old_params = [p.clone() for p in adaptable_parameters]
            gradients = [p.grad.clone() for p in adaptable_parameters]
        optimizer.step()

        losses.append(loss.item())
        if ret_trajs:
            pred_trajs.append(pred_traj.detach().cpu().numpy())

        if verbose:
            for old_p, new_p, grad in zip(old_params, adaptable_parameters, gradients):
                # NOTE: +update, not -update since new_p - old_p already accounts for the -1* direction
                update = new_p - old_p
                write_log(log_file, "Original Grad: %.3f, Pred Grad: %.3f, New P: %.3f" % (
                    -grad.item(), +update.item() / lr, new_p.item()))

            write_log(log_file, "iter %d loss: %.3f" %
                      (iteration, loss.item()))
            write_log(
                log_file, f"Actual params: {params2str(adaptable_parameters)}")

    # Clip learned features within expected range
    if clip_params:
        policy.clip_adaptable_parameters()

    return losses, pred_trajs


def perform_adaptation_rls(policy: Policy, rls: RLS, batch_data: List[Tuple],
                           train_pos: bool, train_rot: bool,
                           n_adapt_iters: int, dstep: float,
                           verbose=False, clip_params=True,
                           ret_trajs=False, log_file=None):
    """
    Recursive Least Squares Adaptation
    """
    batch_data_processed = process_batch_data(
        batch_data, train_rot=None, n_samples=None, is_full_traj=True)

    # Only adapt parameters that aren't frozen
    adaptable_parameters = []
    for p in policy.adaptable_parameters:
        if p.requires_grad:
            adaptable_parameters.append(p)
    if verbose:
        write_log(
            log_file, f"Actual params: {params2str(adaptable_parameters)}")

    losses = []
    pred_trajs = []
    if n_adapt_iters == 0:
        # Don't perform any update, but return loss and predicted trajectory
        with torch.no_grad():
            loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                              batch_data_processed=batch_data_processed,
                                              train_pos=train_pos, train_rot=train_rot,
                                              dstep=dstep)
        return [loss.item()], [pred_traj.detach().cpu().numpy()]

    # Unpack data
    (start_tensors, current_inputs, goal_tensors, goal_rot_inputs, object_inputs, obj_idx_tensors, _, _,
        traj_tensors) = process_batch_data(batch_data, train_rot=None, n_samples=None, is_full_traj=True)
    B, T = traj_tensors.shape[0:2]
    input_dim = traj_tensors.shape[2] - 1  # excluding radius
    pos_dim = 3 if input_dim == 9 else 2
    rot_dim = 6 if input_dim == 9 else 2
    out_T = T - 1
    agent_radii = torch.tensor(
        [Params.agent_radius], device=DEVICE).view(1, 1).repeat(B, 1)

    # run RLS for multiple iters? or just one iter?
    for iteration in range(n_adapt_iters):
        # Perform agent prediction rollout
        pred_traj = model_rollout(goal_tensors=goal_tensors, current_inputs=current_inputs,
                                  start_tensors=start_tensors, goal_rot_inputs=goal_rot_inputs,
                                  object_inputs=object_inputs, obj_idx_tensors=obj_idx_tensors,
                                  intervention_traj=traj_tensors,
                                  model=policy.policy_network,
                                  agent_radii=agent_radii, out_T=out_T,
                                  goal_pos_radius_scale=1.0, dstep=dstep,
                                  train_pos=train_pos, train_rot=train_rot, pos_dim=pos_dim)

        # Compute error for RLS (Least squares loss) (y - yhat)
        left = 0 if train_pos else pos_dim
        right = pos_dim + rot_dim if train_rot else pos_dim
        y = traj_tensors[:, 1:, left:right]
        yhat = pred_traj[:, :, left:right]

        # RLS is slow, so only calculate loss for subset of traj
        error = torch.norm(y - yhat, dim=2)[0]
        rand_indices = torch.argsort(error, descending=True)[:10]
        # rand_indices = torch.randint(low=0, high=out_T, size=(15,))
        y_sampled = y[:, rand_indices, :]
        yhat_sampled = yhat[:, rand_indices, :]
        # y_sampled = y
        # yhat_sampled = yhat

        rls.update(y=y_sampled, yhat=yhat_sampled,
                   thetas=adaptable_parameters, verbose=verbose, log_file=log_file, model=policy.policy_network)

        loss = torch.norm(y - yhat, dim=2).mean()
        losses.append(loss.item())
        if ret_trajs:
            pred_trajs.append(pred_traj.detach().cpu().numpy())

        if verbose:
            write_log(log_file, "iter %d loss: %.3f" %
                      (iteration, loss.item()))
            write_log(
                log_file, f"Actual params: {params2str(adaptable_parameters)}")

    # Clip learned features within expected range
    if clip_params:
        policy.clip_adaptable_parameters()

    return losses, pred_trajs


def perform_adaptation_learn2learn(policy: Policy, learned_opt, batch_data: List[Tuple],
                                   train_pos: bool, train_rot: bool,
                                   n_adapt_iters: int, dstep: float,
                                   verbose=False, clip_params=True,
                                   ret_trajs=False,
                                   reset_lstm=True, log_file=None):
    """
    learned_opt: LearnedOptimizer from learn2learn.py
    """
    batch_data_processed = process_batch_data(
        batch_data, train_rot=None, n_samples=None, is_full_traj=True)

    # Reset hidden states of learned_opt
    if reset_lstm:
        learned_opt.reset_lstm()

    # Only adapt parameters that aren't frozen
    params = [p for p in policy.adaptable_parameters if p.requires_grad]
    if verbose:
        write_log(log_file, f"Actual params: {params2str(params)}")
    losses = []
    pred_trajs = []
    if n_adapt_iters == 0:
        # Don't perform any update, but return loss and predicted trajectory
        with torch.no_grad():
            loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                              batch_data_processed=batch_data_processed,
                                              train_pos=train_pos, train_rot=train_rot,
                                              dstep=dstep)
        return [loss.item()], [pred_traj.detach().cpu().numpy()]

    for iteration in range(n_adapt_iters):
        loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                          batch_data_processed=batch_data_processed,
                                          train_pos=train_pos, train_rot=train_rot,
                                          dstep=dstep)

        is_final = iteration == n_adapt_iters - 1
        new_params, need_reset = learned_opt.step(
            loss, params, verbose=verbose, log_file=log_file, is_final=is_final)

        policy.update_obj_feats_with_grad(new_params, same_var=not need_reset)
        if need_reset:
            policy.policy_network.zero_grad()
            learned_opt.zero_grad()

        # NOTE: cannot just assign params = new_params because policy's network
        # is now using a copy of new_params, so need new reference/ptr to them
        params = [p for p in policy.adaptable_parameters if p.requires_grad]

        losses.append(loss.item())
        if ret_trajs:
            pred_trajs.append(pred_traj.detach().cpu().numpy())
        if verbose:
            write_log(log_file, "iter %d loss: %.3f" %
                      (iteration, loss.item()))
            write_log(log_file, f"Actual params: {params2str(params)}")

    # Clip learned features within expected range
    if clip_params:
        policy.clip_adaptable_parameters()

    return losses, pred_trajs


def perform_adaptation_learn2learn_group(policy: Policy, learned_opts, batch_data: List[Tuple],
                                         train_pos: bool, train_rot: bool,
                                         n_adapt_iters: int, dstep: float,
                                         verbose=False, clip_params=True,
                                         ret_trajs=False,
                                         reset_lstm=True, log_file=None):
    """
    learned_opt: LearnedOptimizer from learn2learn.py
    """
    batch_data_processed = process_batch_data(
        batch_data, train_rot=None, n_samples=None, is_full_traj=True)

    # Reset hidden states of learned_opt
    if reset_lstm:
        learned_opts.reset_lstm()

    # Only adapt parameters that aren't frozen
    params = []
    param_types = []
    for p, ptype in zip(policy.adaptable_parameters,
                        policy.adaptable_parameter_types):
        if p.requires_grad:
            params.append(p)
            param_types.append(ptype)

    losses = []
    pred_trajs = []
    if n_adapt_iters == 0:
        # Don't perform any update, but return loss and predicted trajectory
        with torch.no_grad():
            loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                              batch_data_processed=batch_data_processed,
                                              train_pos=train_pos, train_rot=train_rot,
                                              dstep=dstep)
        return [loss.item()], [pred_traj.detach().cpu().numpy()]

    for iteration in range(n_adapt_iters):
        loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                          batch_data_processed=batch_data_processed,
                                          train_pos=train_pos, train_rot=train_rot,
                                          dstep=dstep)

        new_params, need_reset = learned_opts.step(
            loss, params, verbose=verbose, param_types=param_types)

        policy.update_obj_feats_with_grad(new_params, same_var=not need_reset)
        if need_reset:
            policy.policy_network.zero_grad()

        # NOTE: cannot just assign params = new_params because policy's network
        # is now using a copy of new_params, so need new reference/ptr to them
        params = [p for p in policy.adaptable_parameters if p.requires_grad]

        losses.append(loss.item())
        if ret_trajs:
            pred_trajs.append(pred_traj.detach().cpu().numpy())

        if verbose:
            write_log(log_file, "iter %d loss: %.3f" %
                      (iteration, loss.item()))
            write_log(log_file, f"Actual params: {params}")

    # Clip learned features within expected range
    if clip_params:
        policy.clip_adaptable_parameters()

    return losses, pred_trajs
