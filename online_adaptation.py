from typing import *
import numpy as np
from tqdm import tqdm

import torch

from data_params import Params
from model import Policy, PolicyNetwork, encode_ori_3D, encode_ori_2D, pose_to_model_input
from recursive_least_squares.rls import RLS
from train import batch_inner_loop, process_batch_data, DEVICE


def write_log(log_file, string):
    print(string)
    if log_file is not None:
        log_file.write(string + '\n')
        log_file.flush()


def params2str(params):
    params = np.array([float(v) for v in params])
    return np.array2string(params, precision=4, separator=', ')


class Logger(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.losses = []
        self.pred_trajs = []
        self.iteration = 0

    def update(self, policy, batch_data_processed, train_pos, train_rot, dstep, params):
        ############# Log Analysis Data #############
        with torch.no_grad():
            loss_debug, pred_traj_debug = adaptation_loss(
                model=policy.policy_network,
                batch_data_processed=batch_data_processed,
                train_pos=train_pos, train_rot=train_rot,
                dstep=dstep)
        self.pred_trajs.append(pred_traj_debug.detach().cpu().numpy())
        write_log(self.log_file, "iter %d loss: %.3f" %
                  (self.iteration, loss_debug.item()))
        write_log(self.log_file, f"Actual params: {params2str(params)}")
        self.losses.append(loss_debug.item())
        self.iteration += 1


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
                    goal_pos_radius_scale=1.0, use_rollout=True, batch_data=None):
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
    is_3D = pos_dim == 3
    out_T = T - 1
    agent_radii = torch.tensor(
        [Params.agent_radius], device=DEVICE).view(1, 1).repeat(B, 1)

    # Perform agent prediction rollout
    if use_rollout:
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
            # gt_dists = torch.norm(traj_tensors[:, 1:, :pos_dim] -
            #                       object_inputs[:, 1:, 0, :pos_dim], dim=-1, keepdim=True)
            # pred_dists = torch.norm(pred_traj[:, :, :pos_dim] -
            #                         object_inputs[:, 1:, 0, :pos_dim], dim=-1, keepdim=True)
            # pos_loss = torch.abs(gt_dists - pred_dists).mean()
            pos_loss = torch.norm(traj_tensors[:, 1:, :pos_dim] -
                                  pred_traj[:, :, :pos_dim], dim=-1).mean()
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
    else:
        assert batch_data is not None
        pred_traj = None
        n_samples = 64
        pos_loss, rot_loss = batch_inner_loop(model, batch_data,
                                              is_3D=is_3D,
                                              train_rot=train_rot,
                                              n_samples=n_samples, is_training=False)
        loss = pos_loss + rot_loss  # 0 by default if train_rot/pos is False

    return loss, pred_traj


def perform_adaptation(policy: Policy, batch_data: List[Tuple],
                       train_pos: bool, train_rot: bool,
                       n_adapt_iters: int, dstep: float, is_3D,
                       verbose=False, clip_params=True,
                       ret_trajs=False,
                       Optimizer=torch.optim.Adam,
                       log_file=None,
                       optim_params={'lr': 0.1},
                       detached_steps=True):
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
    import ipdb
    ipdb.set_trace()
    batch_data_processed = process_batch_data(
        batch_data, train_rot=None, n_samples=None, is_full_traj=True)
    logger = Logger(log_file)

    # Only adapt parameters that aren't frozen
    params = []
    for p in policy.adaptable_parameters:
        if p.requires_grad:
            params.append(p)

    optimizer = Optimizer(params, **optim_params)
    if n_adapt_iters == 0:
        # Don't perform any update, but return loss and predicted trajectory
        with torch.no_grad():
            loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                              batch_data_processed=batch_data_processed,
                                              train_pos=train_pos, train_rot=train_rot,
                                              dstep=dstep)
        return [loss.item()], [pred_traj.detach().cpu().numpy()]

    for iteration in (range(n_adapt_iters)):
        logger.update(policy, batch_data_processed, train_pos=train_pos,
                      train_rot=train_rot, dstep=dstep, params=params)
        if detached_steps:
            n_samples = np.Inf  # use all timesteps of traj
            pos_loss, rot_loss = batch_inner_loop(model=policy.policy_network,
                                                  batch_data=batch_data, train_rot=train_rot, is_3D=is_3D, n_samples=n_samples, is_training=False)
            loss = pos_loss + rot_loss
        else:
            loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                              batch_data_processed=batch_data_processed,
                                              train_pos=train_pos, train_rot=train_rot,
                                              dstep=dstep)
        optimizer.zero_grad()
        loss.backward()
        if verbose:
            old_params = [p.clone() for p in params]
            gradients = [p.grad.clone() for p in params]
        optimizer.step()

        if verbose:
            for old_p, new_p, grad in zip(old_params, params, gradients):
                # NOTE: +update, not -update since new_p - old_p already accounts for the -1* direction
                update = new_p - old_p
                write_log(log_file, "-Original Grad: %.3f, -lr * Pred Grad:  %.3f, New P: %.3f" % (
                    -grad.item(), +update.item(), new_p.item()))

    # Clip learned features within expected range
    if clip_params:
        policy.clip_adaptable_parameters()

    return logger.losses, logger.pred_trajs


def perform_adaptation_rls(policy: Policy, rls: RLS, batch_data: List[Tuple],
                           train_pos: bool, train_rot: bool,
                           n_adapt_iters: int, dstep: float, is_3D,
                           verbose=False, clip_params=True,
                           ret_trajs=False, log_file=None, reset_rls=True,
                           detached_steps=True):
    """
    Recursive Least Squares Adaptation
    """
    pos_dim = 3 if is_3D else 2
    rot_dim = 6 if is_3D else 2
    batch_data_processed = process_batch_data(
        batch_data, train_rot=None, n_samples=None, is_full_traj=True)
    logger = Logger(log_file)

    if reset_rls:
        rls.reset()

    # Only adapt parameters that aren't frozen
    params = []
    for p in policy.adaptable_parameters:
        if p.requires_grad:
            params.append(p)

    if n_adapt_iters == 0:
        # Don't perform any update, but return loss and predicted trajectory
        with torch.no_grad():
            loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                              batch_data_processed=batch_data_processed,
                                              train_pos=train_pos, train_rot=train_rot,
                                              dstep=dstep)
        return [loss.item()], [pred_traj.detach().cpu().numpy()]

    # run RLS for multiple iters? or just one iter?
    for iteration in range(n_adapt_iters):
        logger.update(policy, batch_data_processed, train_pos=train_pos,
                      train_rot=train_rot, dstep=dstep, params=params)

        if detached_steps:
            (pred_trans, pred_ori), traj_tensors = batch_inner_loop(
                model=policy.policy_network,
                batch_data=batch_data, train_rot=train_rot, is_3D=is_3D, n_samples=np.Inf, is_training=False, ret_predictions=True)

            if train_rot:
                assert pred_trans is None
                pred_trans = torch.zeros(
                    (pred_ori.shape[0], pos_dim), device=DEVICE)
            else:
                assert pred_ori is None
                pred_ori = torch.zeros(
                    (pred_trans.shape[0], rot_dim), device=DEVICE)
            pred_traj = torch.cat([pred_trans, pred_ori], dim=-1).unsqueeze(0)
            traj_tensors = traj_tensors.unsqueeze(0)

        else:
            # Unpack data
            (start_tensors, current_inputs, goal_tensors, goal_rot_inputs, object_inputs, obj_idx_tensors, _, _,
             traj_tensors) = process_batch_data(batch_data, train_rot=None, n_samples=None, is_full_traj=True)
            traj_tensors = traj_tensors[:, 1:, ]
            B, out_T = traj_tensors.shape[0:2]
            agent_radii = torch.tensor(
                [Params.agent_radius], device=DEVICE).view(1, 1).repeat(B, 1)

            # Perform agent prediction rollout
            pred_traj = model_rollout(
                goal_tensors=goal_tensors, current_inputs=current_inputs,
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
        y = traj_tensors[:, :, left:right]
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
                   thetas=params, verbose=verbose, log_file=log_file, model=policy.policy_network)

    # Clip learned features within expected range
    if clip_params:
        policy.clip_adaptable_parameters()

    return logger.losses, logger.pred_trajs


def perform_adaptation_learn2learn(policy: Policy, learned_opt, batch_data: List[Tuple],
                                   train_pos: bool, train_rot: bool,
                                   n_adapt_iters: int, dstep: float,
                                   is_3D,
                                   verbose=False, clip_params=True,
                                   ret_trajs=False,
                                   reset_lstm=True, log_file=None,
                                   detached_steps=True):
    """
    learned_opt: LearnedOptimizer from learn2learn.py
    """
    batch_data_processed = process_batch_data(
        batch_data, train_rot=None, n_samples=None, is_full_traj=True)
    logger = Logger(log_file)

    # Reset hidden states of learned_opt
    if reset_lstm:
        learned_opt.reset_lstm()

    # Only adapt parameters that aren't frozen
    params = [p for p in policy.adaptable_parameters if p.requires_grad]
    if n_adapt_iters == 0:
        # Don't perform any update, but return loss and predicted trajectory
        with torch.no_grad():
            loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                              batch_data_processed=batch_data_processed,
                                              train_pos=train_pos, train_rot=train_rot,
                                              dstep=dstep)
        return [loss.item()], [pred_traj.detach().cpu().numpy()]

    for iteration in range(n_adapt_iters):
        logger.update(policy, batch_data_processed, train_pos=train_pos,
                      train_rot=train_rot, dstep=dstep, params=params)

        if detached_steps:
            n_samples = 64  # <= min(len(traj)) for easy stacking of batch data
            pos_loss, rot_loss = batch_inner_loop(model=policy.policy_network,
                                                  batch_data=batch_data, train_rot=train_rot, is_3D=is_3D, n_samples=n_samples, is_training=False)
            loss = pos_loss + rot_loss
        else:
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

    # Clip learned features within expected range
    if clip_params:
        policy.clip_adaptable_parameters()

    return logger.losses, logger.pred_trajs


def perform_adaptation_learn2learn_group(policy: Policy, learned_opts, batch_data: List[Tuple],
                                         train_pos: bool, train_rot: bool,
                                         n_adapt_iters: int, dstep: float, is_3D,
                                         verbose=False, clip_params=True,
                                         ret_trajs=False,
                                         reset_lstm=True, log_file=None,
                                         detached_steps=True):
    """
    learned_opt: LearnedOptimizer from learn2learn.py
    """
    batch_data_processed = process_batch_data(
        batch_data, train_rot=None, n_samples=None, is_full_traj=True)
    logger = Logger(log_file)

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

    if n_adapt_iters == 0:
        # Don't perform any update, but return loss and predicted trajectory
        with torch.no_grad():
            loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                              batch_data_processed=batch_data_processed,
                                              train_pos=train_pos, train_rot=train_rot,
                                              dstep=dstep)
        return [loss.item()], [pred_traj.detach().cpu().numpy()]

    for iteration in range(n_adapt_iters):
        logger.update(policy, batch_data_processed, train_pos=train_pos,
                      train_rot=train_rot, dstep=dstep, params=params)

        if detached_steps:
            n_samples = np.Inf
            pos_loss, rot_loss = batch_inner_loop(model=policy.policy_network,
                                                  batch_data=batch_data, train_rot=train_rot, is_3D=is_3D, n_samples=n_samples, is_training=False)
            loss = pos_loss + rot_loss
        else:
            loss, pred_traj = adaptation_loss(model=policy.policy_network,
                                              batch_data_processed=batch_data_processed,
                                              train_pos=train_pos, train_rot=train_rot,
                                              dstep=dstep)

        new_params, need_reset = learned_opts.step(
            loss, params, verbose=verbose, param_types=param_types, log_file=log_file)

        policy.update_obj_feats_with_grad(new_params, same_var=not need_reset)
        if need_reset:
            policy.policy_network.zero_grad()

        # NOTE: cannot just assign params = new_params because policy's network
        # is now using a copy of new_params, so need new reference/ptr to them
        params = [p for p in policy.adaptable_parameters if p.requires_grad]

    # Clip learned features within expected range
    if clip_params:
        policy.clip_adaptable_parameters()

    return logger.losses, logger.pred_trajs
