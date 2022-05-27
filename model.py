"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from typing import *
import numpy as np

from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn

from data_params import Params


def quat2mat(Q: torch.Tensor):
    """
    Convert quaternion to rotation matrix.

    :param Q: quaternion (B x 4)
    :return: rotation matrix (B x 3 x 3)
    """
    q1 = Q[:, 0:1]
    q2 = Q[:, 1:2]
    q3 = Q[:, 2:3]
    q0 = Q[:, 3:4]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    row1 = torch.hstack([r00, r01, r02]).unsqueeze(1)
    row2 = torch.hstack([r10, r11, r12]).unsqueeze(1)
    row3 = torch.hstack([r20, r21, r22]).unsqueeze(1)

    rot_matrix = torch.cat([row1, row2, row3], dim=1)

    return rot_matrix


def calc_dist_ratio(x1, x2, r1, r2):
    dists = (x1 - x2).norm(dim=-1)
    return dists / (r1 + r2)


def encode_ori_2D(theta_traj):
    """
    Converts theta -> [cos(theta), sin(theta)]

    :param theta_traj: theta (N x 1)
    :return: encoded orientation (N x 2)
    """
    if isinstance(theta_traj, np.ndarray):
        return np.hstack([np.cos(theta_traj)[:, np.newaxis],
                          np.sin(theta_traj)[:, np.newaxis]])
    else:
        return torch.hstack([torch.cos(theta_traj)[:, np.newaxis],
                             torch.sin(theta_traj)[:, np.newaxis]])


def encode_ori_3D(ori_traj):
    """
    Converts input quaternions [qx, qy, qz, qw] into [Rx, Ry]

    :param ori_traj: input quaternions (N x 6)
    :return:
    """
    assert isinstance(ori_traj, np.ndarray)
    R_mat_traj = R.from_quat(ori_traj).as_matrix()  # N x 3 x 3
    Rx = R_mat_traj[:, :, 0]
    Ry = R_mat_traj[:, :, 1]
    return np.hstack([Rx, Ry])


def decode_ori(out):
    """
    Converts model output orientation into form used in data generation:
        2D:
            [cos(theta), sin(theta)] -> [theta]
        3D:
            [Rx, Ry] -> [qw, qx, qy, qz]
    :param out: model output orientation
    :return: decoded orientation (N x (1 or 4))
    """
    # Convert to (N x ori_dim) numpy array
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
    ori_dim = out.shape[-1]
    out = out.reshape(-1, ori_dim)

    # Case on 2D or 3D ori and convert to
    if ori_dim == 2:  # [cos, sin]
        cos = out[:, 0]  # [cos, sin]
        sin = out[:, 1]
        theta = np.arctan2(sin, cos)
        return theta[:, np.newaxis]

    elif ori_dim == 6:
        x_axis = out[:, 0:3]
        y_axis = out[:, 3:6]
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        R_mat = np.concatenate([x_axis[:, :, np.newaxis],
                                y_axis[:, :, np.newaxis],
                                z_axis[:, :, np.newaxis]],
                               axis=-1)
        ori_quat = R.from_matrix(R_mat).as_quat()
        return ori_quat
    else:
        raise ValueError("Invalid orientation shape: {}, expected 2 or 6".format(ori_dim))


def pose_to_model_input(pose):
    """
    Converts pose to model input format.
    :param pose: input pose (N x pose_dim)
    :return: model input pose (N x model_pose_dim)
    """
    assert isinstance(pose, np.ndarray)  # Tensors not allowed! Need to use scipy.Rot!
    assert len(pose.shape) == 2  # Only two dimensions Batch x pose_dim
    if pose.shape[1] == 2 + 1:
        pos = pose[:, :2]
        theta = pose[:, -1]
        return np.concatenate([pos, encode_ori_2D(theta)], axis=-1)
    elif pose.shape[1] == 3 + 4:
        pos = pose[:, :3]
        ori_quat = pose[:, -4:]
        return np.concatenate([pos, encode_ori_3D(ori_quat)], axis=-1)
    else:
        raise ValueError("Invalid pose shape")


class Attention(nn.Module):
    """
    Compute attention across incoming edges/messages e_i from objects
    """

    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, edge_feats: torch.Tensor, dists: torch.Tensor):
        """
        Compute attention weights \alpha_i for each object edge e_i.

        :param edge_feats: (B x num_objects x input_size)
        :param dists: (B x num_objects x 1)
        :return:
            alpha: attention weights that sum to 1 across objects (B x num_objects x 1)
        """
        # Skip-like connection that concatenates relative distance values
        # with latent edge features before passing into layers
        alpha = self.layers(torch.cat([edge_feats, dists], dim=-1))
        return alpha


class RelationNetwork(nn.Module):
    """
    Generic Relation Network used to define the Position and Rotation Relation
    Networks.
    """

    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers):
        super().__init__()

        self.output_size = output_size

        # Interleaving Linear + ReLU layers
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

        self.attention = Attention(input_size=output_size + 1)

    def forward(self, edges: torch.Tensor, dists: torch.Tensor):
        """
        Extract latent edge features from input edges and compute attention
        across each.

        :param edges: (B x num_objects x input_size)
        :param dists: (B x num_objects x 1)
        :return:
            edges: processed edge features (B x num_objects x output_size)
            alpha: attention weights that sum to 1 across objects
        """
        edges = self.layers(edges)
        alpha = self.attention(edges, dists)
        return edges, alpha


class PolicyNetwork(nn.Module):
    def __init__(self, n_objects: int, pos_dim: int, rot_dim: int,
                 pos_preference_dim: int, rot_preference_dim: int,
                 hidden_dim: int, device: str):
        """
        Overall policy network that computes translation direction and desired
        orientation of agent/robot.

        :param n_objects: number of objects. used only during training
        :param pos_dim: dimension of position state x_{P,i}
        :param rot_dim: dimension of rotation state x_{R,i}
        :param pos_preference_dim: dimension of learned position preference features c_{P,i}
        :param rot_preference_dim: dimension of learned rotation relation features c_{R,i}
        :param hidden_dim: dimension of hidden layer outputs, same for position and rotation networks
        :param device: cpu or cuda
        """
        super().__init__()
        self.pos_dim = pos_dim
        self.rot_dim = rot_dim
        self.pos_idxs = list(range(pos_dim))
        self.rot_idxs = list(range(pos_dim, pos_dim + rot_dim))
        self.radius_idx = -1  # index of radius in input feature vectors 
        self.device = device

        # Learned preference features **during training**, which is why n_objects is known beforehand
        # +1 objects for goal
        self.pos_pref_feat_train = nn.Parameter(
            torch.randn((n_objects + 1, pos_preference_dim), device=device, dtype=torch.float32),
            requires_grad=True)
        # +2 objects for start and goal
        self.rot_pref_feat_train = nn.Parameter(
            torch.randn((n_objects + 2, rot_preference_dim), device=device, dtype=torch.float32),
            requires_grad=True)
        self.rot_offsets_train = None  # learned rotational offset, initalized below
        self.apply_rot_offset = None  # function to apply rot offset, initalized below
        self.init_rot_offsets(n_objects + 2, device)

        # Test-time preference features separated into lists so gradients
        # can be computed separately
        self.pos_pref_feat_test = []
        self.rot_pref_feat_test = []
        self.rot_offsets_test = []

        # Define position and rotation networks
        pos_input_feat_size = pos_dim + pos_preference_dim + 2  # +2 for sign_wrt_goal and relative distance
        self.pos_relation_network = RelationNetwork(input_size=pos_input_feat_size,
                                                    output_size=pos_dim,
                                                    hidden_size=hidden_dim,
                                                    n_hidden_layers=3)
        rot_input_feat_size = rot_dim + rot_preference_dim + 1  # +1 for relative distance
        # NOTE: could try including sign_wrt_goal as extra input, possibly better
        self.rot_relation_network = RelationNetwork(input_size=rot_input_feat_size,
                                                    output_size=rot_dim,
                                                    hidden_size=hidden_dim,
                                                    n_hidden_layers=3)

    def init_rot_offsets(self, n_objects, device):
        """
        Based on dimension of rotation (2 for 2D or 6 for 3D), perform the following:
            1. initialize learned rotation offsets c_{R,i}^\Delta for each training object
            2. initialize self.apply_rot_offset to be a function that applies the learned offsets
            3. verify self.rot_idxs are correct

        :param n_objects: number of objects (assumes includes start and goal)
        :param device:
        """
        if self.rot_dim == 2:
            # learned offset is stored and applied as 1D \theta offset
            self.rot_offsets_train = nn.Parameter(
                torch.zeros((n_objects, 1), device=device, dtype=torch.float32),
                requires_grad=True)

            # use 2D function
            self.apply_rot_offset = self.apply_rot_offset_2D

            # [pos_x, pos_y, rot_cos, rot_sin]
            assert self.rot_idxs == [2, 3]

        elif self.rot_dim == 6:  # Rx(3), Ry(3)
            # learned offset is stored and applied as 4D quaternion
            zero_rot = torch.tensor([0., 0., 0., 1.], device=self.device, dtype=torch.float32)
            self.rot_offsets_train = nn.Parameter(
                zero_rot.unsqueeze(0).repeat(n_objects, 1), requires_grad=True)

            # use 3D function
            self.apply_rot_offset = self.apply_rot_offset_3D

            # [pos_x, pos_y, pos_z, Rx(3), Ry(3)]
            assert self.rot_idxs == [3, 4, 5, 6, 7, 8]

        else:
            raise ValueError("Unexpected rotation dimensionality: %d" % self.rot_dim)

    def apply_rot_offset_2D(self, input_poses, rot_offset) -> torch.Tensor:
        """
        Apply learned rotation offset to input object orientations.
        For each object i:
            1. convert input <cos_i, sin_i> to \theta_i
            2. add learned rotation offset: \theta_i + c_{R,i}^\Delta
            3. convert back to <cos_i, sin_i>

        :param input_poses: [position, orientation(cos, sin)] (B x n_objects x 4)
        :param rot_offset: learned rotation offset (B x n_objects x 1)
        :return: new_orientations: [cos, sin] (B x n_objects x 2)
        """
        input_thetas = torch.atan2(input_poses[:, :, self.rot_idxs[1]],  # sin
                                   input_poses[:, :, self.rot_idxs[0]]  # cos
                                   ).unsqueeze(-1)
        new_thetas = input_thetas + rot_offset

        return torch.cat([torch.cos(new_thetas), torch.sin(new_thetas)], dim=-1)

    def apply_rot_offset_3D(self, input_poses, rot_offset) -> torch.Tensor:
        """
        Apply learned rotation offset to input object orientations.
        For each object i:
            1. convert input Rx, Ry to full 3x3 rotation matrix R_i
            2. convert learned rotation offset to 3x3 rotation matrix R_i^\Delta
            3. compute new orientation: R_i * R_i^\Delta
                NOTE: ORDER OF MULTIPLICATION IS IMPORTANT!
                    right-multiply by R_i^\Delta to apply learned offset
                    ***relative*** to object's current orientation
            4. convert new orientation back to Rx, Ry

        :param input_poses: [position, orientation(Rx, Ry)] (B x n_objects x 3 + 6)
        :param rot_offset: learned rotation offset (B x n_objects x 4)
        :return: new_orientations: [Rx, Ry] (B x n_objects x 6)

        """
        # Form full 3x3 rotation matrix from input orientations
        Rx = input_poses[:, :, self.rot_idxs[:3]].view(-1, 3)
        Ry = input_poses[:, :, self.rot_idxs[3:]].view(-1, 3)
        Rz = torch.cross(Rx, Ry, dim=1)
        orig_rot_mats = torch.stack([Rx, Ry, Rz], dim=-1)

        # Normalize learned rotation offset quaternions
        # and convert to rotation matrices
        B, n_objects, q_shape = rot_offset.shape
        q_offsets = (rot_offset / rot_offset.norm(dim=-1, keepdim=True)).view(B * n_objects, 4)
        rot_offset_mats = quat2mat(q_offsets)

        # Apply learned rotation offset to input object orientations
        new_rot_mats = torch.bmm(orig_rot_mats, rot_offset_mats)

        # Convert new orientation matrices back to Rx, Ry
        new_Rx = new_rot_mats[:, :, 0].view(B, n_objects, 3)
        new_Ry = new_rot_mats[:, :, 1].view(B, n_objects, 3)

        return torch.cat([new_Rx, new_Ry], dim=-1)

    def calc_rot_vec(self, start: torch.Tensor, goal_rot: torch.Tensor,
                     objects: torch.Tensor, object_indices: torch.Tensor,
                     pos_dist_ratios: torch.Tensor,
                     is_training: bool) -> torch.Tensor:
        """
        Calculate rotation action of policy with the following steps:
            1. Select the correct learned rotation preference features and offsets given object indices
            2. Apply learned rotation offset to object orientations
            3. Feed into Rotation Relation Network to get contribution from each object
            4. Aggregate and normalize (to produce valid orientation)

        Rotation action (desired orientation) is computed without knowledge of the agent/robot's
        current orientation. This means the rotation action can be infeasible
        to complete within a single timestep given bounds on max rotational vel/acc.
        This output should be filtered/bounded on real hardware system.

        Unlike Position Network, we may want agent's orientation to be fixed
        throughout trajectory (ex: holding cup upright). This is achieved by
        including start orientation as an extra "object".

        :param start: start pose (B x 1 x pose_dim+1)
        :param goal_rot: goal pose (B x 1 x pose_dim+1)
        :param objects: actual object poses (B x n_objects x pose_dim+1)
        :param object_indices: object indices NOT including start/goal (B x n_objects)
        :param pos_dist_ratios: size-relative distance to each object (B x n_objects + 2)
        :param is_training: whether to use training or testing object features
        :return: rot_vec: (B x rot_dim)
        """
        # Append start and goal to input object poses and indices
        B = objects.shape[0]
        goal_rot_feat_idx = torch.full(size=(B, 1), fill_value=Params.GOAL_IDX, device=self.device)
        start_rot_feat_idx = torch.full(size=(B, 1), fill_value=Params.START_IDX, device=self.device)
        feat_idxs = torch.cat([object_indices, start_rot_feat_idx, goal_rot_feat_idx], dim=1)
        input_poses = torch.cat([objects, start, goal_rot], dim=1)

        # Select correct learned rotation preference features and offsets
        if is_training:
            rot_pref_feat_rep = self.rot_pref_feat_train[feat_idxs]
            rot_offsets_rep = self.rot_offsets_train[feat_idxs]
        else:
            rot_pref_feat_rep = torch.cat([
                torch.stack([self.rot_pref_feat_test[obj_i] for obj_i in feat_idxs[b]]).unsqueeze(0)
                for b in range(B)
            ], dim=0)
            rot_offsets_rep = torch.cat([
                torch.stack([self.rot_offsets_test[obj_i] for obj_i in feat_idxs[b]]).unsqueeze(0)
                for b in range(B)
            ], dim=0)

        # Apply learned rotation offset to object orientations
        input_rot_effects = self.apply_rot_offset(input_poses=input_poses, rot_offset=rot_offsets_rep)

        # Feed into Rotation Relation Network to get contribution from each object
        edges, rot_alphas = self.rot_relation_network(
            edges=torch.cat([input_rot_effects,
                             pos_dist_ratios.unsqueeze(-1),
                             rot_pref_feat_rep], 2),
            dists=pos_dist_ratios.unsqueeze(-1)  # B x num_relations x 1
        )

        # Aggregate
        # Weighted sum of 3D rotations isn't exactly "correct", but this worked well
        rot_effect_receivers = (rot_alphas * input_rot_effects).sum(dim=1)

        # Normalize
        if self.rot_dim == 2:
            # produce valid unit vector <cos, sin>
            predicted_rot_vec = rot_effect_receivers / torch.norm(rot_effect_receivers, dim=-1, keepdim=True)
        else:
            Rx = rot_effect_receivers[:, 0:3]
            Rx = Rx / torch.norm(Rx, dim=-1, keepdim=True)

            # Gram Schmidt orthogonalization to ensure Rx and Ry orthogonal
            Ry = rot_effect_receivers[:, 3:6]
            Ry = (Ry - torch.einsum("bi,bi->b", Rx, Ry).unsqueeze(1) * Rx)
            Ry = Ry / torch.norm(Ry, dim=-1, keepdim=True)

            predicted_rot_vec = torch.hstack([Rx, Ry])

        return predicted_rot_vec

    def calc_pos_vec(self, current: torch.Tensor, goal: torch.Tensor,
                     objects: torch.Tensor, object_indices: torch.Tensor,
                     pos_dist_ratios: torch.Tensor, is_training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate position action of policy with the following steps:
            1. Select the correct learned position preference features given object indices
            2. Feed into Position Relation Network to get contribution from each object
            3. Aggregate and normalize

        :param current: current agent/robot pose (B x 1 x pose_dim+1)
        :param goal: goal pose (B x 1 x pose_dim+1)
        :param objects: actual object poses (B x n_objects x pose_dim+1)
        :param object_indices: object indices NOT including start/goal (B x n_objects)
        :param pos_dist_ratios: position distance ratios (B x n_objects + 1)
        :param is_training: whether to use training or testing object features
        :return:
        """
        # Append goal to input object poses and indices
        B, n_objects, obj_dim = objects.shape
        input_poses = torch.cat([objects, goal], dim=1)
        agent_pose = current.repeat(1, n_objects + 1, 1)  # +1 for goal
        goal_pos_feat_idx = torch.full(size=(B, 1), fill_value=Params.GOAL_IDX, device=self.device)
        feat_idxs = torch.cat([object_indices, goal_pos_feat_idx], dim=1)

        # Select correct learned position preference features and offsets
        if is_training:
            pos_pref_feat_rep = self.pos_pref_feat_train[feat_idxs]
        else:
            pos_pref_feat_rep = torch.cat([
                torch.stack([self.pos_pref_feat_test[obj_i] for obj_i in feat_idxs[b]]).unsqueeze(0)
                for b in range(B)
            ], dim=0)  # B x num_objects x learned dimension (c)

        # Compute feature: sign of direction to each object wrt direction to goal
        pos_input_vecs = input_poses[:, :, self.pos_idxs] - agent_pose[:, :, self.pos_idxs]
        pos_input_vecs = pos_input_vecs / torch.norm(pos_input_vecs, dim=-1, keepdim=True)
        # (B x K x 2) * (B x 2) = (B x K) dot-prod between all vecs and the goal-agent vecs
        sign_wrt_goal = torch.einsum('bki,bi->bk', pos_input_vecs[:, :], pos_input_vecs[:, -1])

        # Feed into Position Relation Network to get contribution from each object
        pos_effects, pos_alphas = self.pos_relation_network(
            edges=torch.cat([pos_input_vecs,
                             pos_dist_ratios.unsqueeze(-1),
                             sign_wrt_goal.unsqueeze(-1),
                             pos_pref_feat_rep], 2),
            dists=pos_dist_ratios.unsqueeze(-1)  # B x num_relations x 1
        )

        # Aggregate and normalize
        pos_effect_receivers = (pos_alphas * pos_effects).sum(dim=1)  # sum up weighted effects of all objects
        predicted_pos_vec = pos_effect_receivers / torch.norm(pos_effect_receivers, dim=-1, keepdim=True)

        return predicted_pos_vec, pos_effects

    def forward(self, current, start, goal, goal_rot, objects, object_indices, is_training,
                calc_pos=True, calc_rot=True):
        """
        Forward pass of policy.
            1. Calculate size-relative distances between each object/start/goal and the agent
            2. Calculate rotation action
            3. Calculate position action
            4. Combine actions and return

        :param current: current agent/robot pose (B x 1 x pose_dim+1)
        :param start: start pose (B x 1 x pose_dim+1)
        :param goal: goal pose (B x 1 x pose_dim+1)
        :param goal_rot: goal pose for rotation network (B x 1 x pose_dim+1)
                         NOTE: same as goal, but fixed radius
        :param objects: actual object poses (B x n_objects x pose_dim+1)
        :param object_indices: object indices NOT including start/goal (B x n_objects)
        :param calc_pos: whether to calculate position action
        :param calc_rot: whether to calculate rotation action
        :param is_training: whether to use training or testing object features
        :return:
        """
        B, n_objects, obj_dim = objects.shape
        assert object_indices.shape[1] == n_objects  # object_indices only includes objects
        assert current.shape[-1] == self.pos_dim + self.rot_dim + 1  # +1 for radius
        assert start.shape[-1] == self.pos_dim + self.rot_dim + 1
        assert goal.shape[-1] == self.pos_dim + self.rot_dim + 1
        assert goal_rot.shape[-1] == self.pos_dim + self.rot_dim + 1
        assert objects.shape[-1] == self.pos_dim + self.rot_dim + 1

        # Compute size-relative distances between each object/start/goal and the agent
        obj_dist_ratios = calc_dist_ratio(x1=objects[:, :, self.pos_idxs],
                                          x2=current[:, :, self.pos_idxs],
                                          r1=objects[:, :, self.radius_idx],
                                          r2=current[:, :, self.radius_idx])
        goal_dist_ratios = calc_dist_ratio(x1=goal[:, :, self.pos_idxs],
                                           x2=current[:, :, self.pos_idxs],
                                           r1=goal[:, :, self.radius_idx],
                                           r2=current[:, :, self.radius_idx])
        goal_rot_dist_ratios = calc_dist_ratio(x1=goal_rot[:, :, self.pos_idxs],
                                               x2=current[:, :, self.pos_idxs],
                                               r1=goal_rot[:, :, self.radius_idx],
                                               r2=current[:, :, self.radius_idx])
        start_dist_ratios = calc_dist_ratio(x1=start[:, :, self.pos_idxs],
                                            x2=current[:, :, self.pos_idxs],
                                            r1=start[:, :, self.radius_idx],
                                            r2=current[:, :, self.radius_idx])

        predicted_pos_vec = pos_effects = None
        predicted_rot_vec = None

        # Compute position and/or rotation actions
        if calc_pos:
            pos_dist_ratios = torch.cat([obj_dist_ratios, goal_dist_ratios], dim=1)
            predicted_pos_vec, pos_effects = self.calc_pos_vec(
                current, goal, objects, object_indices, pos_dist_ratios, is_training)

        if calc_rot:
            pos_dist_ratios = torch.cat([obj_dist_ratios, start_dist_ratios, goal_rot_dist_ratios], dim=1)
            predicted_rot_vec = self.calc_rot_vec(
                start, goal_rot, objects, object_indices, pos_dist_ratios, is_training)

        return predicted_pos_vec, predicted_rot_vec, pos_effects


class Policy(object):
    def __init__(self, policy_network: PolicyNetwork):
        # NOTE: not necessary to set "requires_grad=False" for all core network weights
        #   because perform_adaptation() in train.py only defines Adam optimizer
        #   using params from self.adaptable_parameters that have requires_grad=True
        self.policy_network = policy_network

        pos_repel_feat = self.policy_network.pos_pref_feat_train[Params.REPEL_IDX].detach()
        pos_attract_feat = self.policy_network.pos_pref_feat_train[Params.ATTRACT_IDX].detach()
        rot_ignore_feat = self.policy_network.rot_pref_feat_train[Params.IGNORE_ROT_IDX].detach()
        rot_care_feat = self.policy_network.rot_pref_feat_train[Params.CARE_ROT_IDX].detach()

        a = 0.35  # TODO: Tune this after training using viz_3D.py
        self.pos_ignore_feat = (a * pos_repel_feat + (1 - a) * pos_attract_feat)

        # self.pos_ignore_feat = self.policy_network.pos_pref_feat_train[Params.GOAL_IDX].detach()

        b = 0.75  # TODO: Tune this after training using viz_3D.py
        self.rot_ignore_feat = (b * rot_ignore_feat + (1 - b) * rot_care_feat)

        self.obj_pos_feats = []
        self.obj_rot_feats = []
        self.obj_rot_offsets = []
        self.adaptable_parameters = []

        # Optional: to avoid unexpected behavior, clamp min/max of features 
        # based on the extrema of object behavior
        # NOTE: due to training initialization, attract feat > repel feat, not applicable for dims > 1
        #   and ignore_rot < care_rot
        self.pos_pref_feat_bounds = (pos_repel_feat, pos_attract_feat)
        self.rot_pref_feat_bounds = (rot_ignore_feat, rot_care_feat)

    def __call__(self, is_training=False, *args, **kwargs):
        # NOTE: set is_training=False by default since Policy() class used for test-time typically
        return self.policy_network(is_training=is_training, *args, **kwargs)

    def init_new_objs(self, pos_obj_types: List[Union[int, None]], rot_obj_types: List[Union[int, None]],
                      pos_requires_grad=None, rot_requires_grad=None):
        pos_repel_feat = self.policy_network.pos_pref_feat_train[Params.REPEL_IDX].detach()
        pos_attract_feat = self.policy_network.pos_pref_feat_train[Params.ATTRACT_IDX].detach()
        rot_care_feat = self.policy_network.rot_pref_feat_train[Params.CARE_ROT_IDX].detach()

        # By default, make requires_grad=True for all new objects unless specified
        n_objects = len(pos_obj_types)
        pos_requires_grad = [True] * n_objects if pos_requires_grad is None else pos_requires_grad
        rot_requires_grad = [True] * n_objects if rot_requires_grad is None else rot_requires_grad

        # Pos/Ori features to either specific value or "ignore" by default
        self.obj_pos_feats = []
        self.obj_rot_feats = []
        self.obj_rot_offsets = []
        for i in range(n_objects):
            # Position
            if pos_obj_types[i] == Params.REPEL_IDX:
                self.obj_pos_feats.append(torch.clone(pos_repel_feat))
            elif pos_obj_types[i] == Params.ATTRACT_IDX:
                self.obj_pos_feats.append(torch.clone(pos_attract_feat))
            else:  # None
                self.obj_pos_feats.append(torch.clone(self.pos_ignore_feat))
            self.obj_pos_feats[-1].requires_grad = pos_requires_grad[i]

            # Rotation
            if rot_obj_types[i] == Params.CARE_ROT_IDX:
                self.obj_rot_feats.append(torch.clone(rot_care_feat))
            else:  # IGNORE_ROT_IDX or None
                # if type not specified, just initialize as "ignore"
                self.obj_rot_feats.append(torch.clone(self.rot_ignore_feat))
            self.obj_rot_feats[-1].requires_grad = rot_requires_grad[i]

            # Rotation offset
            # initialized as "0" which the start feature represents due to training
            self.obj_rot_offsets.append(torch.clone(self.policy_network.rot_offsets_train[Params.START_IDX].data))
            self.obj_rot_offsets[-1].requires_grad = rot_requires_grad[i]

        # Save newly initialized object preference features/offsets
        # also append start and/or goal features with requires_grad=False to freeze them
        self.policy_network.pos_pref_feat_test = self.obj_pos_feats + [
            torch.clone(self.policy_network.pos_pref_feat_train[Params.GOAL_IDX]).data]
        self.policy_network.pos_pref_feat_test[-1].requires_grad = False

        self.policy_network.rot_pref_feat_test = self.obj_rot_feats + [
            torch.clone(self.policy_network.rot_pref_feat_train[Params.START_IDX]).data,
            torch.clone(self.policy_network.rot_pref_feat_train[Params.GOAL_IDX]).data]
        self.policy_network.rot_pref_feat_test[-2].requires_grad = False
        self.policy_network.rot_pref_feat_test[-1].requires_grad = False

        self.policy_network.rot_offsets_test = self.obj_rot_offsets + [
            torch.clone(self.policy_network.rot_offsets_train[Params.START_IDX]).data,
            torch.clone(self.policy_network.rot_offsets_train[Params.GOAL_IDX]).data]
        self.policy_network.rot_offsets_test[-2].requires_grad = False
        self.policy_network.rot_offsets_test[-1].requires_grad = False

        # Define adaptable parameters: only objects are adaptable currently
        # though this should be extendable to also adapt start/goal
        self.adaptable_parameters = self.obj_pos_feats + self.obj_rot_feats + self.obj_rot_offsets

    def add_new_obj(self):
        # Add new "neutral" object features for new object
        # NOTE: Can add extra logic to keep library of seen object types and
        #   their learned features. These could be loaded rather than
        #   initializing as "neutral".
        pass

    def clip_adaptable_parameters(self):
        # Clip all adaptable parameters to be within bounds
        for p in self.obj_pos_feats:
            p.data.clamp_(min=min(self.pos_pref_feat_bounds[0], self.pos_pref_feat_bounds[1]),
                          max=max(self.pos_pref_feat_bounds[0], self.pos_pref_feat_bounds[1]))
        for p in self.obj_rot_feats:
            p.data.clamp_(min=min(self.rot_pref_feat_bounds[0], self.rot_pref_feat_bounds[1]),
                          max=max(self.rot_pref_feat_bounds[0], self.rot_pref_feat_bounds[1]))
