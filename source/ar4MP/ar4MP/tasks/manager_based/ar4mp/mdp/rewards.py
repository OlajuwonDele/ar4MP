# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from .observations import (
    local_condition_index,
    manipulability,
    order_independent_manipulability,
    dynamic_manipulability,
    isotropy_index,
    dynamic_condition_index
)
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

# def reached_ee_goal(env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg):
#     asset: RigidObject = env.scene[asset_cfg.name]

#     # 1. Get the Command Term object from the manager
#     # This replaces the need for 'env.waypoint_cmd'
#     cmd_term = env.command_manager._terms[command_name] 

#     command = env.command_manager.get_command(command_name)
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)

#     curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]] 
#     dist = torch.norm(curr_pos_w - des_pos_w, dim=1)

#     reached = dist < threshold
    
#     # 2. Update the index on the term object we retrieved
#     cmd_term.current_wp_idx[reached] += 1
    
#     last_idx = cmd_term.num_steps - 1
#     finished = cmd_term.current_wp_idx >= last_idx
#     cmd_term.current_wp_idx.clamp_(max=last_idx)

#     # if finished.any():
#     #     env.termination_manager.terminated[finished] = True

#     return reached.float()

def reached_ee_goal(env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    
    cmd_term = env.command_manager._terms[command_name] 

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)

    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]] 
    dist = torch.norm(curr_pos_w - des_pos_w, dim=1)


    reached = dist < threshold
    
    cmd_term.current_wp_idx[reached] += 1

    last_idx = cmd_term.num_steps - 1
    reached_index = cmd_term.current_wp_idx.float() - 1.0
    
    if last_idx > 0:
        index_scale = reached_index / last_idx
    else:
        # Handle case for only one waypoint (last_idx = 0)
        index_scale = torch.ones_like(reached_index)

    cmd_term.current_wp_idx.clamp_(max=last_idx)
    base_reward_value = 5.0
    reach_reward = index_scale * base_reward_value

    progress_reward = reach_reward * (0.1 + 0.9 * torch.exp(-dist))  # encourage agent to keep traversing through the waypoints and not stagnate

    return reach_reward + progress_reward


def reward_lci(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for local condition index (LCI)."""
   
    lci = local_condition_index(env, asset_cfg) 
    return torch.tanh(lci).squeeze(-1)

def reward_manip(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for manipulability (Yoshikawa)."""
    manip = manipulability(env, asset_cfg)
    return torch.tanh(manip / 10.0).squeeze(-1)

def reward_oim(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for order-independent manipulability."""
    oim = order_independent_manipulability(env, asset_cfg)
    return (oim / (oim + 1.0)).squeeze(-1)

def reward_dyn_manip(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for dynamic manipulability."""
    dyn_manip = dynamic_manipulability(env, asset_cfg)
    return (dyn_manip / (dyn_manip + 1.0)).squeeze(-1)

def reward_iso_idx(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for isotropy index."""
    iso_idx = isotropy_index(env, asset_cfg)
    return torch.clamp(iso_idx, 0.0, 1.0).squeeze(-1)

def reward_dci(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for dynamic condition index."""
    dci = dynamic_condition_index(env, asset_cfg)
    return (1.0 / (1.0 + torch.abs(dci))).squeeze(-1)
