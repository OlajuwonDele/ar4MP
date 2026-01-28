# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def kinematic_data(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """Helper to extract robot data safely."""
    robot: Articulation = env.scene[asset_cfg.name]
    # Use the body_ids provided by the config class automatically
    ee_idx = asset_cfg.body_ids[0]
    joint_ids = asset_cfg.joint_ids
    jacobian = robot.root_physx_view.get_jacobians()[:, ee_idx, :, joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, joint_ids, :][:, :, joint_ids]
    
    return jacobian, mass_matrix

def local_condition_index(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Observe the condition number of the end-effector
    """
    jacobian, _ = kinematic_data(env, asset_cfg)
    # N, n, _ = jacobian.shape
    # I = torch.eye(n, device=jacobian.device)
    # N_ = (1/n) * I

    # # Term1: trace(J * N_ * J^T)
    # term1 = torch.einsum("bij,jk,bik->b", jacobian, N_, jacobian)  # shape [N]

    # # Term2: trace(inv(J) * N_ * inv(J)^T)
    # jacobian_inv = torch.linalg.inv(jacobian)
    # term2 = torch.einsum("bij,jk,bik->b", jacobian_inv, N_, jacobian_inv)

    # cond_number = (1/n) * torch.sqrt(term1 * term2)
    # local_cond_index = 1.0 / cond_number

    # return local_cond_index

    # SVD method is more stable and can handle non-square Jacobians i.e better for modularity
    s = torch.linalg.svdvals(jacobian)
    # Cond = max singular value / min singular value
    # 1.0 means the ellipsoid is a sphere (Perfect)
    # 0.0 means the ellipsoid is flat (Singular)
    local_cond_index = s[:, -1] / torch.clamp(s[:, 0], min=1e-6)
    
    return local_cond_index.unsqueeze(-1)

def manipulability(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Observe the manipulability of the end-effector
    """

    jacobian, _ = kinematic_data(env, asset_cfg)
    jj_t = torch.bmm(jacobian, jacobian.transpose(1, 2))
    manip = torch.sqrt(torch.clamp(torch.det(jj_t), min=1e-9))
    return manip.unsqueeze(-1)

def order_independent_manipulability(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    jacobian, _ = kinematic_data(env, asset_cfg)
    n = jacobian.shape[1] # Task space dimension 
    jj_t = torch.bmm(jacobian, jacobian.transpose(1, 2))
    det_jjt = torch.clamp(torch.det(jj_t), min=1e-9)
    return torch.pow(det_jjt, 1/n).unsqueeze(-1)

def dynamic_manipulability(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    jacobian, mass_matrix = kinematic_data(env, asset_cfg)
    m_inv = torch.linalg.pinv(mass_matrix)
    
    m_inv_sq = torch.bmm(m_inv, m_inv) 
    
    dyn_matrix = torch.bmm(torch.bmm(jacobian, m_inv_sq), jacobian.transpose(1, 2))
    return torch.sqrt(torch.clamp(torch.det(dyn_matrix), min=1e-9)).unsqueeze(-1)

def isotropy_index(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    jacobian, _ = kinematic_data(env, asset_cfg)
    n = jacobian.shape[1] # Task space dimension 

    jj_t = torch.bmm(jacobian, jacobian.transpose(1, 2))
    
    # Calculate both means from the same jj_t tensor
    geom_mean = torch.pow(torch.clamp(torch.det(jj_t), min=1e-9), 1/n)
    arith_mean = (torch.einsum("bii->b", jj_t) / n)
    
    return (geom_mean / arith_mean).unsqueeze(-1)

def dynamic_condition_index(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Observe the dynamic condition number of the end-effector
    """
    jacobian, mass_matrix = kinematic_data(env, asset_cfg)
    num_joints = mass_matrix.shape[-1]
    device = mass_matrix.device
    
    # Trace/Mean logic
    eye = torch.eye(num_joints, device=device).unsqueeze(0)
    mass_trace = torch.einsum("bii->b", mass_matrix).unsqueeze(-1).unsqueeze(-1)
    diff_matrix = mass_matrix - (mass_trace / num_joints) * eye

    upper_tri = torch.triu(diff_matrix)
    dyn_cond_mat = 0.5 * torch.bmm(upper_tri.transpose(1, 2), upper_tri)
    
    # Eig returns complex numbers (Real + Imaginary)
    eigvals = torch.linalg.eigvals(dyn_cond_mat)
    # RL policies can only use the Real part
    return torch.real(eigvals).mean(dim=-1).unsqueeze(-1)

def ee_pose_error(env, asset_cfg: SceneEntityCfg, command_name: str):
    """Returns EE pose error: goal current"""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]] 
    
    return des_pos_w - curr_pos_w