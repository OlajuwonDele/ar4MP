# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment with Kinematic Metrics.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inferencing a policy on an AR4 robot with Kinematic Analysis.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
import torch
import omni
import numpy as np
import matplotlib.pyplot as plt # Added for plotting

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Ensure source path is added
root_dir = Path(__file__).resolve().parents[2] 
source_dir = str(root_dir / "source" / "ar4MP")
if source_dir not in sys.path:
    sys.path.append(source_dir)

from ar4MP.tasks.manager_based.ar4mp.kine_joint_pos_env_cfg import AR4MPEnvCfg_Play
# from ar4MP.tasks.manager_based.ar4mp.joint_pos_env_cfg import AR4MPEnvCfg_Play
def main():
    """Main function."""
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    # setup environment
    env_cfg = AR4MPEnvCfg_Play()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("✔ Starting IsaacLab Policy Inference  Analysis...")

    ee_positions = []
    goal_positions = []  
    local_cond_indices = []
    manipulabilities = []
    isotropy_indices = []   
    dyn_manips = []
    pos_errors = []
    
    # Get Robot Handle
    robot = env.scene["robot"]
    
    # Identify indices for the arm (excluding gripper for Jacobian calc)
    # AR4 usually has 6 arm joints. We use regex to find them.
    arm_joint_names = ["joint_.*"]
    arm_joint_ids, _ = robot.find_joints(arm_joint_names)
    n = len(arm_joint_ids) # DoF (likely 6)

    # End-effector link index
    ee_frame_name = "gripper_base_link"
    ee_body_id, _ = robot.find_bodies(ee_frame_name)
    ee_body_id = ee_body_id[0]
    
    # Jacobian index is typically body_id - 1 for fixed base robots in PhysX
    ee_jacobian_index = ee_body_id - 1

    # Constants for math
    I = torch.eye(n, device=env.device)
    N_ = (1/n) * I

    # ---------------------------------------------------------
    # Inference Loop
    # ---------------------------------------------------------
    obs, _ = env.reset()
    count = 0
    max_steps = 200 # Limit to prevent infinite loop during plotting test

    with torch.inference_mode():
        while simulation_app.is_running():
            # 1. Policy Inference
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)
            
            # 2. Retrieve State
            # Get full Jacobian [num_envs, num_links, 6, num_joints]
            full_jac = robot.root_physx_view.get_jacobians()
            # Slice for EE body and Arm joints only
            jacobian = full_jac[:, ee_jacobian_index, :, :n]
            
            # Mass matrix: [N, num_dof, num_dof] -> Slice for arm joints
            full_mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()
            mass_matrix = full_mass_matrix[:, :n, :n]

            # Current EE Pose
            ee_pose = robot.data.body_pose_w[:, ee_body_id]
            ee_pos = ee_pose[:, 0:3]

            # 3. Retrieve Goal (from Command Manager)
            # Assuming the task uses 'ee_pose' commands. 
            # If the command is local, we might need to add env_origins.
            # Usually ManagerBasedRLEnv commands are in local frame if they are observations,
            # but let's try to get the raw command.
            try:
                # Try to get command. Shape usually (N, 7) or (N, 3)
                goals = env.command_manager.get_command("ee_pose")
                # If goals are local to env, add origin
                goal_pos = goals[:, 0:3] + env.scene.env_origins
            except:
                # Fallback if specific command name not found
                goal_pos = ee_pos # Just track itself if no goal found

    
            dist = torch.norm(ee_pos - goal_pos, dim=1)
            # Term1: trace(J * N_ * J^T)
            term1 = torch.einsum("bij,jk,bik->b", jacobian, N_, jacobian)

            # Term2: trace(inv(J) * N_ * inv(J)^T)
            # Use pseudo-inverse for stability if singularity occurs
            try:
                jacobian_inv = torch.linalg.inv(jacobian)
            except:
                jacobian_inv = torch.linalg.pinv(jacobian)
                
            term2 = torch.einsum("bij,jk,bik->b", jacobian_inv, N_, jacobian_inv)

            cond_number = (1/n) * torch.sqrt(term1 * term2)
            local_cond_index = 1.0 / (cond_number + 1e-6)

            manipulability = torch.det(jacobian @ jacobian.transpose(1,2))
            order_indep_manip = manipulability ** (1/n)
            
            # Dynamic Manipulability
            try:
                mm_inv = torch.linalg.inv(mass_matrix)
                inner = jacobian @ mm_inv @ mm_inv.transpose(1,2) @ jacobian.transpose(1,2)
                dyn_manip = torch.sqrt(torch.abs(torch.det(inner)))
            except:
                dyn_manip = torch.zeros_like(order_indep_manip)

            # Isotropy
            eigen_mean = torch.einsum("bii->b", jacobian @ jacobian.transpose(1,2)) / n
            isotropy_index = order_indep_manip / (eigen_mean + 1e-6)

            # Store data (for env 0)
            env_idx = 0
            ee_positions.append(ee_pos[env_idx].cpu().clone())
            goal_positions.append(goal_pos[env_idx].cpu().clone()) 
            local_cond_indices.append(local_cond_index[env_idx].cpu().clone())
            manipulabilities.append(order_indep_manip[env_idx].cpu().clone())
            isotropy_indices.append(isotropy_index[env_idx].cpu().clone())
            dyn_manips.append(dyn_manip[env_idx].cpu().clone())
            pos_errors.append(dist[env_idx].cpu().clone())

            count += 1
            if count % 100 == 0:
                print(f"Step {count}/{max_steps}")
            if count >= max_steps:
                break

    # ---------------------------------------------------------
    # 5. Plotting
    # ---------------------------------------------------------
    print("Plotting results...")

    pos_errors = torch.stack(pos_errors)
    goal_positions = torch.stack(goal_positions)       
    local_cond_indices = torch.stack(local_cond_indices)  
    manipulabilities = torch.stack(manipulabilities)      
    isotropy_indices = torch.stack(isotropy_indices)      
    dyn_manips = torch.stack(dyn_manips)                 
    
    # Calculate Mean Positional Accuracy
    mean_accuracy = torch.mean(pos_errors).item()
    std_accuracy = torch.std(pos_errors).item()
    
    mean_lci = torch.mean(local_cond_indices).item()
    mean_manip = torch.mean(manipulabilities).item()
    mean_isotropy = torch.mean(isotropy_indices).item()
    mean_dyn_manip = torch.mean(dyn_manips).item()

    # Print Summary Table
    print("\n" + "="*50)
    print(f"{'METRIC':<30} | {'MEAN VALUE':<15}")
    print("-" * 50)
    print(f"{'Positional Accuracy (m)':<30} | {mean_accuracy:.6f}")
    print(f"{'Standard deviation Accuracy (m)':<30} | {mean_accuracy:.6f}")
    print(f"{'Local Condition Index':<30} | {mean_lci:.6f}")
    print(f"{'Manipulability (Order-Indep)':<30} | {mean_manip:.6f}")
    print(f"{'Isotropy Index':<30} | {mean_isotropy:.6f}")
    print(f"{'Dynamic Manipulability':<30} | {mean_dyn_manip:.6f}")
    print("="*50 + "\n")

    ee_positions = torch.stack(ee_positions)   
    goal_positions = torch.stack(goal_positions)       
    local_cond_indices = torch.stack(local_cond_indices)  
    manipulabilities = torch.stack(manipulabilities)      
    isotropy_indices = torch.stack(isotropy_indices)      
    dyn_manips = torch.stack(dyn_manips)                  

    fig = plt.figure(figsize=(18, 12))

    # Local Condition Index
    ax1 = fig.add_subplot(231, projection='3d')
    sc1 = ax1.scatter(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                    c=local_cond_indices, cmap='viridis')
    ax1.set_title("Local Condition Index")
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    fig.colorbar(sc1, ax=ax1, shrink=0.5, label='LCI')

    # Manipulability
    ax2 = fig.add_subplot(232, projection='3d')
    sc2 = ax2.scatter(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                    c=manipulabilities, cmap='plasma')
    ax2.set_title("Manipulability")
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_zlabel('Z [m]')
    fig.colorbar(sc2, ax=ax2, shrink=0.5, label='Manipulability')

    # Isotropy Index
    ax3 = fig.add_subplot(233, projection='3d')
    sc3 = ax3.scatter(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                    c=isotropy_indices, cmap='cool')
    ax3.set_title("Isotropy Index")
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_zlabel('Z [m]')
    fig.colorbar(sc3, ax=ax3, shrink=0.5, label='Isotropy Index')

    # Dynamic Manipulability
    ax4 = fig.add_subplot(234, projection='3d')
    sc4 = ax4.scatter(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                    c=dyn_manips, cmap='inferno')
    ax4.set_title("Dynamic Manipulability")
    ax4.set_xlabel('X [m]')
    ax4.set_ylabel('Y [m]')
    ax4.set_zlabel('Z [m]')
    fig.colorbar(sc4, ax=ax4, shrink=0.5, label='Dynamic Manipulability')

    # Planned vs Actual Trajectory
    ax5 = fig.add_subplot(235, projection='3d')
    ax5.plot(goal_positions[:, 0], goal_positions[:, 1], goal_positions[:, 2],
            color='orange', linestyle='--', label='Planned/Commanded')
    ax5.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
            color='blue', linestyle='-', label='Actual Policy Path')
    ax5.set_title("Trajectory Comparison")
    ax5.set_xlabel('X [m]')
    ax5.set_ylabel('Y [m]')
    ax5.set_zlabel('Z [m]')
    ax5.legend()

    ax6 = fig.add_subplot(236)
    steps = np.arange(len(pos_errors))
    # Convert to cm for easier reading if preferred, here kept as meters
    ax6.plot(steps, pos_errors.numpy(), color='red')
    ax6.set_title("Positional Accuracy (Tracking Error)")
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Error [m]")
    ax6.grid(True)
    # Calculate Mean Error for label
    mean_err = torch.mean(pos_errors).item()
    ax6.axhline(y=mean_err, color='k', linestyle='--', label=f'Mean: {mean_err:.4f}m')
    ax6.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    simulation_app.close()