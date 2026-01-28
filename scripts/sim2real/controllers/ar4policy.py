import isaacsim.core.utils.stage as stage_utils
import numpy as np
import omni
import torch
import io

from typing import Optional
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_world_pose_from_relative
from isaacsim.core.utils.types import ArticulationAction
from .policy_controller import PolicyController
import os 
from pathlib import Path
from isaacsim.core.prims import XFormPrim

class AR4Policy(PolicyController):
    def __init__(
        self,
        prim_path: str,
        name: str = "ar4",
        root_path: Optional[str] = None,
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        
        root_dir = Path(__file__).resolve().parents[3]
            
        if usd_path == None:
            usd_path = str(
                root_dir / "source" / "ar4MP" / "ar4MP" / "robot" / "ar_mk3" / "ar_mk3.usd"
            )           
        # usd_path = "/home/juwon/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/ar_mk3.usd"
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)
        
        policy_dir = root_dir / "scripts" / "sim2real" / "controllers" / "config" / "rl_motion_plan"

        self.load_policy(
            str(policy_dir / "policy.pt"),
            str(policy_dir / "env.yaml")
        )

        self._action_scale = 0.5
        self._previous_action = np.zeros(7)
        self._policy_counter = 0
        self._decimation = 2

        
        # self.num_joints = len(self.robot.dof_names)
        
        self.has_joint_data = False
        self.current_joint_positions = np.zeros(8)
        self.current_joint_velocities = np.zeros(8)
        self.gripper_base_link = XFormPrim("/World/ar4/gripper_base_link")

        # self.default_pos = np.zeros(8, dtype=np.float32)
        
        # # --- FIX 2: Initialize Default Velocity (was missing) ---
        # self.default_vel = np.zeros(8, dtype=np.float32)

        # self.num_joints = len(self.robot.dof_names)
        # self.gripper_base_link = XFormPrim("/World/ar4/gripper_base_link")
        
        # Optional: Force the robot to this position immediately on start
        # self.robot.set_joint_positions(self.default_pos)
        # self.robot.set_joint_velocities(self.default_vel)


    def update_joint_state(self, position, velocity) -> None:
            """
            Update the current joint state.

            Args:
                position: A list or array of joint positions.
                velocity: A list or array of joint velocities.
            # """

            self.current_joint_positions = np.array(position[:self.num_joints], dtype=np.float32)
            self.current_joint_velocities = np.array(velocity[:self.num_joints], dtype=np.float32)
            self.has_joint_data = True
        
    
    def _compute_observation(self, ee_pose_command: np.ndarray):
        obs = np.zeros(26)

        if not self.has_joint_data:
            return None
        # 1. joint_pos_rel (8 joints: 6 arm + 2 gripper)
        obs[0:8] = self.current_joint_positions - self.default_pos

        # 2. joint_vel_rel (8 joints)
        obs[8:16] = self.current_joint_velocities - self.default_vel
        
        # 3. ee_pose_error (delta pose: 3 pos)
        gripper_base_link_pos, _  =  self.gripper_base_link.get_world_poses()

        obs[16:19] = ee_pose_command[0:3] - gripper_base_link_pos
        print(f"error ={ee_pose_command[0:3] - gripper_base_link_pos}")

        # 4. last_action (6 arm joints)
        obs[19:26] = self._previous_action

        return obs
    
    def forward(self, dt, ee_pose_command:np.ndarray):

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(ee_pose_command)
            # print(f"obs = {obs}")
            if obs is None:
                return None
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy() 

        # self.action = np.zeros(6)
        joint_positions = np.concatenate([
            self.default_pos[:6] + self.action[:6] * self._action_scale,
            self.default_pos[6:] + self.action[6:] * self._action_scale,
        ])
        
     
        action_msg = ArticulationAction(joint_positions=joint_positions)
        self.robot.apply_action(action_msg)
        self._policy_counter += 1

        """Debugging"""
        # print(f"current joint_pos = {self.current_joint_positions}")
        # print(f"current velo = {self.current_joint_velocities}")
        # print(f"previous_action = {self._previous_action}")
        # print(f"action = {self.action}")
        # print(f"joint positions = {joint_positions}")
        # joint_positions = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        """Debugging"""
        return joint_positions


    def initialize(self, physics_sim_view = None) -> None:

        super().initialize(physics_sim_view= physics_sim_view, control_mode="position", set_articulation_props = True, set_limits = False, set_gains = False)

        # self.robot.set_solver_position_iteration_count(6)
        # self.robot.set_solver_velocity_iteration_count(0)

        
        # if hasattr(self.robot, "get_joint_default_positions"):
        #     # self.default_pos = self.robot.get_joint_default_positions()
        #     self.robot.set_joint_positions(self.default_pos)
        #     self.robot.set_joint_velocities(self.default_vel)
        #     self.default_pos = self.robot.get_joint_default_positions()
        
        self.num_joints = len(self.robot.dof_names)
        self.gripper_base_link = XFormPrim("/World/ar4/gripper_base_link")
        """Debugging"""
        print(f"self.robot.dof_names = {self.robot.dof_names}")
        print(f"dof_properties = {self.robot.dof_properties}")
        for name in self.robot.dof_names:
            print(f"{name} = {self.robot.get_dof_index(name)}")
        print(f"joint pos = {self.default_pos}")
        """Debugging"""
        # self.robot.set_joint_positions(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
