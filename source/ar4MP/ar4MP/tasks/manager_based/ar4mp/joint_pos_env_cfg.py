# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass
from . import ar4mp_env_cfg 
from . import mdp
##
# Pre-defined configs
##
from ....robot import AR4_MK3_CFG  # isort: skip
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG 

##
# Environment configuration
##


@configclass
class AR4MPEnvCfg(ar4mp_env_cfg.AR4MPEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to AR4
        self.scene.robot = AR4_MK3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        for r in [
                self.rewards.end_effector_position_tracking,
                self.rewards.end_effector_position_tracking_fine_grained,
                self.rewards.end_effector_orientation_tracking,
                self.rewards.reached_goal,
                self.observations.policy.pose_error,
            ]:
                r.params["asset_cfg"].body_names = ["gripper_base_link"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_.*"], scale=0.5, use_default_offset=True, debug_vis=True, preserve_order = True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_jaw.*_joint"],
            open_command_expr={"gripper_jaw.*_joint": 0.014},
            close_command_expr={"gripper_jaw.*_joint": 0.0},
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "gripper_base_link"

@configclass
class AR4MPEnvCfg_Play(AR4MPEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
