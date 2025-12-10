# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG 

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from . import ar4mp_env_cfg
from . import mdp
from isaaclab_assets import AR4_MK3_PD_CFG   # isort: skip


@configclass
class AR4MPEnvCfg(ar4mp_env_cfg.AR4MPEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set AR4 as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = AR4_MK3_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (AR4)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_.*"],
            body_name="gripper_base_link",
            debug_vis=True,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        


@configclass
class AR4MPEnvCfg_Play(ar4mp_env_cfg.AR4MPEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
