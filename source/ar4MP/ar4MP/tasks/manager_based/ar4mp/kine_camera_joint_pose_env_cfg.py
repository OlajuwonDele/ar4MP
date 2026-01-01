# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass
from . import camera_joint_pos_env_cfg 
from . import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
##
# Pre-defined configs
##
# from ....robot import AR4_MK3_CFG  # isort: skip
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG 
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm

##
# Environment configuration
##


@configclass
class AR4MPCameraEnvCfg(camera_joint_pos_env_cfg.AR4MPCameraEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        

        asset = SceneEntityCfg("robot", body_names=["gripper_base_link"])
        # # override observation policy to include kinematic metrics
        self.observations.policy.local_condition_index = ObsTerm(func=mdp.local_condition_index, params={"asset_cfg": asset})
        self.observations.policy.manipulability = ObsTerm(func=mdp.manipulability, params={"asset_cfg": asset})
        self.observations.policy.order_independent_manipulability = ObsTerm(func=mdp.order_independent_manipulability, params={"asset_cfg": asset})
        self.observations.policy.dynamic_manipulability = ObsTerm(func=mdp.dynamic_manipulability, params={"asset_cfg": asset})
        self.observations.policy.isotropy_index = ObsTerm(func=mdp.isotropy_index, params={"asset_cfg": asset})
        self.observations.policy.dynamic_condition_index = ObsTerm(func=mdp.dynamic_condition_index, params={"asset_cfg": asset})
        self.observations.policy.concatenate_terms = True

        # # override rewards
        self.rewards.lci = RewTerm(func=mdp.reward_lci, weight=0.0001, params={"asset_cfg": asset})
        self.rewards.manip = RewTerm(func=mdp.reward_manip, weight=0.0001, params={"asset_cfg": asset})
        self.rewards.oim = RewTerm(func=mdp.reward_oim, weight=0.0001, params={"asset_cfg": asset})
        self.rewards.dyn_manip = RewTerm(func=mdp.reward_dyn_manip, weight=0.0001, params={"asset_cfg": asset})
        self.rewards.iso_idx = RewTerm(func=mdp.reward_iso_idx, weight=0.0001, params={"asset_cfg": asset})
        self.rewards.dci = RewTerm(func=mdp.reward_dci, weight=0.0001, params={"asset_cfg": asset})
        self.rewards.position_tracking = RewTerm(
            func=mdp.position_command_error_tanh, 
            weight=20.0, # Increased weight
            params={"std": 0.5, "command_name": "ee_pose", "asset_cfg": asset}
        )

@configclass
class AR4MPEnvCfg_Play(AR4MPCameraEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
