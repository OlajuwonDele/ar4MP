# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass
from . import joint_pos_env_cfg 
from . import mdp
from . import ar4mp_env_cfg
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
from isaaclab.managers import CurriculumTermCfg as CurrTerm

asset = SceneEntityCfg("robot", body_names=["gripper_base_link"])

@configclass
class KinematicRewardsCfg(ar4mp_env_cfg.RewardsCfg):
    """Kinematic Reward terms for the MDP."""

    lci = RewTerm(func=mdp.reward_lci, weight=0.02, params={"asset_cfg": asset})
    manip = RewTerm(func=mdp.reward_manip, weight=0.02, params={"asset_cfg": asset})
    oim = RewTerm(func=mdp.reward_oim, weight=0.02, params={"asset_cfg": asset})
    dyn_manip = RewTerm(func=mdp.reward_dyn_manip, weight=0.02, params={"asset_cfg": asset})
    iso_idx = RewTerm(func=mdp.reward_iso_idx, weight=0.02, params={"asset_cfg": asset})
    dci = RewTerm(func=mdp.reward_dci, weight=0.02, params={"asset_cfg": asset})


@configclass
class KinematicCurriculumCfg(ar4mp_env_cfg.CurriculumCfg):
    """Kinematic Curriculum terms for the MDP."""

    lci = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "lci", "weight": 0.1, "num_steps": 4500}
    )

    manip = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "manip", "weight": 0.1, "num_steps": 4500}
    )

    oim = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "oim", "weight": 0.1, "num_steps": 4500}
    )

    dyn_manip = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "dyn_manip", "weight": 0.1, "num_steps": 4500}
    )

    iso_idx = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "iso_idx", "weight": 0.1, "num_steps": 4500}
    )

    dci = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "dci", "weight": 0.1, "num_steps": 4500}
    )
@configclass
class AR4MPKineEnvCfg(joint_pos_env_cfg.AR4MPEnvCfg):
    """Configuration for the Kinematic inspired AR4MP environment."""
    rewards: KinematicRewardsCfg = KinematicRewardsCfg()
    curriculum: KinematicCurriculumCfg = KinematicCurriculumCfg()
    def __post_init__(self):
        super().__post_init__()
        
@configclass
class AR4MPEnvCfg_Play(AR4MPKineEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
