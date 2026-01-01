# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96 # Increased for better trajectory data
    max_iterations = 1500  # Give it more time to learn the complex state space
    save_interval = 50
    experiment_name = "ar4_kinematic_motion_planning"
    run_name = ""
    resume = False
    
    # CRITICAL: This fixes the "disruption" by scaling different obs types
    empirical_normalization = True 
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Larger network to handle kinematic complexity
        actor_hidden_dims=[256, 128, 64], 
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.05, # Slightly increased to encourage exploration of better postures
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4, # Lowered slightly for more stable convergence with larger nets
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
