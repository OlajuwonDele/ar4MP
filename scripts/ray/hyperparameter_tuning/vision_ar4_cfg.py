# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import pathlib
import sys
from typing import Any

# Allow for import of items from the ray workflow.
CUR_DIR = pathlib.Path(__file__).parent
UTIL_DIR = CUR_DIR.parent
sys.path.extend([str(UTIL_DIR), str(CUR_DIR)])
import util
import scripts.ray.hyperparameter_tuning.ar4mp_cfg as ar4mp_cfg
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from ray.tune.stopper import Stopper

# Original

class AR4MPNoTuneJobCfg(ar4mp_cfg.ar4MPJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-AR4MP-RGB-v0"])
        super().__init__(cfg, vary_env_count=False, vary_cnn=False, vary_mlp=False)


class AR4MPCNNOnlyJobCfg(ar4mp_cfg.ar4MPJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-AR4MP-RGB-v0"])
        super().__init__(cfg, vary_env_count=False, vary_cnn=True, vary_mlp=False)


class AR4MPJobCfg(ar4mp_cfg.ar4MPJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-AR4MP-RGB-v0"])
        super().__init__(cfg, vary_env_count=True, vary_cnn=True, vary_mlp=True)

class AR4MPNetJobCfg(ar4mp_cfg.ResNetCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-AR4MP-RGB-v0"])
        super().__init__(cfg)


class AR4MPTheiaJobCfg(ar4mp_cfg.TheiaCameraJob):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-AR4MP-RGB-v0"])
        super().__init__(cfg)


# Kinematic based RL

class AR4MPKineNoTuneJobCfg(ar4mp_cfg.ar4MPJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-AR4MP-Kine-RGB-v0"])
        super().__init__(cfg, vary_env_count=False, vary_cnn=False, vary_mlp=False)


class AR4MPKineCNNOnlyJobCfg(ar4mp_cfg.ar4MPJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-AR4MP-Kine-RGB-v0"])
        super().__init__(cfg, vary_env_count=False, vary_cnn=True, vary_mlp=False)


class AR4MPKineJobCfg(ar4mp_cfg.ar4MPJobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-AR4MP-Kine-RGB-v0"])
        super().__init__(cfg, vary_env_count=True, vary_cnn=True, vary_mlp=True)


class AR4MPKineResNetJobCfg(ar4mp_cfg.ResNetJob):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-AR4MP-Kine-RGB-v0"])
        super().__init__(cfg)


class AR4MPKineTheiaJobCfg(ar4mp_cfg.TheiaJob):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-AR4MP-Kine-RGB-v0"])
        super().__init__(cfg)


class CustomAR4MPProgressReporter(CLIReporter):
    def __init__(self):
        super().__init__(
            metric_columns={
                "training_iteration": "iter",
                "time_total_s": "total time (s)",
                "Episode/Episode_Reward/reached_goal": "goal",
                "Episode/Episode_Reward/end_effector_position_tracking_fine_grained": "pos fine",
                "rewards/time": "rewards/time",
            },
            max_report_frequency=5,
            sort_by_metric=True,
        )


class ar4MPEarlyStopper(Stopper):
    def __init__(self):
        self._bad_trials = set()

    def __call__(self, trial_id: str, result: dict[str, Any]) -> bool:
        iter = result.get("training_iteration", 0)
        
        time_out = result.get("Episode/Episode_Termination/time_out")

        # Mark the trial for stopping if conditions are met
        if 20 <= iter and time_out is not None and time_out > 0:
            self._bad_trials.add(trial_id)

        return trial_id in self._bad_trials

    def stop_all(self) -> bool:
        return False  # only stop individual trials
