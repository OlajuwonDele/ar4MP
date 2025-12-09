import torch
import pandas as pd
import numpy as np
from dataclasses import MISSING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass



class WaypointCommand(CommandTerm):
    """Command provider that outputs a fixed set of waypoints loaded from Excel."""

    def __init__(self, cfg: "WaypointCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        df = pd.read_excel(cfg.waypoints_path)
        xyz = (df[['X', 'Y', 'Z']] / 100.0).to_numpy()
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (xyz.shape[0], 1))
        waypoints_np = np.hstack((xyz, quat))
        
        waypoints = torch.tensor(
            waypoints_np,
            dtype=torch.float32,
            device=env.device
        )

        self.paths = waypoints.unsqueeze(0).repeat(env.num_envs, 1, 1)
        self.current_wp_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.num_steps = self.paths.shape[1]
        self.metrics["waypoint_index"] = torch.zeros(env.num_envs, device=env.device)

    def _resample_command(self, env_ids: torch.Tensor):
        """
        Called when the command needs to be reset (e.g., at episode start 
        or when the resampling timer expires).
        """
        # Reset the waypoint index to 0 for the specified environments
        self.current_wp_idx[env_ids] = 0
        
        # Return the command (pose) at index 0
        # Shape required: (len(env_ids), 7)
        return self.paths[env_ids, self.current_wp_idx[env_ids]]

    def _update_command(self):
        """
        Called at every physics step to update the command.
        """
        # Logic to update the command. 
        # If you want the robot to move through the waypoints over time, 
        # you need logic here to increment self.current_wp_idx.
        
        # For now, we return the pose at the current index.
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self.paths[env_ids, self.current_wp_idx]

    def _update_metrics(self):
        """
        Called to update metrics for logging (optional but required by ABC).
        """
        # Example: Log how far along the path the agents are
        self.metrics["waypoint_index"][:] = self.current_wp_idx


    @property
    def command(self):
        env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        return self.paths[env_ids, self.current_wp_idx]

    def reset(self, env_ids):
        self.current_wp_idx[env_ids] = 0
        return {}



@configclass
class WaypointCommandCfg(CommandTermCfg):
    """Configuration for the waypoint command generator."""
    class_type: type = WaypointCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    waypoints_path: str = MISSING

    """Path to the Excel file containing waypoints."""

    def __post_init__(self):
        # Set scale of visualization markers
        self.goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
