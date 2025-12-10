import torch
import pandas as pd
import numpy as np
from dataclasses import MISSING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers



class WaypointCommand(CommandTerm):
    """Command provider that outputs a fixed set of waypoints loaded from Excel."""

    def __init__(self, cfg: "WaypointCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        df = pd.read_excel(cfg.waypoints_path)
        xyz = (df[['X', 'Y', 'Z']]).to_numpy()
        quat = np.tile(np.array([0.7071, 0.70711, 0, 0]), (xyz.shape[0], 1)) 
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
        if cfg.debug_vis:
            self.goal_pose_visualizer = VisualizationMarkers(cfg.goal_pose_visualizer_cfg)
            self.current_pose_visualizer = VisualizationMarkers(cfg.current_pose_visualizer_cfg)
        else:
            self.goal_pose_visualizer = None
            self.current_pose_visualizer = None

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
        Called at every physics step to update the command. Visualization happens HERE.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        cmd = self.paths[env_ids, self.current_wp_idx]  
        
        if self.cfg.debug_vis:
            goal_trans = cmd[:, :3]
            goal_orient = cmd[:, 3:7]

            # Grab environment origins to convert local command -> world position
            env_origins = self._env.scene.env_origins # Shape (N, 3)
            
            # NOTE: VisualizationMarkers can often accept Torch tensors directly on GPU 
            # in newer Isaac Lab versions. If this errors, keep your .cpu().numpy() conversion.
            self.goal_pose_visualizer.visualize(
                translations=goal_trans + env_origins, 
                orientations=goal_orient
            )

            asset = self._env.scene[self.cfg.asset_name]
            body_id = asset.body_names.index(self.cfg.body_name)

            curr_pos = asset.data.body_pos_w[:, body_id]      # (N, 3)
            curr_quat = asset.data.body_quat_w[:, body_id]    # (N, 4)

            self.current_pose_visualizer.visualize(
                translations=curr_pos,
                orientations=curr_quat,
            )
            

        return cmd

    def _update_metrics(self):
        """
        Called to update metrics for logging (optional but required by ABC).
        """
        # Example: Log how far along the path the agents are
        self.metrics["waypoint_index"][:] = self.current_wp_idx


    @property
    def command(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
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

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    waypoints_path: str = MISSING

    """Path to the Excel file containing waypoints."""

    def __post_init__(self):
        # Set scale of visualization markers
        self.goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
