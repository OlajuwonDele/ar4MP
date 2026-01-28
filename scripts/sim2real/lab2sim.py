from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import carb

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path

from controllers import AR4Policy
first_step = True
reset_needed = False

ee_pose_command = np.array([0.4, 0.0, 0.3, 0.7071, 0.70711, 0.0, 0.0], dtype=np.float32)

# ee_pose_command = np.array([0.2948,	-0.05,	0.25, 0.7071, 0.70711, 0.0, 0.0], dtype=np.float32)
def on_physics_step(step_size: float) -> None:
    global first_step, reset_needed

    if first_step:
        carb.log_info("Initializing AR4 policy")
        ar4.initialize()
        ar4.update_joint_state(ar4.robot.get_joint_positions(), ar4.robot.get_joint_velocities())
        ar4.post_reset()
        first_step = False

    elif reset_needed:
        carb.log_info("Resetting simulation")
        world.reset()
        first_step = True
        reset_needed = False

    else:
        ar4.update_joint_state(ar4.robot.get_joint_positions(), ar4.robot.get_joint_velocities())
        ar4.forward(step_size, ee_pose_command)


world = World(
   stage_units_in_meters = 1.0,
    physics_dt = 1.0 / 60.0, 
    rendering_dt = 1.0 / 30.0,
)

assets_root = get_assets_root_path()
if assets_root is None:
    carb.log_error("Could not find Isaac Sim assets folder")

define_prim("/World/Ground", "Xform").GetReferences().AddReference(
    assets_root + "/Isaac/Environments/Grid/default_environment.usd"
)

ar4 = AR4Policy(
    prim_path="/World/ar4",
    name="ar4",
    position=np.array([0.0, 0.0, 0.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0])
)


world.reset()
world.add_physics_callback("physics_step", on_physics_step)

carb.log_info("Starting AR4 deployment")

while simulation_app.is_running():
    world.step(render=True)
    if world.is_stopped():
        reset_needed = True

simulation_app.close()
