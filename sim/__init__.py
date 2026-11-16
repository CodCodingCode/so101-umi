"""MuJoCo robot playground. No learned policy or training framework required."""

import os

# Real red cube: 3.2 cm edge (measured from the rig's grip test, 2026-11-10). scene.py reads this at import
# time, so it is set before anything else loads; an explicit PI_SIM_CUBE_SIZE still wins.
os.environ.setdefault("PI_SIM_CUBE_SIZE", "0.032")

from gymnasium.envs.registration import register  # noqa: E402

from .env import RobotEnv  # noqa: E402

register(id="PiSim-Reach-v0", entry_point="sim:RobotEnv", kwargs={"task": "reach"})
register(id="PiSim-PickPlace-v0", entry_point="sim:RobotEnv", kwargs={"task": "pick_place"})

__all__ = ["RobotEnv"]
