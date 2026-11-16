# sim/: this repo's own copy of the SO-101 bench twin

`sim/` is a Python package (`import sim`): the parts of our SO-101 bench simulator (`pi_sim`, plus the
`squeeze` clamp-task environment) that the SO-UMI conversions need, copied 2026-11-14 and flattened into one
package.

| Path                                     | What                                                                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`, `env.py`, `scene.py`, ... | pi_sim (was `pi_sim/`): 232 mm two-arm scene, Menagerie SO-101 model, calibrated cameras, dynamics                    |
| `assets/`                                | pi_sim's calibration JSONs, textures and vendored meshes                                                              |
| `squeeze_env.py`                         | `SqueezeEnv` (was `squeeze/env.py`), the bench-matched clamp twin (real joint ranges, grey box, real-matched colours) |

Not included: the clamp task's teachers, RL environments, trained models and data.

Used by `umi/to_robot.py` and `umi/workspace.py` (via `SqueezeEnv`) and `umi/so101_kin.py` (the arm XML).
Run the interactive viewer from the repo root with `python -m sim`.

## Calibration in `assets/` (the reference rig: yours will differ)

- `so101_camera_rig.json`: the overhead camera (lens + pose relative to the two arms) and both robot wrist
  cameras, fitted on our bench. The overhead sits on The Robot Studio's
  [overhead camera mount](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/Overhead_Cam_Mount_Webcam),
  so with the same mount and a 232 mm arm spacing it is a close starting point; refit the lens with
  `calibration/calibrate.py` and check that tag 13 lands flat on the table (`umi/README.md`).
  `so101_camera_rig.arducam_nominal.json` is the uncalibrated nominal version.
- `so101_real_ranges.json`: each servo's usable range on our arms after joint-offset calibration. Replace with
  your own arms' ranges.
- `so101_kinematics.json`, `so101_dynamics.json`: kinematic offsets and fitted joint dynamics.
- Flat layout: `squeeze_env.py` imports `.env` / `.scene`, the gym ids point at `sim:RobotEnv`, and the
  real cube size (`PI_SIM_CUBE_SIZE=0.032`) is set in `__init__.py`, so it applies before `scene.py` loads.
