# sim/: this repo's own copy of the SO-101 bench twin

`sim/` is a Python package (`import sim`): a private copy of the parts of the shared
`robots/sim/pi_sim_so101` (git `a19af9c`, copied 2026-11-14) that the SO-UMI conversions need. There, the
code is split into `pi_sim/` and `squeeze/`. Here both sit directly in `sim/`. Edits here don't touch the
shared training sim, and edits there don't reach here.

| Path                                     | What                                                                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`, `env.py`, `scene.py`, ... | pi_sim (was `pi_sim/`): 232 mm two-arm scene, Menagerie SO-101 model, calibrated cameras, dynamics                    |
| `assets/`                                | pi_sim's calibration JSONs, textures and vendored meshes                                                              |
| `squeeze_env.py`                         | `SqueezeEnv` (was `squeeze/env.py`), the bench-matched clamp twin (real joint ranges, grey box, real-matched colours) |

Not copied: the clamp task's teachers, RL envs, trained models and data (~120 GB in `squeeze/`), and `pi_model/`.

Used by `umi/to_robot.py` and `umi/workspace.py` (via `SqueezeEnv`) and `umi/so101_kin.py` (the arm XML).
Run the interactive viewer from the repo root with `python -m sim`.

## Differences from the shared copy

- `assets/so101_real_ranges.json`: shifted by the round-6 joint offsets
  (`robots/so101/calibration/actuators/kinematics_refit.json`, tag `bb1b28`) so the ranges match the live
  `C:\robot\real\rig.json`. The shared copy still holds the 2026-11-11 values.
- Flat layout: `squeeze_env.py` imports `.env` / `.scene`, the gym ids point at `sim:RobotEnv`, and the
  real cube size (`PI_SIM_CUBE_SIZE=0.032`, set by `squeeze/env.py` in the shared copy) is set in
  `__init__.py`, so it applies before `scene.py` loads.

Calibration otherwise matches `robots/so101/calibration` (cameras `6349a0`, kinematics `bb1b28`, dynamics).
When that folder changes, re-copy the affected `pi_sim/assets/*.json` files into `assets/`.
