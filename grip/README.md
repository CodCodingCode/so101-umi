# grip/: squeeze reflex for the SO-101 gripper

**The problem:** the gripper closes until it touches the object and then stops squeezing. A policy or replay that
learned jaw _angles_ (UMI handheld, teleop) commands the angle where the jaw touched. A position servo pushes in
proportion to (command − position), so once it gets there it pushes with almost nothing. In one of our real UMI
replays the jaw held at 22.7 and 13.3 open with the command within 0.1 of it for seconds, barely holding the object.
[`docs/umi-setup.md` §8](../docs/umi-setup.md#8-grip-force) predicted this.

**The fix:** `GripReflex` (`reflex.py`, pure Python, no dependencies) sits between the policy and the gripper servo,
one call per 25 Hz tick, in the plans' gripper units (0 shut .. 100 open):

```python
from grip.reflex import GripReflex
reflex = GripReflex()                                  # once per run
target[5] = reflex.step(float(target[5]), float(measured[5]))   # every tick, just before the goal write
```

When the command has just closed and settled, and the jaw has reached it or stopped short of it, the reflex commands
4 units past where the jaw really is. It keeps doing that, following the jaw, until the command opens. The servo then
pushes at its torque limit, so the **gripper torque limit sets the squeeze force** (STS3215 torque limit, register 48).
That is the one parameter to calibrate, and later to choose per object. If nothing is there (free air), the jaw closes
at most 12 units, then goes back to the command.

## Sim check

From the so101-umi folder, in the repo's environment (see the top-level README's Install):

```bash
python -m grip.sim_check      # ~2 min
```

432 grasps in the `sim/` twin: widths 22–38 mm, cube 15–100 g, friction 0.25–0.6, three contact softnesses, and the
command at the first-touch angle −0.5 to +1.5 units (calibration error). The test closes, lifts 6 cm, holds 1 s and
opens, at 25 Hz with encoder-rounded positions. Same results on MuJoCo 3.15 (the repo's pin) and 4.0.1, within a point.

|                                        | Reflex off | Reflex on |
| -------------------------------------- | ---------- | --------- |
| Lifted and held                        | 33%        | 100%      |
| … with the command 0.5–1.5 units loose | 0–2%       | 100%      |
| Squeeze while held (median)            | 0 N        | 18.3 N    |
| Let go when the command opened         | 100%       | 100%      |

- **Free air:** the jaw closes at most 10.9 units past the command for about 0.6 s, then returns to it. With the
  command shut it does nothing.
- **Force vs torque limit:** 9.3 / 14.0 / 18.6 N at 50 / 75 / 100% of the fitted 1.32 N m cap, so linear. At 25% the
  sim jaw is too weak to reach the cube in 2 s.

**Sim caveats.** The gripper servo model (`sim/assets/so101_dynamics.json`) was fitted on free moves, never squeezes.
That model is stiff right at contact, which is why "reflex off" still lifts 42% at zero calibration error; the real
gripper apparently doesn't. MuJoCo's soft contacts are not real squishy objects. Three thresholds were tuned against
this same sim while fixing bugs it exposed: `near`, `settle` and `held_gap`. So read these numbers as "the logic
works", not as real-rig rates.

## Before using it on the real arm

1. **Free-air settle gap:** command the jaw part way in free air. It must settle within `held_gap` (0.5 units) of
   its command, or the reflex will mistake free air for an object.
2. **Squeeze force:** squeeze a kitchen scale or load cell at `depth` 4 and the gripper torque limit you run. Check
   P gain 16 vs 32.
3. **Overload protection:** read the gripper's protection registers once (34 protective torque, 35 protection time,
   36 overload torque). If the servo drops to a low overload torque after a stall, it will also "stop squeezing"
   whatever the command says.

**Wiring it in.** Call `reflex.step` in whatever loop sends goals to the servos (plan replay or policy), right
before the gripper goal write and after any clamp that can still change the target:
`target[5] = reflex.step(float(target[5]), float(measured[5]))`. Use one `GripReflex` per arm and call `reset()`
between episodes. Log both the plan's gripper value and the one you sent. It hasn't been run on the real arm yet.
