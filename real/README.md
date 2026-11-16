# real/: run a SO-UMI plan on the real SO-101

```
handheld demo ─ umi/convert.py ─> trajectory.csv ─ umi/to_robot.py ─> plan_<side>.json ─ real/run.py ─> the real arm
```

| File        | Runs on       | What                                                                                    |
| ----------- | ------------- | --------------------------------------------------------------------------------------- |
| `run.py`    | Mac           | Uploads `replay.py` and the plan to `<SOUMI_RIG_REAL_DIR>\umi\` on the rig PC, then runs it there |
| `replay.py` | rig PC        | Checks the plan against the real arms, then (only with `--live`) replays it on one arm              |

Rig settings (ssh host, python, folders) come from `.env`; copy `.env.example`.

> **Needs the rig's arm drivers.** `replay.py` imports `so101_real`, `safety` and `rt_control` from
> `SOUMI_RIG_REAL_DIR` on the rig PC. They are not part of this repo yet, so on another setup `replay.py`
> won't run as-is: use it as a reference for the checks to do before moving an arm.

## Run

Stop `monitor.py` on the rig first (it holds the COM ports).

```bash
# 1. dry run: reads both arms (torque stays off), preflights the whole motion, nothing moves
python3 real/run.py 2026-11-15T01-01-42-040Z --side right

# 2. live: a person at the rig, hand on the power switch. Asks you to type the side to start.
python3 real/run.py 2026-11-15T01-01-42-040Z --side right --live

# slower, or keep the jaw still:
python3 real/run.py 2026-11-15T01-01-42-040Z --side right --live -- --max-deg-s 30 --no-gripper
```

Ctrl+C, or creating a file named `STOP` in `SOUMI_RIG_REAL_DIR`, stops it. Every stop folds the arm to rest and turns torque off. Logs go
to `<SOUMI_RIG_REAL_DIR>\runs\umi_replay_<ts>\`.

Agents (e.g. Claude) only ever run the dry run. A person at the rig starts `--live`.

## Film it: demo vs robot overlap

On http://localhost:8000/studio.html (`python3 viewer/server.py --rig`, with `viewer/rig/umi_capture.py` running
on the rig), the **Run on robot** tab:

1. Pick the recording and arm, tick "I'm at the rig", press **▶ Run on the real arm**. The rig starts recording the
   overhead camera and runs `replay.py --live` (preflight, then the arm starts; `-- --confirm` brings back the typed prompt).
2. When the arm is back at rest, the Mac copies the overhead frames, fetches the replay's `log.json` and runs
   `real/overlay.py`.
3. The comparison is saved to `data/episodes/<ep>/real_runs/<recording>/` and shown on the tab.

`overlay.py` draws, through the calibrated overhead lens: the demo's tool point (green), the plan's TCP (grey) and the
real arm's TCP from its measured joints (magenta). The left half of the video is the robot run, and the right half is the
original demo at the matching demo time. The plot shows the top view, the height, and the robot's distance from the demo
and from the plan.

## What the replay does

1. Warns (does not refuse) when the IK misses the demo by more than 40 mm (p95): those frames are the planner's nearest-safe-pose substitutions. `--max-ik-p95-mm N` brings back a hard refusal.
2. Slows the plan so no arm joint goes faster than 45°/s (`--max-deg-s`), and never faster than the plan's
   `recommended_time_scale`, then resamples it to the rig's 25 Hz control rate.
3. Builds the rig's `safety.Guard` on the training sim's collision shapes and each servo's real range, and
   preflights every pose: the move from where the arm is now, raised, to the plan's start, then the whole plan,
   with the other arm where it really is.
4. Live only: torque on holding the current pose, per-joint torque caps, raise, move to the start
   (minimum-jerk ramps), then stream the plan. Every tick, `Guard.step` re-checks limits, clearances and
   tracking, and limits how far a command may lead the joint. A floor guard holds the arm if the hand would go
   below 0.5 cm.

Only the plan's arm is powered. The other arm is just read for the collision check.

## Status (2026-11-14)

- `2026-11-15T01-01-42-040Z` right: the dry run passes on the rig (1025 poses, lowest hand point 1.1 cm,
  replay 37.8 s at ×2.93). Not yet run live.
- The left plans clamp most frames at the midline (the demo was done on the right half of the table), so the arm
  would not follow the demo even when the plan passes the checks.
