# SO-UMI converter

Turns a handheld demo (wrist + overhead camera recording) into a joint plan for one SO-101 arm.

```
recording (wrist + overhead, see Input)
  └─ convert.py    → data/episodes/<ts>/trajectory.csv   gripper-body pose in the table frame (tag 13) + jaw opening, 30 Hz
       └─ to_robot.py → data/episodes/<ts>/plan_<side>.json  joint targets for the arm + safety checks
                        data/episodes/<ts>/plan_<side>.mp4   real overhead next to the sim replay
```

None of these scripts touches hardware. Calibrate first: [`../calibration/README.md`](../calibration/README.md)
(or copy `calibration/example/` to try the reference rig's values).

## Input

`convert.py <ts>` looks in `data/` for one recording, in either form:

- **Frame folders** (one clock for both cameras, preferred): `data/capture-<ts>-rig/umi/` and
  `data/capture-<ts>-rig/overhead/`, each with `frames/000001.jpg …` and `frames.csv`
  (`index,t,…`: frame number and seconds since the recording started, same clock for both cameras).
- **Videos**: `data/capture-<ts>-umi-*.mp4` (wrist) and `data/capture-<ts>-overhead-*.mp4` (or an
  `-overhead-*/` frame folder). Container timestamps are used, so start both together.

640×480 at 30 fps, matching the lens calibration. The wrist camera must see the finger tag (ID 1) and the
overhead camera the cube tags (IDs 20–23) and table tag 13.

## Run

```bash
# 1. capture -> episode
uv run --with "opencv-python<5" --with numpy --with matplotlib umi/convert.py <capture-timestamp>

# 2. episode -> plan for the left arm (uses the bench-matched clamp twin, sim/squeeze_env.py in this repo)
uv run --python 3.12 --with mujoco==3.15.0 --with gymnasium==1.4.0 --with numpy --with scipy --with pillow \
    --with "opencv-python<5" --with matplotlib umi/to_robot.py <capture-timestamp> --side left
```

## The full process: handheld demo → real arm

It isn't one big optimization but a chain of steps. Each turns the demo into something the SO-101 can
really do, and each later step changes the motion only where an earlier result would be unsafe or
impossible.

**1. Measure the demo (`convert.py`)**

- The overhead camera finds the cube tags (IDs 20–23) and solves the handle's pose from the corners of every
  visible face.
- **Cube calibration** (`cube_calib.py` → `calibration/cube_tags.json`): where each sticker really sits on the
  printed cube. Without it they were up to 0.5 mm and ~1° off, and 1° there is ~2.7 mm at the jaw tip.
- **Jaw-tip calibration** (`pivot_calib.py` → `calibration/handheld_tip.json`): from the cube's pose to the
  jaw tips.
- **Table measurement** (tag 13 → `calibration/overhead_table_pose.json`): the table surface, so heights are
  measured from it. Tape the tag sheet down flat: a bent or covered sheet tilts the whole table.
- The wrist camera reads the jaw opening from the finger tag.
- Result: the gripper's position, tilt and opening at 30 Hz (`trajectory.csv`).

**2. Hand path → joint angles (`to_robot.py`)**

- **IK every frame:** the 5 joint angles that put the robot's tool point where the jaw tips were, and its
  gripper tilt as close to the demo's as possible. With 5 joints the tilt can't always match; position wins.
- Each frame starts from the previous frame's answer, so the motion stays continuous.
- **Step cap** (`--max-step-deg`): a joint may move only so far per frame. This stops the arm from flipping
  between two poses that put the gripper in the same place.

**3. Safety shaping, only where needed**

- **Out of reach:** where the demo goes somewhere the arm can't reach, or only by overloading a servo
  (holding-torque cap), the planner uses the nearest safe pose.
- **Floor** (`--floor-mm`, default 5): no part of the robot's hand (jaw tips, gripper body, wrist camera mount,
  wrist) may come closer than 5 mm to the table. Lower frames are lifted. If staying above the floor needs a
  bigger joint step than the cap allows, the floor wins and the sim marks the frame "BIG JOINT STEP".
- **Other arm:** too close to the parked other arm → held back.

**4. Gentle at real speed**

- **Smoothing** (`--smooth-s`, 0.1 s): zero-phase, so no delay.
- **Real-time limiter** (`--realtime-vmax` 150°/s, `--realtime-amax` 800°/s²): every joint follows the plan under
  a speed and acceleration cap, run forward and backward in time and averaged, so it neither lags nor leads the
  demo. Only the fastest, most abrupt moments get rounded off (a few degrees). Plans play at 1× on the robot.
  `--realtime-vmax 0` gives the old slowed-down plans instead.
- **Safety beats smoothness:** where that rounding would dip the hand under the floor or overload a servo,
  the frame is pulled back only as far as needed. The hand sits on the floor height and the plan continues.

**5. Check before running**

- The plan's checks (below) and the sim video: the planned arm drawn on the real overhead video. When everything
  is right, the sim gripper sits on the handheld gripper the whole time.

**6. On the real arm (not in this repo)**

- `plan_<side>.json` holds joint targets for every frame (degrees, gripper 0 = closed … 100 = open) plus the
  checks below. Stream them to your arm with your own driver (e.g. LeRobot's SO-101 follower), with its own
  limit and collision checks, slowly the first time.

The only true optimizations are the IK solve in each frame (step 2) and the one-time cube calibration, a bundle
adjustment over ~600 views (step 1). Everything else limits the demo, and only where the robot couldn't do it
as recorded. The most common reason a plan is clamped is a demo outside the arm's reach: keep the hand within
the arm's reach.

## How it works

- **Handle pose:** the overhead camera finds the cube tags (IDs 20–23) and solves one PnP over
  every visible face, using the cube geometry from `cad/src`. Single-face pose flips are rejected
  with a median filter, then the stream is resampled to 30 Hz and lightly smoothed.
- **Table frame:** tag 13. The overhead camera doesn't move, so its pose is saved to
  `calibration/overhead_table_pose.json` whenever tag 13 is visible, and reused when it is covered.
- **Gripper opening:** the wrist camera is rigid on the fixed jaw, so only the moving finger moves
  in the image. Its position comes from the ID 1 tag where readable and from template matching
  elsewhere (`calibration/finger_templates.npz`).
  - Finger tags sit at the spot near the tips (`cad/src/finger_tag_patches.json`, since 2026-11-15 02:43).
    There the 8 mm tags blur enough that one or two cells flip and OpenCV's strict decode rejects them, so
    `read_finger_tag` re-reads the squares the detector found against IDs 0/1 only (≤ 2 bits off, ≥ 3 bits
    closer to one ID than the other).
  - Templates: 8 crops spread evenly over the opening range. A capture replaces the saved bank only if its
    range is at least as wide; otherwise it uses the saved one. Current bank: capture 02-43-09 (fully
    closed to wide open, x 348–620 px). The old-spot bank is kept as `finger_templates.old-spot-2026-11-15T00-00.npz`.
  - Template matches must lie on the tag's one-DOF path through the image (±10 px) and score ≥ 0.65
    (true locks ≥ 0.69, wood-grain locks ≤ 0.60); isolated > 40 px jumps are dropped.
  - Wide open, the moving finger leaves the wrist view; those frames are left empty and `to_robot.py`
    holds the last (widest) gripper value through them.
- **Into the robot:** `calibration/overhead_lens_640x480.json` is the rig's calibrated overhead
  camera (lens + pose in the pi_sim world). Tag 13 lands flat on the table with it (normal
  0.00, −0.01, 1.00; 7 mm low), which validates the chain. IK runs on the gripper body
  (the frame the handle replicates), steering the rig's confirmed TCP.

## Checks in `plan_<side>.json`

IK position/orientation error, frames clamped at the midline (each arm keeps 30 mm on its own
side) and at the floor (hand ≥ 5 mm above the table), joint-limit margin, the largest joint step
between frames, and peak joint speed with the playback factor (1× for real-time plans, capped at
150°/s and 800°/s²). Collision (mount post, box) and tracking checks are left to the safety checks of
whatever runs the plan on the arm.

## Known limits

- The overhead lag between the two videos isn't measured reliably yet (assumed 0).
- Gripper units are approximate (CAD lever → jaw angle → sim 0–100 mapping).
- The overhead stream is 15 fps JPEG at 640×480; cube tags are only ~25–40 px.

## Sim and calibration sources (2026-11-15)

- **Sim:** `umi/to_robot.py` uses this repo's `sim/` (the bench-matched `SqueezeEnv`: both arms in the
  232 mm layout, real joint ranges, the grey box, real colours). The other arm is parked at its folded
  rest pose; the render places the grey box (found once from early overhead frames) and the red cube
  (found in every frame) where the overhead camera sees them, and hides pi_sim's green goal disk.
- **Overhead camera:** read from `sim/assets/so101_camera_rig.json` (the camera the sim renders with), so
  the demo-to-world mapping and the render use one calibration. `--lens` overrides it.
- **Table frame (tag 13):** saved in `calibration/overhead_table_pose.json`. A recording only replaces it
  with a better estimate (≥ 30 inlier sightings, < 5 mm spread, agreeing with the saved one within
  3° / 15 mm); a mostly covered tag 13 can no longer overwrite it.
- **Cube tags:** detections under 15 px or badly distorted are ignored (false hits from screens and
  keyboards), and handle poses off the table (> 70 cm from tag 13) or under it are dropped.

## How the jaw tracking was fixed (2026-11-15)

After the finger tags were moved toward the tips (between captures 01-50 and 02-43), ID 1 stopped
decoding and the jaw trace went mostly empty (03-07: 0 tag reads, 11% of the 30 Hz grid). Each step
below was measured against the tag positions before moving on.

1. **Scan before changing anything.** Count tag reads per frame in both cameras. The cube tags were
   fine (100% pose); ID 1 had 0 reads and the template matcher locked on in 9% of frames.
2. **Rule out settings.** Every 4×4 dictionary, default detector parameters, mirrored frames, upscaling,
   sharpening, CLAHE: still nothing, so the cause wasn't a parameter.
3. **Compare with a capture that worked.** Next to 01-36 (ID 1 reads), sharpness and brightness were
   unchanged, but the stickers had moved; the marks left by the old stickers showed where.
4. **Find the failing stage.** ArUco finds a square, then reads its pattern. The squares the detector
   rejected showed the finding worked; the pattern was within 1–2 cells of ID 1. Fix: re-read those
   squares (`read_finger_tag`) instead of writing a new detector.
5. **Guard against false reads.** Accept only IDs 0/1, ≤ 2 cells off and ≥ 3 cells closer to one ID
   than the other. Sanity check: ID 0 stays put (fixed jaw) and ID 1 moves along one path.
6. **Test templates on a different capture.** Build from one capture, test on another. This exposed
   three problems in turn:
   - templates bunched where the jaw lingered, leaving fully closed and wide open uncovered → spread them evenly over the range;
   - 03-07 never fully closes → build from 02-43 and stop a narrower capture from overwriting the saved bank;
   - occasional readings far from their neighbours → a jump filter.
7. **Look at what the filter dropped.** The dropped frames were the finger half out of view with the
   match landing on wood grain, and neighbouring frames had the same error unflagged. The real
   constraints:
   - geometry: the finger has one degree of freedom, so its tag can only appear on one path in the
     image (matches must be within ±10 px of it);
   - match score: true locks ≥ 0.69, wood grain ≤ 0.60, so the cutoff is 0.65.
8. **Fewer readings, all correct.** 03-07 coverage fell from 98% to 81% at this step, because the extra
   coverage was false locks. Every remaining gap sits next to a wide-open reading (finger out of view).
9. **Check what uses the output.** `to_robot.py`'s 54.2 mm lever already matches the new spot, and it
   holds the last (widest) value through gaps.

Result with tag reads hidden (templates alone vs the tag positions): worst error 1.5 px (~0.25 mm), no
false locks, on both 02-43 and 03-07.

**What carries over:**

- Compare against a version that worked rather than guess.
- Find the exact stage that fails before writing a fix.
- Test on data the fix wasn't built from.
- Look at the frames a filter rejects or keeps, not just the summary numbers.
- Prefer limits from the physical setup or a measured gap in the scores over thresholds tuned by feel.
- Fewer correct readings beat more readings with some false.

## How `to_robot.py` optimizes the arm's path

Frame by frame (30 Hz), greedily, then smoothed. It is **not** one global optimization over the whole
trajectory (see "Limits" below).

### 1. Target per frame

Handle pose from the cube tags → table frame (tag 13) → robot world (calibrated overhead camera) → shifted by
the pivot-calibrated tip offset (`calibration/handheld_tip.json`). That gives the **target**: where the TCP
should be and how the gripper should be tilted.

### 2. Joint angles (IK)

Damped least squares (the rig's `dream_kin.py` update) on the bench twin's MuJoCo model (`sim/`):

- minimizes **position error** (weight 1) + **tilt error** (weight 0.15): a 5-joint arm can't match position and
  full tilt at once, so tilt is weighted lower;
- each iteration is a small damped step, clipped to 0.15 rad and kept **5° inside every joint limit**
  (`--limit-margin-deg`);
- starts from the **previous frame's joints**, so it stays on the same arm configuration; a **20° per-frame**
  guard (`--max-step-deg`) blocks elbow/wrist branch flips.

### 3. Checks

| Hard limits (never broken)                                                                                                               | Soft goal (best effort)                          |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| hand ≥ floor above the table (`--floor-mm`, default 5)                                                                                   | TCP within ~12 mm of the demo (`--reach-tol-mm`) |
| holding torque ≤ 0.9 Nm per joint: gravity load from the model's masses, contacts off (`--max-hold-nm`; STS3215 stall ≈ 1.9 Nm at 7.4 V) | tilt as close as the arm allows                  |
| ≥ 15 mm from the other arm, parked at `--other-pose` (`--clearance-mm`)                                                                  |                                                  |
| ≥ 5° from joint limits (inside the IK)                                                                                                   |                                                  |

### 4. Unsafe frame → nearest safe pose

Greedy, first good answer wins:

1. **same spot, looser tilt** (tilt weight 0.15 → 0.02);
2. **straight back toward the shoulder**, same height, in 2.5 % steps (the usual "too far" case);
3. **shells around the target**: 14 directions at radii 5 mm … 30 cm, then bisection along the best direction
   for the closest point;
4. **nothing meets the accuracy goal** → keep only the hard limits and take the **closest possible** pose,
   then carry on; only if nothing at all is safe, keep the last safe pose.

Every candidate costs a full IK solve plus the checks (why out-of-range demos take longer to convert).
Video banners say which case applied: `TILT CHANGED`, `NEAREST SAFE POSE: N mm …`, `CLOSEST POSSIBLE: N mm …`.

### 5. Whole-path clean-up

1. **smoothing**: zero-phase Gaussian, 0.1 s (`--smooth-s`); frames where it would break the floor keep their
   unsmoothed pose;
2. **real-time limiter**: each joint follows the plan at ≤ 150 °/s and ≤ 800 °/s² (`--realtime-vmax`,
   `--realtime-amax`), run forward and backward and averaged so it neither lags nor leads;
3. **safety re-check**: where 1 or 2 would push a frame below the floor or over the torque cap, it is blended
   back toward the safe pose, only as far as needed.

### 6. Report

`plan_<side>.json`: the joint path plus every check (misses, torque, floor, clearance, speed, which frames
were adjusted and why). The video banners read from it.

### Limits, and the upgrade

- **Greedy, frame by frame**: each frame only sees the previous one, so it can't plan ahead (e.g. start
  pulling back early so a reach limit is approached smoothly instead of swerving late). Smoothing hides
  some of that.
- **Fixed tilt weight** (0.15), not tuned per demo.
- **The safe-pose search is a heuristic**: it finds _a_ nearby safe pose, not provably the closest.

The principled upgrade is a **trajectory optimization**: solve all frames together, minimizing tracking error

- smoothness (joint velocity and acceleration) subject to the hard limits as constraints. That gives smooth,
  anticipatory motion and the true best compromise, using the same MuJoCo model.

### `--optimize` (default): whole-demo trajectory optimization (`umi/trajopt.py`)

On by default; `--no-optimize` gives the frame-by-frame plan alone. The optimizer's cost has no other-arm term, so the re-check below also enforces the other-arm clearance.

Unknowns: every frame's 5 joint angles at once. One least-squares cost (scipy `least_squares`, banded sparse
Jacobian), started from the frame-by-frame plan:

| Term | What | Weight |
| --- | --- | --- |
| tracking | TCP vs your hand (mm), tilt vs yours (deg), on frames where the handle was seen | 1 / 0.25 |
| smoothness | joint velocity and acceleration between frames | 0.6 / 3 |
| floor | hinge: hand's lowest point (mesh subsample) below the floor | 60 per mm |
| torque | hinge: holding torque over the cap | 400 per Nm |
| joint limits | hard bounds, 5° inside each range | — |

Afterwards every frame is re-checked with the full meshes; a frame a hair outside a hard limit is blended back
toward the safe frame-by-frame pose. Smoothing is skipped (it is part of the cost); the real-time limiter still runs.

Episode `02-43-09` (356 frames, right arm), miss vs your hand on tracked frames: median 4.4 → **0.5 mm**,
p95 27 → **5.0 mm**, max 55 → **9.0 mm**; about the same jerkiness (accel RMS 0.138 vs 0.149 °/frame²);
0 frames below the floor. The optimizer itself takes about 1 s on recent episodes.

```bash
uv run --python 3.12 --with mujoco==3.15.0 --with gymnasium==1.4.0 --with numpy --with scipy --with pillow \
    --with "opencv-python<5" --with matplotlib --with scipy umi/to_robot.py <ts> --side right
```
