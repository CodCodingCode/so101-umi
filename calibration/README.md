# calibration/

Per-rig calibration. The scripts and checkerboards are in git; the files they write here are **not**
(see `.gitignore`), because they only fit one setup.

`example/` holds the reference rig's results (Innomaker wrist camera, The Robot Studio
[overhead camera mount](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/Overhead_Cam_Mount_Webcam),
arms 232 mm apart). To try the pipeline on the sample episode, or to get started before calibrating, copy them in:

```bash
cp -r calibration/example/. calibration/
```

Then replace them one by one with your own, in this order. Every `umi/*.py` script prints its exact `uv run`
line in its docstring.

| # | File written here | Made by | Record | Without it |
|---|---|---|---|---|
| 1 | `results/<name>_640x480.json` (wrist lens) | `calibration/calibrate.py <video> --name <name>` | Checkerboard (`checkerboard_9x6_25mm_*.pdf`, printed at 100%) waved in front of the wrist camera, ~30 s, covering the corners | `umi/convert.py` stops (pass `--umi-lens`) |
| 2 | `overhead_lens_640x480.json` (overhead lens + its pose relative to the arms) | lens: `calibrate.py` on the overhead camera; pose: **no script yet**, copied from `sim/assets/so101_camera_rig.json` (`cameras.overhead`) | Same checkerboard under the overhead camera | `convert.py` stops |
| 3 | `overhead_table_pose.json` (table tag 13) | automatic in `umi/convert.py` | Any demo with tag 13 taped flat and visible | Computed per capture |
| 4 | `cube_tags.json` (where each sticker really sits on the cube) | `umi/cube_calib.py <ts>` | 40–60 s turning the handle under the overhead so it sees pairs of faces | CAD cube (up to ~3 mm at the jaw tip) |
| 5 | `handheld_tip.json` (cube → jaw tip) | `umi/pivot_calib.py <ts>` | 20 s with the closed tips on one spot, rocking the handle ±30° | CAD tip offset |
| 6 | `table_plane.json` (true table height/tilt) | `umi/table_touch.py <ts>` | ~30 s touching the table at ~5 spots with the closed tips | Table = tag 13's plane |
| 7 | `finger_templates.npz` (jaw-opening templates) | automatic in `umi/convert.py` | One demo that goes from fully closed to wide open | Tag reads only; gaps where the tag blurs |

Robot side, also per rig and not yet in this folder:

- The arm's tool point (`TCP_IN_GRIPPER` in `umi/to_robot.py`) was measured on the reference arm.
- Each robot base relative to the overhead camera: `umi/to_robot.py` uses the pose in step 2 (the sim's
  arm layout, 232 mm apart). There is no script yet to measure it on another bench.
- Servo ranges: `sim/assets/so101_real_ranges.json`.

Scripts only overwrite a saved result when the new fit passes its quality checks (`*.rejected.json` records a
fit that didn't).
