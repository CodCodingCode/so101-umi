# Camera Viewer

A dependency-free browser page showing the **UMI camera** (plugged into this
laptop) next to the rig's **overhead camera** (on the rig PC), with capture and
snapshot for both.

## Run

```bash
python3 viewer/server.py --tunnel dzkchen     # optional: --port 8000
```

Then open http://localhost:8000 and allow camera access when prompted.

- `--tunnel dzkchen` opens `ssh -L 8104:127.0.0.1:8104 dzkchen` (or reuses it if
  port 8104 is already forwarded). Leave it off to run with the UMI camera only.
- The overhead feed comes from `real\monitor.py` on the rig PC, which holds the
  rig cameras, so **monitor.py must be running there**. The page shows
  "Overhead offline" and keeps retrying until it is.
- Don't press Start on the monitor's own page (http://localhost:8104) unless you
  mean to run the model on the arms.

Captures and snapshots are saved to `data/` at the repo root, sharing a timestamp, e.g.
`data/capture-2026-11-14T10-00-00-000Z-umi-640x480.mp4` (wrist camera, recorded in the browser) and
`data/capture-2026-11-14T10-00-00-000Z-overhead-640x480/` (overhead, recorded by `server.py`:
`frames/000001.jpg …` plus `frames.csv` with each frame's arrival time, and `meta.json`).

The overhead is recorded server-side on purpose: the rig streams it at ~15 fps with uneven
spacing, and re-recording it in the browser at a fixed rate made playback speed wobble. The
server keeps the feed connected all the time, so a recording starts on the next frame.

## Features

- Pick any connected camera from the dropdown (list updates when devices are plugged/unplugged)
- Choose resolution (640×480 up to 4K). Exact native modes only: if the camera is busy in
  another mode you get an error instead of a silently different size. Every mode is locked to
  ~30 fps, so cameras with a slow raw 1080p mode (e.g. Innomaker U20CAM: 5 fps) use their fast
  30 fps mode instead
- Live resolution + FPS readout for both cameras
- Mirror toggle (UMI view and snapshots)
- **Capture**: click to start recording, click again to stop (MP4 in Chrome/Safari, WebM
  fallback). Records the overhead too while "Record overhead" is ticked and the feed is live
- Snapshot to PNG (both cameras)

## Caveats

- The overhead feed is what monitor.py streams: 640×480 JPEG at ~15 fps (`--fps 15`), with
  frames 40–130 ms apart. Its timestamps are arrival times on this laptop, so they include the
  tunnel's latency and jitter (tens of ms); monitor.py doesn't send capture times.
- Browser recordings have no per-frame timestamps. For SLAM data, use a dedicated recorder.

## Recording on the rig PC (`rig/`)

`rig/umi_capture.py` is the half of the viewer that runs **on the rig PC** (on the rig it lives at `C:\robot\umi\umi_capture.py`):
it opens the wrist (UMI) and overhead cameras there, timestamps every frame on one clock, and writes
recordings to the rig's disk; only a small preview crosses the network. Stop `monitor.py` first (it holds
the cameras). It never touches the arms.

It also records the robot arm's wrist camera as `robot_wrist` (`--robot-wrist 2`, rig.json `right_wrist`;
`-1` turns it off). After a robot run, `real/overlay.py` uses it to write a "what the arm sees" video:
UMI demo wrist | robot wrist | 50/50 overlay, saved with the run as `real_runs/<run>/wrist.mp4` and shown
first under "Saved comparisons" in the studio. (The left arm's wrist camera, index 3, is the handheld's
camera, so this only works for the right arm.)

```bash
python3 viewer/server.py --rig dzkchen     # opens the 8106 tunnel; then use http://localhost:8000/studio.html
```

Recordings are fetched into `data/<name>-rig/` (`umi/convert.py` reads that layout).

## UMI Studio (`studio.html`): the simple way

```bash
python3 viewer/server.py --rig dzkchen     # then open http://localhost:8000/studio.html
```

Sidebar tabs:

- **Record**: the wrist and overhead camera previews (labelled, with what each should show), one Record
  button, and which arm to make the sim for. When you stop, the recording is processed automatically
  (copy from the rig → track the handle and gripper → plan the arm → render the sim vs real video) with a
  progress screen.
- **Episodes**: every recording, on the rig or the Mac, with its status. Pick one to play the sim vs real
  video and read the checks in plain words; "Re-make sim" reruns it (e.g. after a new calibration).
- **Run on robot**: pick a recording with a plan and the arm, check its sim and checks, tick "I'm at the rig
  with a hand on the power switch", press Run. The rig preflights every pose (real joint limits and
  clearances, `real/replay.py`); then the arm starts. ■ STOP folds the arm to rest. The
  overhead films the run, and a comparison of the UMI demo vs the robot is saved to
  `data/episodes/<ep>/real_runs/<recording>/` (comparison.mp4/.png, stats.json, umi_demo.csv, plan.json,
  robot_log.json, meta.json) and shown on the tab.

`studio.html` is the only page (the old `rig.html` was merged into it).
