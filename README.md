# SO-101 UMI

UMI-style data collection for the SO-101 arm. A handheld copy of the SO-101's own
gripper records demonstrations, and the policy trained on them runs on an SO-101
follower. Also includes a gripper for kinesthetic teaching on an SO-101 leader arm.

> **Status:** printed and in use. Demos are recorded on a two-arm SO-101 bench, converted to joint
> plans, and replayed on the real arm. Policy training isn't in this repo yet.

![SO-UMI handheld](cad/soumi_preview.png)

## Start here

- **How to set it all up:** [`docs/umi-setup.md`](docs/umi-setup.md). Covers what's different
  from real UMI, tags, cameras, recording, turning demos into robot data, the overhead-camera
  problem and masking, grip force, kinesthetic teaching, and the arm's limits.
- **What to print:** [`cad/README.md`](cad/README.md). The overhead camera sits on The Robot Studio's
  [overhead camera mount](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/Overhead_Cam_Mount_Webcam)
  (STLs also in `sim/assets/vendor/robotstudio_overhead_mount/`).
- **Calibrate your rig:** [`calibration/README.md`](calibration/README.md). Our rig's results are in
  `calibration/example/` to start from.
- **Rig PC settings:** `cp .env.example .env` and fill it in (ssh host, folders on the rig PC). Only the
  recorder and the real-arm runner need it; converting demos doesn't.
- **Kinesthetic teaching on a leader arm:** [`cad/kinesthetic_leader/README.md`](cad/kinesthetic_leader/README.md)
- **Camera viewer and UMI Studio:** [`viewer/README.md`](viewer/README.md)
- **How a handheld demo becomes a real-arm motion (every step):** [`umi/README.md`](umi/README.md#the-full-process-handheld-demo--real-arm)

## Folders

| Path                      | What's in it                                                                                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `cad/`                    | Printable STLs for the handheld (handle, two jaw-with-loop parts, camera mount, fit-test coupon), the ArUco tag sheet PDF, renders, a 3D preview (`.glb`) and the CAD check results        |
| `cad/kinesthetic_leader/` | The 3 parts that turn an SO-101 leader arm into a hand-guided arm with a real gripper and wrist camera, plus an offline 3D viewer                                                          |
| `cad/src/`                | Python scripts that build the parts, run the checks and make the tag sheet                                                                                                                 |
| `cad/stock_umi/`          | The original Stanford UMI gripper for reference: its official Cura 3MF print files, the same parts as STLs, its tag PDFs and print settings. Doesn't fit the SO-101 as-is (see its README) |
| `calibration/`            | Checkerboards (9×6 inner corners, 25 mm squares, A4 and Letter), the lens calibration script, and `example/`: the reference rig's calibration results. Your own results are written here and git-ignored |
| `umi/`                    | Demo → trajectory (`convert.py`), trajectory → SO-101 joint plan with safety checks and trajectory optimization (`to_robot.py`, `trajopt.py`), and the handheld's calibrations. See its README |
| `sim/`                    | MuJoCo twin of the two-arm SO-101 bench (Menagerie SO-101 model, calibrated cameras), used by the `umi/` conversions. See its README                                                      |
| `real/`                   | Runs a `umi/` plan on the real SO-101: a dry-run check, and a live replay a person starts at the rig. Needs the rig's arm drivers (not in this repo yet). See its README                  |
| `docs/`                   | The setup guide                                                                                                                                                                            |
| `viewer/`                 | A browser page for viewing and recording the connected cameras                                                                                                                             |
| `data/`                   | Captures and episodes (git-ignored)                                                                                                                                                        |

The SO-101 gripper geometry comes from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0).
