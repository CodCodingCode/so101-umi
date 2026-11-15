#!/usr/bin/env python3
"""Draw where an SO-101 arm can actually grasp onto the real overhead image.

For a grid of table positions at grasp height, solve IK (the rig's MuJoCo model, the same IK as
umi/to_robot.py) for a jaws-down gripper with the camera facing away from the arm, and keep the
points the arm reaches within tolerance without its hand going under the floor or crossing the
midline. Those points are projected through the rig's calibrated overhead camera onto a real
overhead frame, so you can see where to perform handheld demos for that arm.

Output: calibration/workspace_<side>.png  (+ .json with the reachable world points)

Usage
  uv run --python 3.12 --with mujoco==3.15.0 --with gymnasium==1.4.0 --with numpy --with pillow \
      --with "opencv-python<5" umi/workspace.py --side left --frame data/capture-...-overhead-640x480.mp4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import to_robot as TR

REPO = TR.REPO


def jaws_down(yaw):
    """Gripper-body rotation in the world: jaws (-z body) pointing down, camera (+y body) toward yaw."""
    z_b = np.array([0, 0, 1.0])                  # body +z = up (jaws point down)
    y_b = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    x_b = np.cross(y_b, z_b)
    return np.stack([x_b, y_b, z_b], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left", choices=("left", "right"))
    ap.add_argument("--frame", help="overhead video or image to draw on (default: newest overhead capture)")
    ap.add_argument("--heights-mm", type=float, nargs="+", default=[20.0, 60.0], help="TCP heights to test")
    ap.add_argument("--floor-mm", type=float, default=10.0)
    ap.add_argument("--midline-mm", type=float, default=30.0)
    ap.add_argument("--tol-mm", type=float, default=5.0)
    ap.add_argument("--tol-deg", type=float, default=25.0)
    ap.add_argument("--step-mm", type=float, default=15.0)
    args = ap.parse_args()

    lens = json.loads((REPO / "calibration/overhead_lens_640x480.json").read_text())
    K, D = np.array(lens["K"], float), np.array(lens["D"], float)
    R_wc = TR.quat_wxyz_to_mat(lens["world_pose"]["quat_wxyz"]) @ np.diag([1, -1, -1])
    c_w = np.array(lens["world_pose"]["pos"])
    rvec = cv2.Rodrigues(R_wc.T)[0]
    tvec = -R_wc.T @ c_w

    src = args.frame or str(sorted((REPO / "data").glob("capture-*-overhead-*.mp4"))[-1])
    if src.endswith((".png", ".jpg")):
        img = cv2.imread(src)
    else:
        cap = cv2.VideoCapture(src)
        ok, img = cap.read()
        cap.release()
    arm = TR.ArmModel(args.side)
    sign = -1.0 if args.side == "left" else 1.0
    base = arm.d.xpos[arm.m.body(f"{args.side}_base").id].copy()
    step = args.step_mm / 1000.0
    xs = np.arange(-0.45, 0.45 + 1e-9, step)
    ys = np.arange(-0.25, 0.45 + 1e-9, step)
    results = {}
    for h in args.heights_mm:
        good = []
        for x in xs:
            if sign * x < args.midline_mm / 1000.0:
                continue
            for y in ys:
                p = np.array([x, y, h / 1000.0])
                if np.linalg.norm(p[:2] - base[:2]) > 0.45:
                    continue
                yaw = math.atan2(p[1] - base[1], p[0] - base[0])   # camera facing away from the arm
                R = jaws_down(yaw)
                q, pe, ae = arm.ik(p, R, TR.SIM_HOME6[:5], iters=120)
                if pe * 1000 <= args.tol_mm and ae <= args.tol_deg and arm.hand_lowest_z() >= args.floor_mm / 1000.0:
                    good.append(p.tolist())
        results[h] = good
        print(f"height {h:g} mm: {len(good)} reachable grid points")

    overlay = img.copy()
    colors = {0: (60, 200, 60), 1: (40, 160, 255), 2: (255, 120, 60)}
    for k, (h, pts) in enumerate(sorted(results.items())):
        if not pts:
            continue
        P = np.array(pts)
        uv = cv2.projectPoints(P, rvec, tvec, K, D)[0].reshape(-1, 2)
        r = max(3, int(step * 1000 * 0.35))
        for u, v in uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(overlay, (int(u), int(v)), r, colors[k % 3], -1)
    out = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    by = cv2.projectPoints(np.array([base]), rvec, tvec, K, D)[0].reshape(2)
    cv2.drawMarker(out, (int(by[0]), int(by[1])), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    y0 = 20
    for k, h in enumerate(sorted(results)):
        cv2.putText(out, f"{args.side} arm reach, jaws down, TCP at {h:g} mm", (8, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colors[k % 3], 2)
        y0 += 20
    cv2.putText(out, "red + = arm base", (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    path = REPO / f"calibration/workspace_{args.side}.png"
    cv2.imwrite(str(path), out)
    (REPO / f"calibration/workspace_{args.side}.json").write_text(json.dumps(
        {"side": args.side, "frame": "pi_sim world (m)", "jaws": "down, camera facing away from the base",
         "tol_mm": args.tol_mm, "tol_deg": args.tol_deg, "floor_mm": args.floor_mm, "midline_mm": args.midline_mm,
         "reachable": {str(h): p for h, p in results.items()}}, indent=1))
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
