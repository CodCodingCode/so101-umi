#!/usr/bin/env python3
"""Touch calibration: where is each robot base relative to table tag 13?

Torque stays OFF the whole time: nothing on the rig is written, only joint angles are read
(umi/rig/read_joints.py over SSH). You move the arm by hand.

CLOSE the gripper by hand, then for each corner of tag 13's black square put the point where the closed
fingertips meet exactly on the corner, hold it still, and press Enter. The arm's own kinematics turn the joint
angles into that point's position in the base frame (the rig's person-confirmed TCP, tcp_confirmed.json);
fitting the known corner positions gives the base <-> tag transform.

Corners are named as you READ the printed page (the "Table tag: ID 13" header is at the top):

    top-left  +------------+  top-right
              |   tag 13   |
    bot-left  +------------+  bot-right

Run (from the Mac, with monitor.py stopped on the rig):
    python3 umi/touch_calib.py --side left
    python3 umi/touch_calib.py --side right
    python3 umi/touch_calib.py --side left --check     # touch any spot with the closed fingertips, see it in tag coordinates

Writes calibration/robot_bases.json (one entry per side).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rig_link as RL          # noqa: E402
from so101_kin import Arm      # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "calibration" / "robot_bases.json"
H = 0.080                                   # half of the 160 mm black square
CORNERS = {"top-left": (-H, H, 0.0), "top-right": (H, H, 0.0), "bottom-right": (H, -H, 0.0), "bottom-left": (-H, -H, 0.0)}


def read_units(side):
    out = RL.run_py(f"{RL.UMI}\\read_joints.py", side, timeout=60)
    line = next((l for l in out.splitlines() if l.startswith("{")), None)
    if line is None:
        raise SystemExit(f"no reading from the rig:\n{out}")
    r = json.loads(line)
    if "error" in r:
        raise SystemExit(r["error"])
    return np.array(r["units"], float)


def read_still(side, n=3):
    """Average a few readings; refuse if the arm moved in between."""
    us = []
    for _ in range(n):
        us.append(read_units(side))
    us = np.array(us)
    spread = np.ptp(us[:, :5], axis=0).max()
    return us.mean(0), spread


def kabsch(A, B):
    """R, t minimising |R a + t - b|."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True, choices=("left", "right"))
    ap.add_argument("--check", action="store_true", help="after calibrating: touch spots and print them in tag coordinates")
    a = ap.parse_args()

    arm = Arm()
    print("copying the read-only joint reader to the rig…", flush=True)
    RL.sync_rig_scripts()

    if a.check:
        cal = json.loads(OUT.read_text())[a.side]
        T_base_tag = np.array(cal["T_base_tag"])
        T_tag_base = np.linalg.inv(T_base_tag)
        print("Touch any spot with the closed fingertips and press Enter (Ctrl+C to quit). The tag centre should read 0, 0, 0.")
        while True:
            input("> ")
            u, spread = read_still(a.side)
            p = T_tag_base[:3, :3] @ arm.tip(u[:5]) + T_tag_base[:3, 3]
            print(f"   tag frame: x {p[0] * 1000:+7.1f}  y {p[1] * 1000:+7.1f}  z {p[2] * 1000:+6.1f} mm"
                  + ("   (arm moved while reading: hold still)" if spread > 0.5 else ""))

    print(f"\nTouch calibration, {a.side} arm. Torque is OFF; move the arm by hand.")
    print("Close the gripper. Put the closed fingertips exactly on each corner of the black square, hold still, press Enter.")
    print("(Type 's' + Enter to skip a corner the arm can't reach; 3 corners are enough.)\n")
    tag_pts, base_pts, rows = [], [], []
    for name, xyz in CORNERS.items():
        while True:
            ans = input(f"  {name:12s} corner > ").strip().lower()
            if ans == "s":
                break
            u, spread = read_still(a.side)
            if spread > 0.5:
                print(f"     arm moved {spread:.1f} deg during the reading; hold it still and press Enter again")
                continue
            tip = arm.tip(u[:5])
            print(f"     joints {np.round(u[:5], 1).tolist()}  ->  tip in base frame "
                  f"{np.round(tip * 1000, 1).tolist()} mm")
            tag_pts.append(xyz); base_pts.append(tip)
            rows.append({"corner": name, "units": u.round(3).tolist(), "tip_base_m": tip.round(5).tolist()})
            break
    if len(tag_pts) < 3:
        raise SystemExit("need at least 3 corners")
    A, B = np.array(tag_pts), np.array(base_pts)
    R, t = kabsch(A, B)
    res = np.linalg.norm((A @ R.T + t) - B, axis=1) * 1000
    side_len = [np.linalg.norm(B[i] - B[j]) * 1000 for i in range(len(B)) for j in range(i + 1, len(B))]
    T_base_tag = np.eye(4)
    T_base_tag[:3, :3], T_base_tag[:3, 3] = R, t
    tilt = float(np.degrees(np.arccos(np.clip(R[:, 2] @ [0, 0, 1], -1, 1))))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    print(f"\nfit residuals per corner (mm): {np.round(res, 1).tolist()}   rms {np.sqrt((res ** 2).mean()):.1f} mm")
    print(f"tag centre in the {a.side} base frame: x {t[0] * 1000:+.0f}  y {t[1] * 1000:+.0f}  z {t[2] * 1000:+.0f} mm; "
          f"tag yaw {yaw:+.1f} deg; table tilt vs base {tilt:.1f} deg")
    print(f"measured corner-to-corner distances (mm): {np.round(side_len, 0).tolist()} "
          f"(should be 160 and 226)")
    worst = float(res.max())
    if worst > 6:
        print("\n!! A corner is more than 6 mm off. Check the corner ORDER (as you read the page) and that the "
              "closed fingertips were on the corner, then run again. Not saved.")
        sys.exit(1)
    if tilt > 5:
        print(f"\n!! The table looks tilted {tilt:.1f} deg relative to the arm's base. Check the base is flat on the table.")

    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data["_about"] = ("T_base_tag: pose of table tag 13 in each arm's base frame (so101.xml 'base' body), metres. "
                      "From umi/touch_calib.py (closed fingertips = rig TCP on the tag's corners, torque off).")
    data[a.side] = {"T_base_tag": T_base_tag.round(6).tolist(), "rms_mm": round(float(np.sqrt((res ** 2).mean())), 2),
                    "max_mm": round(worst, 2), "tilt_deg": round(tilt, 2), "yaw_deg": round(yaw, 2),
                    "touches": rows, "date": time.strftime("%Y-%m-%d %H:%M")}
    OUT.write_text(json.dumps(data, indent=1))
    print(f"\nsaved {OUT.relative_to(REPO)} [{a.side}]. Verify with: python3 umi/touch_calib.py --side {a.side} --check")


if __name__ == "__main__":
    main()
