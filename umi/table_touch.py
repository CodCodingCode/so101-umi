#!/usr/bin/env python3
"""Table-plane calibration from a "table touch" recording.

Record (UMI Studio -> Record, ~30 s): jaws closed, handle held like in a demo (jaws down, back face 23
toward the overhead camera). Touch the table with the closed fingertips at ~5 spots spread over the work
area (left / right of tag 13, far left, far forward-right near the box, where you grasp the cube), holding
each one still for ~3 s, lifting in between.

Every still touch is a point that is really ON the table (z = 0). Using the pivot-calibrated jaw tip
(calibration/handheld_tip.json), this finds where those points APPEAR in the table frame defined by tag 13,
fits the plane they really lie on (tilt + height, with tag 13's centre as an extra on-table point) and
writes calibration/table_plane.json. umi/to_robot.py applies it, so demo heights are measured from the real
table everywhere, not from tag 13's slightly tilted estimate.

Usage
  uv run --with "opencv-python<5" --with numpy umi/table_touch.py <capture-timestamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import convert as C

REPO = C.REPO
OUT = REPO / "calibration" / "table_plane.json"


def still_segments(t, P, min_s=1.2, max_std_mm=4.0):
    """Runs of frames where the tip is still: rolling window of min_s seconds with per-axis std <= max_std_mm."""
    segs, i, n = [], 0, len(t)
    while i < n:
        j = i
        while j < n and t[j] - t[i] < min_s:
            j += 1
        if j >= n:
            break
        if P[i:j].std(0).max() <= max_std_mm:
            k = j
            while k < n and P[i:k + 1].std(0).max() <= max_std_mm:
                k += 1
            segs.append((i, k))
            i = k
        else:
            i += 1
    return segs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--lens", default=str(REPO / "calibration/overhead_lens_640x480.json"))
    ap.add_argument("--table-pose", default=str(REPO / "calibration/overhead_table_pose.json"))
    ap.add_argument("--min-still-s", type=float, default=1.2)
    args = ap.parse_args()

    ts = Path(args.capture).name.removeprefix("capture-").removesuffix("-rig")
    rig = C.DATA / f"capture-{ts}-rig" / "overhead"
    if not (rig / "frames.csv").exists():
        raise SystemExit(f"no rig overhead recording at {rig}")
    tip_file = REPO / "calibration" / "handheld_tip.json"
    if not tip_file.exists():
        raise SystemExit("needs the pivot calibration first (calibration/handheld_tip.json)")
    cam, det = C.Camera(Path(args.lens)), C.make_detector()
    rows, table_src = C.track_cube(rig, cam, det, Path(args.table_pose))
    print(f"table frame (tag 13) from {table_src}; contact point = calibrated jaw tip {np.round(C.JAW_TIP, 1).tolist()} mm")

    ok = [r for r in rows if r["valid"]]
    t = np.array([r["t"] for r in ok])
    P = np.array([r["tip"] for r in ok]) * 1000.0            # contact point, table frame, mm
    segs = still_segments(t, P, args.min_still_s)
    touches = []
    for a, b in segs:
        p = np.median(P[a:b], 0)
        if p[2] > 40:            # held still in the air (e.g. before the first touch): not a touch
            continue
        if b - a < 20 or t[b - 1] - t[a] < 0.8:   # a real touch is many frames held still, not a blip across a gap
            continue
        touches.append({"t": [round(float(t[a]), 2), round(float(t[b - 1]), 2)], "frames": int(b - a),
                        "xyz_mm": np.round(p, 1).tolist(), "spread_mm": np.round(P[a:b].std(0), 2).tolist()})
    print(f"{len(touches)} still touches found:")
    for k, tc in enumerate(touches, 1):
        print(f"  {k}. {tc['t'][0]:5.1f}-{tc['t'][1]:5.1f}s  at x {tc['xyz_mm'][0]:6.0f}  y {tc['xyz_mm'][1]:6.0f} mm   "
              f"measured height {tc['xyz_mm'][2]:+6.1f} mm   (still to {max(tc['spread_mm']):.1f} mm)")
    if len(touches) < 3:
        raise SystemExit("need at least 3 still touches spread over the table (hold each ~3 s)")
    X = np.array([tc["xyz_mm"] for tc in touches])
    spread = np.ptp(X[:, :2], 0)
    if spread.min() < 80:
        print(f"  warning: touches only span {spread.round().tolist()} mm; spread them wider for a reliable tilt")
    if len(touches) < 4 or spread.min() < 150:
        # Too few / too clustered to measure a tilt: correct the height only. Every touch is on the table, so
        # their median measured height is the bias of the handheld's height measurement.
        hz = X[:, 2]
        c = float(np.median(hz))
        res = hz - c
        good = float(np.abs(res).max()) < 6.0
        out = {"capture": ts, "good": good, "mode": "height only (touches too clustered for a tilt)",
               "plane": {"a": 0.0, "b": 0.0, "c_mm": c}, "tilt_deg": 0.0,
               "residual_max_mm": round(float(np.abs(res).max()), 2), "touches": touches,
               "note": "z_true = z_measured - c (handheld height bias), mm in tag 13's frame"}
        target = OUT if good else OUT.with_suffix(".rejected.json")
        target.write_text(json.dumps(out, indent=2, default=float))
        print(f"\nheight-only correction: the handheld reads {c:+.1f} mm at table contact "
              f"(touches agree within {np.abs(res).max():.1f} mm) -> every demo height is raised by {-c:.1f} mm")
        print(f"{'GOOD' if good else 'NOT USED (touches disagree)'}: wrote {target.relative_to(REPO)}")
        return
    # plane z = a x + b y + c through the touches; tag 13's centre (0, 0, 0) is on the table too
    A = np.c_[X[:, 0], X[:, 1], np.ones(len(X))]
    zs = X[:, 2]
    A = np.vstack([A, [0, 0, 1]])
    zs = np.r_[zs, 0.0]
    (a, b, c), *_ = np.linalg.lstsq(A, zs, rcond=None)
    res = zs - A @ [a, b, c]
    tilt = float(np.degrees(np.arctan(np.hypot(a, b))))
    print(f"\nreal table plane, in tag 13's frame: height {c:+.1f} mm at the tag, tilt {tilt:.2f} deg "
          f"(slope {a * 1000:+.1f} mm/m in x, {b * 1000:+.1f} mm/m in y)")
    print(f"  fit residuals per touch (mm): {np.round(res[:-1], 1).tolist()}   (tag centre {res[-1]:+.1f})")
    for tc, xyz in zip(touches, X):
        tc["corrected_height_mm"] = round(float(xyz[2] - (a * xyz[0] + b * xyz[1] + c)), 1)
    good = len(touches) >= 3 and float(np.abs(res).max()) < 8.0 and tilt < 8.0
    out = {"capture": ts, "good": good, "plane": {"a": a, "b": b, "c_mm": c}, "tilt_deg": round(tilt, 3),
           "residual_max_mm": round(float(np.abs(res).max()), 2), "touches": touches,
           "note": "z_true = z_measured - (a*x + b*y + c), x/y/z in mm in tag 13's frame"}
    target = OUT if good else OUT.with_suffix(".rejected.json")
    target.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n{'GOOD' if good else 'NOT USED (residuals or tilt too large; re-record with stiller touches)'}: "
          f"wrote {target.relative_to(REPO)}")


if __name__ == "__main__":
    main()
