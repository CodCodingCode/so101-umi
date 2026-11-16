#!/usr/bin/env python3
"""Pivot calibration of the handheld's jaw tip from a rig recording.

Record (rig page): jaws closed, fingertips held on ONE fixed spot (e.g. the centre of tag 13),
rock and twist the handle in every direction (~±30°) for ~20 s.

Every overhead frame with 2+ cube faces gives the gripper-body pose (R_i, p_i) in the table frame.
The tip c (gripper frame) stays at one table point P, so R_i c + p_i = P for all i: a linear
least-squares problem in (c, P), solved robustly (iteratively dropping frames whose residual is
far above the median). Writes calibration/handheld_tip.json, which convert.py uses in place of the
CAD tip when present.

Usage
  uv run --with "opencv-python<5" --with numpy --with matplotlib umi/pivot_calib.py <capture-timestamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import convert as C

REPO = C.REPO
CAD_TIP = C.CAD_JAW_TIP.copy()                                   # cad/src/asm.py, mm
TCP_CONFIRMED = np.array([-6.6, -0.2, -99.6])               # rig's person-confirmed TCP, mm


def solve(Rs, ps, w=None):
    n = len(Rs)
    A = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    for i, (R, p) in enumerate(zip(Rs, ps)):
        A[3 * i:3 * i + 3, :3] = R
        A[3 * i:3 * i + 3, 3:] = -np.eye(3)
        b[3 * i:3 * i + 3] = -p
    if w is not None:
        W = np.repeat(w, 3)
        A, b = A * W[:, None], b * W
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3], x[3:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--lens", default=str(REPO / "calibration/overhead_lens_640x480.json"))
    ap.add_argument("--table-pose", default=str(REPO / "calibration/overhead_table_pose.json"))
    ap.add_argument("--out", default=str(REPO / "calibration/handheld_tip.json"))
    ap.add_argument("--min-faces", type=int, default=2)
    ap.add_argument("--inlier-mm", type=float, default=10.0, help="frames whose tip is within this of the pivot count")
    args = ap.parse_args()

    ts = Path(args.capture).name.removeprefix("capture-").removesuffix("-rig")
    rig = C.DATA / f"capture-{ts}-rig" / "overhead"
    if not (rig / "frames.csv").exists():
        raise SystemExit(f"no rig overhead recording at {rig}")
    cam, det = C.Camera(Path(args.lens)), C.make_detector()
    rows, table_src = C.track_cube(rig, cam, det, Path(args.table_pose))
    print(f"table frame from {table_src}")

    sel = [r for r in rows if r["valid"] and len(r["faces"].split("+")) >= args.min_faces]
    print(f"{len(rows)} overhead frames, {sum(r['valid'] for r in rows)} with a cube pose, "
          f"{len(sel)} with >= {args.min_faces} faces (used)")
    if len(sel) < 20:
        raise SystemExit("too few multi-face frames; rock the handle so two faces face the camera")
    Rs = [C.quat_to_mat(r["q"]) for r in sel]
    ps = [np.asarray(r["p"]) * 1000.0 for r in sel]          # mm, table frame

    # rotation coverage: how far the handle was rocked around its mean orientation
    Rm = Rs[len(Rs) // 2]
    angles = np.array([C.rot_angle_deg(R, Rm) for R in Rs])
    axes = []
    for R in Rs:
        rv = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
        axes.append(Rm.T @ rv)
    spread = np.sqrt(np.linalg.eigvalsh(np.cov(np.array(axes).T)))[::-1]
    print(f"rocking: median {np.median(angles):.0f} deg, max {angles.max():.0f} deg from the middle pose; "
          f"spread per axis (rad-ish) {np.round(spread, 2).tolist()}")

    # RANSAC: the clip usually starts/ends with the handle lifted (not pivoting), so find the largest
    # set of frames whose tip shares one fixed point, then refine on those.
    rng = np.random.default_rng(0)
    best = None
    for _ in range(400):
        idx = rng.choice(len(sel), 4, replace=False)
        if max(C.rot_angle_deg(Rs[a], Rs[b]) for a in idx for b in idx) < 10:
            continue                      # too little rotation in the sample to locate the tip
        c, P = solve([Rs[i] for i in idx], [ps[i] for i in idx])
        res = np.array([np.linalg.norm(R @ c + p - P) for R, p in zip(Rs, ps)])
        inl = res < args.inlier_mm
        if best is None or inl.sum() > best.sum():
            best = inl
    keep = best if best is not None else np.ones(len(sel), bool)
    for _ in range(5):
        c, P = solve([R for R, k in zip(Rs, keep) if k], [p for p, k in zip(ps, keep) if k])
        res = np.array([np.linalg.norm(R @ c + p - P) for R, p in zip(Rs, ps)])
        new = res < args.inlier_mm
        if (new == keep).all():
            break
        keep = new
    tk = np.array([r["t"] for r in sel])[keep]
    print(f"pivoting segment: {keep.sum()} of {len(sel)} frames, t = {tk.min():.1f}-{tk.max():.1f} s "
          f"(the rest: handle lifted / moving, ignored)")
    Ak = [R for R, k in zip(Rs, keep) if k]
    rock = np.array([C.rot_angle_deg(R, Ak[len(Ak) // 2]) for R in Ak])
    angles = rock
    print(f"rocking within the pivot segment: median {np.median(rock):.0f} deg, max {rock.max():.0f} deg")
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    # uncertainty of c: residual-based covariance of the LSQ
    A = np.vstack([np.hstack([R, -np.eye(3)]) for R, k in zip(Rs, keep) if k])
    cov = np.linalg.inv(A.T @ A) * rms ** 2
    sd = np.sqrt(np.diag(cov))[:3]

    print(f"\npivot fit: {keep.sum()}/{len(sel)} frames kept, residual RMS {rms:.1f} mm "
          f"(median {np.median(res[keep]):.1f}, p90 {np.percentile(res[keep], 90):.1f})")
    print(f"  tip in gripper frame (mm):  {np.round(c, 1).tolist()}   +/- {np.round(sd, 1).tolist()} (1 sd)")
    print(f"  CAD tip:                    {np.round(CAD_TIP, 1).tolist()}   -> difference {np.round(c - CAD_TIP, 1).tolist()} "
          f"({np.linalg.norm(c - CAD_TIP):.1f} mm)")
    print(f"  rig confirmed TCP:          {TCP_CONFIRMED.tolist()}  -> difference {np.round(c - TCP_CONFIRMED, 1).tolist()}")
    print(f"  pivot point in table frame (mm): {np.round(P, 1).tolist()}  (tag 13 centre = 0,0,0; z = height above the tag)")

    verdict = []
    if np.median(angles) < 8:
        verdict.append("handle was barely rocked: tip direction poorly constrained, rock more (~±30 deg)")
    if rms > 6:
        verdict.append("large residuals: the tip moved, or the cube poses are noisy")
    if sd.max() > 3:
        verdict.append("tip uncertain by > 3 mm on some axis")
    good = not verdict
    out = {"capture": ts, "tip_in_gripper_mm": np.round(c, 2).tolist(), "tip_sd_mm": np.round(sd, 2).tolist(),
           "pivot_in_table_mm": np.round(P, 2).tolist(), "residual_rms_mm": round(rms, 2),
           "frames_used": int(keep.sum()), "frames_multi_face": len(sel),
           "rocking_median_deg": round(float(np.median(angles)), 1), "rocking_max_deg": round(float(angles.max()), 1),
           "cad_tip_mm": CAD_TIP.round(2).tolist(), "good": good, "warnings": verdict}
    if good:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nGOOD: wrote {Path(args.out).relative_to(REPO)} (convert.py will use this tip)")
    else:
        Path(args.out).with_suffix(".rejected.json").write_text(json.dumps(out, indent=2))
        print("\nNOT GOOD ENOUGH:\n  - " + "\n  - ".join(verdict))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tt = np.array([r["t"] for r in sel])[keep]
        tips = np.array([R @ c + p for R, p, k in zip(Rs, ps, keep) if k])
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for j, n in enumerate("xyz"):
            ax[0].plot(tt, tips[:, j] - P[j], ".", ms=3, label=f"tip {n} - pivot")
        ax[0].set_ylabel("tip offset from pivot (mm)"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(tt, angles, ".", ms=3); ax[1].set_ylabel("rocked (deg)"); ax[1].set_xlabel("t (s)"); ax[1].grid(alpha=.3)
        fig.suptitle(f"pivot calibration {ts}: RMS {rms:.1f} mm"); fig.tight_layout()
        png = C.DATA / f"capture-{ts}-rig" / "pivot_calib.png"
        fig.savefig(png, dpi=100); print(f"plot: {png.relative_to(REPO)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
