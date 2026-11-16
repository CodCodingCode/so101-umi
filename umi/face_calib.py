#!/usr/bin/env python3
"""Cube face calibration: where each sticker really sits on the printed handle cube.

Record (rig page, 40-60 s): turn the handle slowly under the overhead camera so it sees pairs of
faces (20+22, 21+22, 23+21, 23+20), pausing ~1 s in each orientation.

Bundle adjustment over every overhead frame with 2+ cube tags: unknowns are the handle pose in
each frame and a small correction (rotation + translation) of tags 20, 21, 22 relative to the CAD
model; tag 23 (back face) is the fixed reference. Minimizes corner reprojection error with a
robust loss. Writes calibration/cube_tags.json (corner positions of each tag in the gripper frame,
mm), which convert.py uses in place of the CAD cube when present and marked good.

Usage
  uv run --with "opencv-python<5" --with numpy --with scipy umi/face_calib.py <capture-timestamp> \
      [--check-pivot <pivot-capture-timestamp>]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import convert as C

REPO = C.REPO
REF = 23
FREE = [20, 21, 22]


def detect(rig_dir: Path, cam, det, step=1):
    frames = []
    for k, (t, f) in enumerate(C.read_frames(rig_dir)):
        if k % step:
            continue
        c, ids, _ = det.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        found = {int(i): cam.normalize(q.reshape(4, 2)) for q, i in zip(c, ids.ravel()) if int(i) in C.CUBE_FACES}
        if len(found) >= 2:
            frames.append((t, found))
    return frames


def tag_model(deltas):
    """Corner positions (gripper frame, mm) for each tag given per-tag corrections {id: (rotvec, trans)}.
    Each correction rotates the tag about its own centre, then shifts it."""
    out = {}
    for i, base in C.CUBE_TAG_CORNERS.items():
        if i in deltas:
            rv, tr = deltas[i]
            R = cv2.Rodrigues(np.asarray(rv, float))[0]
            c = base.mean(0)
            out[i] = (base - c) @ R.T + c + np.asarray(tr, float)
        else:
            out[i] = base.copy()
    return out


def project(model, rv, tv, ids):
    obj = np.vstack([model[i] for i in ids])
    R = cv2.Rodrigues(rv)[0]
    X = obj @ R.T + tv
    return X[:, :2] / X[:, 2:3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--lens", default=str(REPO / "calibration/overhead_lens_640x480.json"))
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--check-pivot", help="pivot recording to re-fit with the corrected cube (before/after)")
    ap.add_argument("--out", default=str(REPO / "calibration/cube_tags.json"))
    args = ap.parse_args()

    ts = Path(args.capture).name.removeprefix("capture-").removesuffix("-rig")
    rig = C.DATA / f"capture-{ts}-rig" / "overhead"
    cam, det = C.Camera(Path(args.lens)), C.make_detector()
    fx = cam.K[0, 0]
    frames = detect(rig, cam, det)
    pairs = Counter("+".join(map(str, sorted(f))) for _, f in frames)
    print(f"{len(frames)} frames with 2+ cube tags; combos: {dict(pairs)}")
    if len(frames) > args.max_frames:  # spread evenly, but keep every combo represented
        idx = np.linspace(0, len(frames) - 1, args.max_frames).astype(int)
        frames = [frames[i] for i in idx]
    seen_with_ref = {i: sum(1 for _, f in frames if i in f and (REF in f or any(j in f for j in FREE if j != i))) for i in FREE}
    print(f"using {len(frames)} frames; frames linking each free tag: {seen_with_ref}")

    # initial poses from the CAD model
    model0 = tag_model({})
    init, obs = [], []
    for t, f in frames:
        ids = sorted(f)
        obj = np.vstack([model0[i] for i in ids]); img = np.vstack([f[i] for i in ids])
        ok, rv, tv = cv2.solvePnP(obj, img, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        init.append(np.r_[rv.ravel(), tv.ravel()]); obs.append((ids, img))
    n = len(obs)
    nf = len(FREE)

    def unpack(x):
        deltas = {i: (x[6 * k:6 * k + 3], x[6 * k + 3:6 * k + 6]) for k, i in enumerate(FREE)}
        poses = x[6 * nf:].reshape(n, 6)
        return deltas, poses

    def residuals(x):
        deltas, poses = unpack(x)
        model = tag_model(deltas)
        r = [(project(model, poses[k, :3], poses[k, 3:], ids) - img).ravel() * fx for k, (ids, img) in enumerate(obs)]
        return np.concatenate(r)

    x0 = np.r_[np.zeros(6 * nf), np.array(init).ravel()]
    m_res = sum(8 * len(ids) for ids, _ in obs)
    S = lil_matrix((m_res, len(x0)), dtype=int)
    row = 0
    for k, (ids, _) in enumerate(obs):
        for i in ids:
            S[row:row + 8, 6 * nf + 6 * k:6 * nf + 6 * k + 6] = 1
            if i in FREE:
                j = FREE.index(i)
                S[row:row + 8, 6 * j:6 * j + 6] = 1
            row += 8
    r0 = residuals(x0)
    print(f"before: corner reprojection RMS {np.sqrt(np.mean(r0 ** 2)):.2f} px, median |r| {np.median(np.abs(r0)):.2f} px")
    sol = least_squares(residuals, x0, jac_sparsity=S, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=200, verbose=0)
    r1 = sol.fun
    print(f"after:  corner reprojection RMS {np.sqrt(np.mean(r1 ** 2)):.2f} px, median |r| {np.median(np.abs(r1)):.2f} px")

    deltas, _ = unpack(sol.x)
    report = {}
    print("\nsticker corrections relative to the CAD placement (tag 23 = reference):")
    for i in FREE:
        rv, tr = deltas[i]
        ang = float(np.degrees(np.linalg.norm(rv)))
        print(f"  ID {i}: moved {np.round(tr, 1).tolist()} mm (|{np.linalg.norm(tr):.1f}| mm), rotated {ang:.1f} deg")
        report[i] = {"shift_mm": np.round(tr, 2).tolist(), "rot_deg": round(ang, 2), "rotvec": np.round(rv, 5).tolist()}
    model = tag_model(deltas)
    rms_after = float(np.sqrt(np.mean(r1 ** 2)))
    big = [i for i in FREE if report[i]["rot_deg"] > 15 or np.linalg.norm(report[i]["shift_mm"]) > 10]
    good = rms_after < 2.0 and not big and all(v >= 20 for v in seen_with_ref.values())
    out = {"capture": ts, "good": good, "reference": REF, "reprojection_rms_px": round(rms_after, 3),
           "frames": n, "combos": dict(pairs), "corrections": report,
           "corners_mm": {str(i): np.round(model[i], 3).tolist() for i in model},
           "note": "corner order = ArUco (TL, TR, BR, BL), gripper-body frame"}
    target = Path(args.out) if good else Path(args.out).with_suffix(".rejected.json")
    target.write_text(json.dumps(out, indent=2))
    print(f"\n{'GOOD' if good else 'NOT USED'}: wrote {target.relative_to(REPO)}"
          + ("" if good else f"  (rms {rms_after:.2f}px, large corrections {big}, links {seen_with_ref})"))

    if args.check_pivot:
        import pivot_calib as PC
        pts = Path(args.check_pivot).name.removeprefix("capture-").removesuffix("-rig")
        pf = detect(C.DATA / f"capture-{pts}-rig" / "overhead", cam, det)
        for label, mdl in (("CAD cube", model0), ("calibrated cube", model)):
            Rs, ps = [], []
            for t, f in pf:
                ids = sorted(f)
                obj = np.vstack([mdl[i] for i in ids]); img = np.vstack([f[i] for i in ids])
                ok, rv, tv = cv2.solvePnP(obj, img, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
                if ok:
                    Rs.append(cv2.Rodrigues(rv)[0]); ps.append(tv.ravel())
            c, P = PC.solve(Rs, ps)
            res = np.array([np.linalg.norm(R @ c + p - P) for R, p in zip(Rs, ps)])
            print(f"pivot clip {pts} with {label}: {len(Rs)} frames, residual RMS {np.sqrt(np.mean(res**2)):.1f} mm, "
                  f"median {np.median(res):.1f} mm, tip {np.round(c, 1).tolist()}")


if __name__ == "__main__":
    main()
