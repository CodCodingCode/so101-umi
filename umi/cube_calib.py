#!/usr/bin/env python3
"""Cube-face calibration: where each sticker (IDs 20-23) really sits on the printed cube.

Record (rig page, 40-60 s): turn the handle slowly under the overhead camera so it sees pairs of
faces (20+22, 21+22, 23+21, 23+20), pausing ~1 s in each orientation.

Every overhead frame with 2+ cube faces is one view. A bundle adjustment solves, jointly,
  * one 6-DoF pose per view (cube -> camera), and
  * one 6-DoF correction per tag (where its sticker really is on the cube),
minimising the reprojection error of all tag corners (Huber loss). The reference tag (22 by
default) keeps its CAD pose: from the cube alone only the tags' placement relative to one
another is observable; the whole cube's offset from the gripper body belongs to the pivot
calibration (umi/pivot_calib.py).

Repeatability: the same solve on the even and on the odd views; the difference is reported.

Writes calibration/cube_tags.json (corners of each tag in the gripper frame, mm), which
convert.py uses in place of the CAD corners when present and "good".

Usage
  uv run --with "opencv-python<5" --with numpy --with scipy --with matplotlib umi/cube_calib.py <capture-timestamp>
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import convert as C

REPO = C.REPO
IDS = sorted(C.CUBE_FACES)
CAD = {i: C.tag_corners(C.CUBE_CENTRE + np.array(n) * (C.CUBE_SIDE / 2 + 0.25), n, u, C.TAG_MM)
       for i, (n, u) in C.CUBE_FACES.items()}


def rodrigues(v):
    return cv2.Rodrigues(np.asarray(v, float).reshape(3, 1))[0]


def tag_pts(i, corr):
    """Tag i corners with its correction (rotvec about the tag centre, then translation)."""
    c0 = CAD[i].mean(0)
    return (rodrigues(corr[:3]) @ (CAD[i] - c0).T).T + c0 + corr[3:]


def find_capture(arg):
    p = Path(arg)
    if p.is_dir():
        return p
    name = arg if arg.startswith("capture-") else f"capture-{arg}"
    d = C.DATA / f"{name}-rig"
    if d.is_dir():
        return d
    raise SystemExit(f"no {d}")


def detect(over_dir, cam, det, max_views):
    views = []
    for t, f in C.read_frames(over_dir):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detectMarkers(g)
        if ids is None:
            continue
        found = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if int(i) in C.CUBE_FACES}
        if len(found) >= 2:
            views.append({"t": t, "px": found, "norm": {i: cam.normalize(p) for i, p in found.items()}})
    if len(views) > max_views:
        views = [views[k] for k in np.linspace(0, len(views) - 1, max_views).astype(int)]
    return views


def init_pose(v, prev=None):
    faces = sorted(v["norm"])
    obj = np.vstack([CAD[i] for i in faces])
    img = np.vstack([v["norm"][i] for i in faces])
    r = C.solve_pose(obj, img, prev)
    if r is None:
        return None
    return np.r_[cv2.Rodrigues(r[0])[0].ravel(), r[1]], r[2]


def solve(views, poses, ref, f_px, free=None, loss_px=1.0):
    """Joint fit. Returns (corrections dict, poses array, per-corner residuals px)."""
    free = [i for i in IDS if i != ref] if free is None else free
    nv, nt = len(views), len(free)
    obs = [(k, i) for k, v in enumerate(views) for i in sorted(v["norm"])]

    def unpack(x):
        corr = {i: np.zeros(6) for i in IDS}
        for j, i in enumerate(free):
            corr[i] = x[6 * j:6 * j + 6]
        return corr, x[6 * nt:].reshape(nv, 6)

    def resid(x):
        corr, P = unpack(x)
        pts = {i: tag_pts(i, corr[i]) for i in IDS}
        out = []
        for k, i in obs:
            proj = cv2.projectPoints(pts[i], P[k, :3], P[k, 3:], np.eye(3), None)[0].reshape(-1, 2)
            out.append(((proj - views[k]["norm"][i]) * f_px).ravel())
        return np.concatenate(out)

    S = lil_matrix((8 * len(obs), 6 * nt + 6 * nv), dtype=int)
    for r, (k, i) in enumerate(obs):
        if i in free:
            j = free.index(i)
            S[8 * r:8 * r + 8, 6 * j:6 * j + 6] = 1
        S[8 * r:8 * r + 8, 6 * nt + 6 * k:6 * nt + 6 * k + 6] = 1

    x0 = np.r_[np.zeros(6 * nt), np.asarray(poses).ravel()]
    res = least_squares(resid, x0, jac_sparsity=S, loss="huber", f_scale=loss_px, x_scale="jac",
                        method="trf", max_nfev=200)
    corr, P = unpack(res.x)
    return corr, P, resid(res.x).reshape(-1, 4, 2), obs


def corner_rms(r):
    return float(np.sqrt(np.mean(np.sum(r ** 2, -1))))


def summarize_corr(corr):
    out = {}
    for i in IDS:
        c = corr[i]
        R = rodrigues(c[:3])
        n = np.array(C.CUBE_FACES[i][0], float)
        u = np.array(C.CUBE_FACES[i][1], float)
        # tilt = rotation of the face normal; spin = rotation about the normal
        n2 = R @ n
        tilt = float(np.degrees(np.arccos(np.clip(n2 @ n, -1, 1))))
        u2 = R @ u
        u2p = u2 - (u2 @ n) * n
        spin = float(np.degrees(np.arctan2(np.cross(u, u2p) @ n, u @ u2p)))
        out[i] = {"shift_mm": c[3:].round(2).tolist(), "shift_norm_mm": round(float(np.linalg.norm(c[3:])), 2),
                  "along_normal_mm": round(float(c[3:] @ n), 2),
                  "rot_deg": round(float(np.degrees(np.linalg.norm(c[:3]))), 2),
                  "tilt_deg": round(tilt, 2), "spin_deg": round(spin, 2)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--lens", default=str(REPO / "calibration/overhead_lens_640x480.json"))
    ap.add_argument("--ref", type=int, default=22, help="tag that keeps its CAD pose")
    ap.add_argument("--max-views", type=int, default=600)
    ap.add_argument("--out", default=str(REPO / "calibration/cube_tags.json"))
    args = ap.parse_args()

    cap = find_capture(args.capture)
    cam = C.Camera(Path(args.lens))
    f_px = cam.K[0, 0]
    det = C.make_detector()
    print(f"detecting cube tags in {cap / 'overhead'} ...")
    views = detect(cap / "overhead", cam, det, args.max_views)
    pairs = Counter("+".join(map(str, sorted(v["norm"]))) for v in views)
    print(f"{len(views)} views with 2+ faces: {dict(pairs)}")
    if len(views) < 30:
        raise SystemExit("too few multi-face views")

    poses, cad_err, keep = [], [], []
    prev = None
    for v in views:
        r = init_pose(v, prev)
        if r is None:
            continue
        x, e = r
        keep.append(v); poses.append(x); cad_err.append(e * f_px)
        prev = (rodrigues(x[:3]), x[3:])
    views = keep
    cad_err = np.array(cad_err)
    # drop gross detection failures before the fit (CAD model is within a few px of the truth)
    good = cad_err < max(6.0, 4 * np.median(cad_err))
    views = [v for v, g in zip(views, good) if g]
    poses = [p for p, g in zip(poses, good) if g]
    print(f"CAD-model fit: median {np.median(cad_err):.2f} px, {(~good).sum()} gross outliers dropped")

    corr, P, res, obs = solve(views, poses, args.ref, f_px)
    per_view_rms = np.sqrt(np.mean(np.sum(res ** 2, -1), 1))
    # refit without views whose fit is clearly bad (blur, partial occlusion)
    thr = max(1.5, 3 * np.median(per_view_rms))
    bad_views = {k for (k, _), e in zip(obs, per_view_rms) if e > thr}
    if bad_views:
        views = [v for k, v in enumerate(views) if k not in bad_views]
        P = np.array([p for k, p in enumerate(P) if k not in bad_views])
        corr, P, res, obs = solve(views, P, args.ref, f_px)
    rms_cal = corner_rms(res)

    # CAD baseline on the same views: refit poses only
    _, _, res_cad, _ = solve(views, P, args.ref, f_px, free=[])
    rms_cad = corner_rms(res_cad)

    # per-tag residual
    tag_rms = {}
    for i in IDS:
        rr = [res[r] for r, (_, j) in enumerate(obs) if j == i]
        rc = [res_cad[r] for r, (_, j) in enumerate(obs) if j == i]
        if rr:
            tag_rms[i] = (corner_rms(np.array(rc)), corner_rms(np.array(rr)), len(rr))

    # repeatability: even vs odd views
    ev = [v for k, v in enumerate(views) if k % 2 == 0]
    od = [v for k, v in enumerate(views) if k % 2 == 1]
    ce, _, _, _ = solve(ev, P[0::2], args.ref, f_px)
    co, _, _, _ = solve(od, P[1::2], args.ref, f_px)
    rep = {i: (float(np.linalg.norm(ce[i][3:] - co[i][3:])),
               float(np.degrees(np.linalg.norm(cv2.Rodrigues(rodrigues(ce[i][:3]).T @ rodrigues(co[i][:3]))[0]))))
           for i in IDS if i != args.ref}

    summ = summarize_corr(corr)
    pairs = Counter("+".join(map(str, sorted(v["norm"]))) for v in views)
    print(f"\nfitted on {len(views)} views ({len(bad_views)} more dropped): {dict(pairs)}")
    print(f"corner reprojection RMS: CAD {rms_cad:.3f} px -> calibrated {rms_cal:.3f} px")
    print(f"\n tag  views   CAD px  cal px   shift mm (x,y,z)        |shift|  normal  rot°  tilt°  spin°   repeat mm/°")
    for i in IDS:
        s = summ[i]
        rp = rep.get(i)
        rps = f"{rp[0]:.2f} / {rp[1]:.2f}" if rp else "  (ref)"
        a, b, n = tag_rms.get(i, (np.nan, np.nan, 0))
        print(f" {i:3d}  {n:5d}   {a:6.2f}  {b:6.2f}   {str(s['shift_mm']):22s}  {s['shift_norm_mm']:5.2f}  "
              f"{s['along_normal_mm']:6.2f}  {s['rot_deg']:4.2f}  {s['tilt_deg']:5.2f}  {s['spin_deg']:5.2f}   {rps}")

    # lever arm: what each tag's rotation error does to the jaw tip if it were the only face seen
    tip_arm = float(np.linalg.norm(C.JAW_TIP - C.CUBE_CENTRE))
    print(f"\n(cube centre -> jaw tip lever {tip_arm:.0f} mm: 1° = {np.radians(1) * tip_arm:.1f} mm at the tip)")

    max_rep_mm = max(r[0] for r in rep.values())
    max_rep_deg = max(r[1] for r in rep.values())
    seen = {i for v in views for i in v["norm"]}
    good = bool(rms_cal < rms_cad and max_rep_mm < 1.0 and max_rep_deg < 0.75 and seen == set(IDS))
    out = {
        "capture": cap.name, "lens": args.lens, "ref_tag": args.ref, "views": len(views),
        "pairs": dict(pairs), "rms_px_cad": round(rms_cad, 3), "rms_px_calibrated": round(rms_cal, 3),
        "repeatability": {str(i): {"mm": round(a, 2), "deg": round(b, 2)} for i, (a, b) in rep.items()},
        "corrections": {str(i): summ[i] for i in IDS},
        "corners_gripper_mm": {str(i): tag_pts(i, corr[i]).round(3).tolist() for i in IDS},
        "good": good,
    }
    dest = Path(args.out) if good else Path(args.out).with_suffix(".rejected.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n{'GOOD' if good else 'REJECTED'} -> {dest}")

    # preview: per-view residual before/after over time
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.array([views[k]["t"] for k, _ in obs])
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(t, np.sqrt(np.mean(np.sum(res_cad ** 2, -1), 1)), ".", ms=2, label="CAD tags")
        ax[0].plot(t, np.sqrt(np.mean(np.sum(res ** 2, -1), 1)), ".", ms=2, label="calibrated")
        ax[0].set_ylabel("tag RMS px"); ax[0].legend(); ax[0].set_ylim(0, 3)
        tv = [v["t"] for v in views]
        for j, (a, b) in enumerate(itertools.combinations(IDS, 2)):
            m = [v["t"] for v in views if a in v["norm"] and b in v["norm"]]
            if m:
                ax[1].plot(m, [j] * len(m), "|", ms=8)
        ax[1].set_yticks(range(6)); ax[1].set_yticklabels([f"{a}+{b}" for a, b in itertools.combinations(IDS, 2)])
        ax[1].set_xlabel("s"); ax[1].set_title(f"{len(tv)} views")
        fig.tight_layout()
        fig.savefig(cap / "cube_calib.png", dpi=110)
        print(f"plot -> {(cap / 'cube_calib.png').relative_to(REPO)}")
    except Exception as e:
        print(f"(no plot: {e})")


if __name__ == "__main__":
    main()
