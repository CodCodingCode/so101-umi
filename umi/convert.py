#!/usr/bin/env python3
"""SO-UMI converter: viewer capture pair -> handheld gripper trajectory.

Input: a capture recorded by viewer/ (two files sharing a timestamp):
    data/capture-<ts>-umi-640x480.mp4        wrist camera on the handheld
    data/capture-<ts>-overhead-640x480.mp4   rig overhead camera (via monitor.py)

Output: data/episodes/<ts>/
    gripper_pose.csv   per overhead frame: pose of the handheld's gripper body in the TABLE frame
                       (ArUco ID 13), from the 4-face tag cube (IDs 20-23)
    jaw.csv            per wrist frame: moving-jaw angle (rad) from the moving finger
    trajectory.csv     both resampled on one 30 Hz clock, gaps filled + flagged, smoothed
    meta.json          inputs, calibrations, coverage, lag estimate
    preview.png        plots for a quick sanity check

Frames and units
  * gripper body frame = the official SO-101 MJCF "gripper" body (the handle replicates the
    robot's horn plate, so the handheld and the robot share it). mm in the CAD, m here.
  * table frame = ArUco tag 13: origin at the tag centre, x right / y up in the printed tag
    image, z out of the table. Expressing it in a robot's base frame is a separate step.
  * jaw angle: rotation of the moving jaw about its hinge, 0 = the CAD pose used for the
    finger-tag spot (see cad/src/finger_tag_patches.json), sign: + opens.

Usage
  uv run --with "opencv-python<5" --with numpy --with matplotlib umi/convert.py 2026-11-15T00-06-19-417Z
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _rel(p) -> str:
    """Repo-relative path for files written into outputs (no home-folder paths in shared data)."""
    p = Path(p).resolve()
    return os.path.relpath(p, REPO) if p.is_relative_to(REPO) else p.name
DATA = REPO / "data"

# ---------------------------------------------------------------- geometry (cad/src, mm)
CUBE_SIDE = 36.0
CUBE_CENTRE = np.array([0.0, -2.0, 58.0])          # gripper frame
TAG_MM = 25.0
CUBE_FACES = {  # id: (outward normal, tag "up" direction), gripper frame (cad/src/tag_cube_view.py)
    20: ([1, 0, 0], [0, 0, -1]),
    21: ([-1, 0, 0], [0, 0, -1]),
    22: ([0, 1, 0], [0, 0, -1]),
    23: ([0, 0, 1], [0, 1, 0]),
}
TABLE_TAG, TABLE_TAG_MM = 13, 160.0
CAD_JAW_TIP = np.array([-7.9, -0.218, -98.13])      # gripper frame (cad/src/asm.py), never overridden
JAW_TIP = CAD_JAW_TIP.copy()
_TIP_FILE = REPO / "calibration" / "handheld_tip.json"   # written by umi/pivot_calib.py only when the fit is good
if _TIP_FILE.exists():
    _tip = json.loads(_TIP_FILE.read_text())
    if _tip.get("good"):
        JAW_TIP = np.array(_tip["tip_in_gripper_mm"], float)
# moving-jaw hinge and wrist-camera pose in the gripper frame (cad/src/asm.py)
JAW_HINGE = np.array([20.2, 18.8, -23.4])
FINGER_TAG_ID, FINGER_TAG_MM = 1, 8.0
PATH_TOL_PX = 10   # template matches must sit within this of the moving tag's path through the wrist image
# min template score. Measured on the new-spot captures (02-43, 03-07): true locks score >= 0.69 (p1),
# wood-grain locks while the finger is half out of view (wide open) score <= 0.60
MATCH_MIN = 0.65


def quat_to_mat(q):  # (x, y, z, w) -> 3x3
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):  # 3x3 -> (x, y, z, w), w >= 0
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q = np.array(q)
    return q if q[3] >= 0 else -q


def tag_corners(centre, normal, up, size):
    """3D corners (TL, TR, BR, BL as ArUco returns them) of a square tag."""
    n, u = np.asarray(normal, float), np.asarray(up, float)
    r = np.cross(u, n)
    h = size / 2
    c = np.asarray(centre, float)
    return np.array([c - r * h + u * h, c + r * h + u * h, c + r * h - u * h, c - r * h - u * h])


CUBE_TAG_CORNERS = {  # gripper frame, mm
    i: tag_corners(CUBE_CENTRE + np.array(n) * (CUBE_SIDE / 2 + 0.25), n, u, TAG_MM)
    for i, (n, u) in CUBE_FACES.items()
}
_CAD_TAG_CORNERS = {i: c.copy() for i, c in CUBE_TAG_CORNERS.items()}
_cube = {}
_CUBE_FILE = REPO / "calibration" / "cube_tags.json"    # written by umi/cube_calib.py only when the fit is good
if _CUBE_FILE.exists():
    _cube = json.loads(_CUBE_FILE.read_text())
    if _cube.get("good"):
        CUBE_TAG_CORNERS.update({int(i): np.array(c, float) for i, c in _cube["corners_gripper_mm"].items()})


# ---------------------------------------------------------------- cameras
class Camera:
    def __init__(self, path: Path):
        c = json.loads(path.read_text())
        self.model = c.get("model", "pinhole")
        self.K = np.array(c["K"], float)
        self.D = np.array(c.get("D", c.get("dist")), float).ravel()
        self.size = tuple(c.get("image_size", c.get("image", [640, 480])))
        self.source = str(path)

    def normalize(self, px):
        p = np.asarray(px, np.float64).reshape(-1, 1, 2)
        if self.model == "fisheye":
            return cv2.fisheye.undistortPoints(p, self.K, self.D).reshape(-1, 2)
        return cv2.undistortPoints(p, self.K, self.D).reshape(-1, 2)


def make_detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # tuned on SO-UMI captures: small, slightly blurred tags (see umi/README.md)
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 4
    p.minMarkerPerimeterRate = 0.01
    p.polygonalApproxAccuracyRate = 0.08
    p.perspectiveRemovePixelPerCell = 8
    p.perspectiveRemoveIgnoredMarginPerCell = 0.2
    p.minDistanceToBorder = 1
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p)


_FINGER_BITS = {i: (cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                                   i, 6, borderBits=1) > 127).astype(int)[1:5, 1:5] for i in (0, 1)}


def read_finger_tag(g, quad, max_bits=2, margin=3, cell=8):
    """Relaxed decode of a quad the detector found but could not read, restricted to the two
    finger tags. At the finger-tag spot near the tips (moved there 2026-11-15) the 8 mm tags are
    blurred enough that one or two of the 16 cells flip, so the strict decoder rejects them
    even though the black square is found cleanly.

    Returns (id, quad with corner 0 = the tag's top-left) when the bits are within max_bits of
    ID 0 or 1, at least `margin` closer to it than to the other, and the border is dark;
    else None."""
    n = 6 * cell
    dst = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    m = cell // 4
    best = []
    for s in range(4):
        q = np.roll(quad.reshape(4, 2), -s, axis=0).astype(np.float32)
        w = cv2.warpPerspective(g, cv2.getPerspectiveTransform(q, dst), (n, n)).astype(float)
        cells = np.array([[w[r * cell + m:(r + 1) * cell - m, c * cell + m:(c + 1) * cell - m].mean()
                           for c in range(6)] for r in range(6)])
        border = np.r_[cells[0], cells[5], cells[1:5, 0], cells[1:5, 5]]
        inner = cells[1:5, 1:5]
        if inner.max() - border.mean() < 15:
            return None
        thr = (border.mean() + inner.max()) / 2
        if (border < thr).mean() < 0.9:
            return None
        bits = (inner > thr).astype(int)
        for i, ref in _FINGER_BITS.items():
            best.append((int((bits != ref).sum()), i, q))
    best.sort(key=lambda b: b[0])
    h, i, q = best[0]
    other = min(b[0] for b in best if b[1] != i)
    if h <= max_bits and other - h >= margin:
        return i, q
    return None


def rot_angle_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def solve_pose(obj_mm, img_norm, prev=None, ref_R=None):
    """Pose of an object (points in mm) from normalized image points -> (R, t[mm], err) or None.

    A single planar square (4 points) has two valid poses (the mirror ambiguity that IPPE
    returns as two solutions). When ref_R is given, keep the candidate whose orientation is
    closest to it instead of the one with the lowest reprojection error, which flips easily
    on a small, nearly face-on tag."""
    obj = np.asarray(obj_mm, np.float64)
    img = np.asarray(img_norm, np.float64)
    if len(obj) == 4 and ref_R is not None:
        n, rvs, tvs, errs = cv2.solvePnPGeneric(obj, img, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        if n == 0:
            return None
        cands = [(cv2.Rodrigues(rvs[i])[0], tvs[i], float(np.ravel(errs)[i])) for i in range(n)]
        R, tv, _ = min(cands, key=lambda c: rot_angle_deg(c[0], ref_R))
        rv = cv2.Rodrigues(R)[0]
        ok = True
    elif len(obj) == 4:
        ok, rv, tv = cv2.solvePnP(obj, img, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    elif prev is not None:
        rv0, tv0 = cv2.Rodrigues(prev[0])[0], prev[1].reshape(3, 1).copy()
        ok, rv, tv = cv2.solvePnP(obj, img, np.eye(3), None, rv0, tv0, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rv, tv = cv2.solvePnP(obj, img, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(obj, img, np.eye(3), None, rv, tv)
    R = cv2.Rodrigues(rv)[0]
    err = np.sqrt(np.mean(np.sum((cv2.projectPoints(obj, rv, tv, np.eye(3), None)[0].reshape(-1, 2) - img) ** 2, 1)))
    return R, tv.ravel(), float(err)


def read_frames(path: Path):
    """Yield (t_seconds, frame).

    A directory is a server-side overhead recording (frames/*.jpg + frames.csv with each frame's
    arrival time): read as is. A video file uses its container timestamps, dropping exact repeats."""
    if path.is_dir():
        rows = (path / "frames.csv").read_text().splitlines()[1:]
        for line in rows:
            i, t = line.split(",")[:2]
            f = cv2.imread(str(path / "frames" / f"{int(i):06d}.jpg"))
            if f is not None:
                yield float(t), f
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    last_sig = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        sig = cv2.resize(f, (32, 24), interpolation=cv2.INTER_AREA).tobytes()
        if sig == last_sig:  # canvas recording repeats the 15 fps relay frames
            continue
        last_sig = sig
        yield t, f
    cap.release()


# ---------------------------------------------------------------- overhead: cube -> gripper pose
def slerp_R(Ra, Rb, w):
    """Rotation a fraction w of the way from Ra to Rb."""
    qa, qb = mat_to_quat(Ra), mat_to_quat(Rb)
    if np.dot(qa, qb) < 0:
        qb = -qb
    d = float(np.clip(np.dot(qa, qb), -1, 1))
    th = math.acos(d)
    q = qa if th < 1e-6 else (math.sin((1 - w) * th) * qa + math.sin(w * th) * qb) / math.sin(th)
    return quat_to_mat(q / np.linalg.norm(q))


def solve_translation(R, obj_mm, img_norm):
    """Translation (mm) that best projects R @ obj + t onto normalized image points, rotation fixed."""
    A, b = [], []
    for P, (u, v) in zip(np.asarray(obj_mm, float), np.asarray(img_norm, float)):
        X = R @ P
        A.append([1, 0, -u]); b.append(u * X[2] - X[0])
        A.append([0, 1, -v]); b.append(v * X[2] - X[1])
    t, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    Xc = (R @ np.asarray(obj_mm, float).T).T + t
    err = float(np.sqrt(np.mean(np.sum((Xc[:, :2] / Xc[:, 2:3] - img_norm) ** 2, 1))))
    return t, err


def live_detections(cam_dir: Path):
    """Detections made live on the rig while recording (viewer/rig/umi_capture.py: <cam>/detections.jsonl).
    Used when the frames themselves were not copied. Yields (t, {tag id: 4x2 corners px}, raw record)."""
    with open(cam_dir / "detections.jsonl") as f:
        for line in f:
            r = json.loads(line)
            yield r["t"], {int(k): np.array(v, float).reshape(4, 2) for k, v in r["tags"].items()}, r


def is_live(cam_dir: Path):
    """Use the rig's live detections whenever they exist (seconds instead of minutes); frames only otherwise."""
    return (cam_dir / "detections.jsonl").exists()


def track_cube(video: Path, cam: Camera, det, table_pose_file: Path, max_reproj_px=2.0, max_hold_s=1.0,
               min_tag_px=15.0, max_range_m=0.7):
    """Pose of the handheld's gripper body per overhead frame.

    Frames with 2+ cube faces: full PnP over all faces (well constrained).
    Frames with 1 face: a ~30 px tag can't give a usable orientation (its two mirror solutions
    were ~170 deg off the 2-face poses around them), so the orientation is interpolated from the
    nearest 2-face poses (within max_hold_s) and the face only fixes the position."""
    table_R, table_t, table_n = [], [], 0
    dets = []
    def frames_or_live():
        if is_live(video):
            for t, found, _ in live_detections(video):
                yield t, found
            return
        for t, f in read_frames(video):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = det.detectMarkers(g)
            yield t, ({} if ids is None else {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel())})
    for t, found in frames_or_live():
        if TABLE_TAG in found:
            obj = tag_corners([0, 0, 0], [0, 0, 1], [0, 1, 0], TABLE_TAG_MM)
            r = solve_pose(obj, cam.normalize(found[TABLE_TAG]))
            if r and r[2] * cam.K[0, 0] < max_reproj_px:
                table_R.append(r[0]); table_t.append(r[1]); table_n += 1
        def plausible(q):
            e = np.linalg.norm(q - np.roll(q, 1, 0), axis=1)
            return e.mean() >= min_tag_px and e.max() / max(e.min(), 1e-6) < 3.0
        # cube tags are 25-45 px in practice; tiny ones are false hits (keyboards, screens) or too oblique to trust
        faces = [i for i in CUBE_FACES if i in found and plausible(found[i])]
        dets.append((t, faces, {i: cam.normalize(found[i]) for i in faces}))

    # pass 1: multi-face frames
    rows = [{"t": t, "faces": faces, "R": None, "p": None, "err_px": None, "mode": ""} for t, faces, _ in dets]
    prev = None
    for k, (t, faces, norm) in enumerate(dets):
        if len(faces) < 2:
            continue
        obj = np.vstack([CUBE_TAG_CORNERS[i] for i in faces])
        img = np.vstack([norm[i] for i in faces])
        r = solve_pose(obj, img, prev)
        if r and r[2] * cam.K[0, 0] < max_reproj_px * (1 + len(faces)):
            rows[k].update(R=r[0], p=r[1], err_px=r[2] * cam.K[0, 0], mode="multi")
            prev = (r[0], r[1])

    # pass 2: single-face frames, orientation from the neighbouring multi-face poses
    multi = [(rows[k]["t"], rows[k]["R"]) for k in range(len(rows)) if rows[k]["mode"] == "multi"]
    mt = np.array([m[0] for m in multi])
    for k, (t, faces, norm) in enumerate(dets):
        if len(faces) != 1 or not len(mt):
            continue
        j = int(np.searchsorted(mt, t))
        before = multi[j - 1] if j > 0 and t - mt[j - 1] <= max_hold_s else None
        after = multi[j] if j < len(multi) and mt[j] - t <= max_hold_s else None
        if before and after:
            R = slerp_R(before[1], after[1], (t - before[0]) / max(after[0] - before[0], 1e-6))
        elif before or after:
            R = (before or after)[1]
        else:
            continue  # no reliable orientation nearby: leave this frame out
        p, err = solve_translation(R, CUBE_TAG_CORNERS[faces[0]], norm[faces[0]])
        if err * cam.K[0, 0] < 4 * max_reproj_px:
            rows[k].update(R=R, p=p, err_px=err * cam.K[0, 0], mode="single")

    # Table frame: the overhead camera is fixed, so one pose serves every session. Only replace the saved
    # pose with a clearly good estimate from this capture (many sightings, tight spread) that agrees with it;
    # a mostly-covered tag 13 gives a few bad sightings, and those must never overwrite a good pose.
    saved = json.loads(table_pose_file.read_text()) if table_pose_file.exists() else None
    cand = None
    if table_n >= 5:
        T = np.array(table_t)
        inl = np.linalg.norm(T - np.median(T, 0), axis=1) < 10.0   # drop sightings with the tag half covered
        T, Rs = T[inl], np.array(table_R)[inl]
        tt = np.median(T, 0)
        Rm = Rs[np.argmin([np.linalg.norm(R - np.median(Rs, 0)) for R in Rs])]
        spread = T.std(0)
        cand = {"R": Rm.tolist(), "t_mm": tt.tolist(), "sightings": int(inl.sum()), "sightings_total": table_n,
                "spread_mm": spread.round(2).tolist(), "from": str(video)}
    good_new = cand is not None and cand["sightings"] >= 30 and max(cand["spread_mm"]) < 5.0
    if good_new and saved:
        dang = rot_angle_deg(np.array(cand["R"]), np.array(saved["R"]))
        dpos = float(np.linalg.norm(np.array(cand["t_mm"]) - np.array(saved["t_mm"])))
        if dang > 3.0 or dpos > 15.0:
            good_new = False
            print(f"      tag 13 here disagrees with the saved table pose ({dang:.1f} deg, {dpos:.0f} mm): kept the "
                  f"saved one. If the camera or tag really moved, delete {table_pose_file.name} and re-run.")
    if good_new and (not saved or cand["sightings"] >= saved.get("sightings", 0)):
        table_pose_file.write_text(json.dumps(cand, indent=2))
        table, source = cand, f"this capture ({table_n} sightings)"
    elif saved:
        table = saved
        why = "tag 13 not seen" if cand is None else f"only {cand['sightings']} sightings, spread {max(cand['spread_mm']):.0f} mm" \
            if not good_new else "saved pose has more sightings"
        source = f"saved {table_pose_file.name} ({Path(saved['from']).parent.name}, {saved['sightings']} sightings; here: {why})"
    elif cand is not None:
        table, source = cand, f"this capture ({table_n} sightings, below the quality bar; not saved)"
    else:
        raise SystemExit("Table tag 13 not visible and no saved table pose. Record a second with tag 13 clear.")
    R_ct, t_ct = np.array(table["R"]), np.array(table["t_mm"])  # table -> camera

    # pass 3: single-face frames with NO 2-face pose nearby (e.g. only the back face toward the camera for a
    # long stretch). A single small tag has two mirror solutions; keep the one that is physically plausible:
    # jaws pointing down toward the table and the jaw tip above the table, not the flipped one in the air.
    for k, (t, faces, norm) in enumerate(dets):
        if len(faces) != 1 or rows[k]["mode"]:
            continue
        obj = CUBE_TAG_CORNERS[faces[0]]
        n_sol, rvs, tvs, errs = cv2.solvePnPGeneric(obj, norm[faces[0]], np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        plaus = []
        for i in range(n_sol):
            R = cv2.Rodrigues(rvs[i])[0]
            tv = tvs[i].ravel()
            R_tg = R_ct.T @ R
            jaw_down = (R_tg @ np.array([0, 0, -1.0]))[2]            # < 0: jaws point at the table
            tip_z = (R_ct.T @ (tv + R @ JAW_TIP - t_ct))[2]
            err = float(np.ravel(errs)[i]) * cam.K[0, 0]
            if jaw_down < -0.3 and -40.0 <= tip_z <= 500.0 and err < 4 * max_reproj_px:
                plaus.append((err, R, tv))
        if len(plaus) == 1:                                           # exactly one sensible solution: take it
            err, R, tv = plaus[0]
            rows[k].update(R=R, p=tv, err_px=err, mode="single-prior")

    out = []
    for r in rows:
        if r["R"] is None:
            out.append({"t": r["t"], "valid": 0, "faces": "", "p": [np.nan] * 3, "q": [np.nan] * 4,
                        "tip": [np.nan] * 3, "err_px": np.nan})
            continue
        R_tg = R_ct.T @ r["R"]                     # gripper -> table rotation
        p_tg = R_ct.T @ (r["p"] - t_ct)            # gripper origin in table frame (mm)
        tip = p_tg + R_tg @ JAW_TIP
        if np.linalg.norm(p_tg[:2]) > max_range_m * 1000 or not (-50 <= tip[2] <= 600):
            out.append({"t": r["t"], "valid": 0, "faces": "", "p": [np.nan] * 3, "q": [np.nan] * 4,
                        "tip": [np.nan] * 3, "err_px": np.nan})   # impossible place: off the table / under it
            continue
        out.append({"t": r["t"], "valid": 1, "faces": "+".join(map(str, r["faces"])), "p": p_tg / 1000.0,
                    "q": mat_to_quat(R_tg), "tip": tip / 1000.0, "err_px": r["err_px"]})
    return out, source


# ---------------------------------------------------------------- wrist: moving finger -> jaw angle
def track_jaw(video: Path, cam: Camera, det, bank_file: Path):
    """Moving-jaw position from the ID 1 finger tag where readable, extended to the other
    frames by template-matching the finger (the camera is rigid w.r.t. the fixed jaw).

    Templates + the pixel->mm map come from this capture when its tag sightings cover a wide
    range of openings (and are then saved to bank_file), else from bank_file."""
    frames, tag, relaxed = [], {}, set()
    obj = tag_corners([0, 0, 0], [0, 0, 1], [0, 1, 0], FINGER_TAG_MM)
    if is_live(video):
        # The rig already found the finger tag and template-matched the moving finger on every frame.
        z = np.load(bank_file, allow_pickle=True) if bank_file.exists() else None
        a, b = (z["px_to_mm"] if z is not None else (None, None))
        out, n_tag, n_tmp = [], 0, 0
        for t, found, rec in live_detections(video):
            if FINGER_TAG_ID in found:
                r = solve_pose(obj, cam.normalize(found[FINGER_TAG_ID]))
                if r:
                    out.append({"t": t, "valid": 1, "x_px": float(found[FINGER_TAG_ID].mean(0)[0]),
                                "lateral_mm": float(r[1][0]), "source": "tag"})
                    n_tag += 1
                    continue
            if "finger" in rec and a is not None:
                x = float(rec["finger"][0])
                out.append({"t": t, "valid": 1, "x_px": x, "lateral_mm": float(a * x + b), "source": "template"})
                n_tmp += 1
            else:
                out.append({"t": t, "valid": 0, "x_px": np.nan, "lateral_mm": np.nan, "source": ""})
        return out, {"tag_frames": n_tag, "frames": len(out), "px_to_mm": [float(a or 0), float(b or 0)],
                     "templates": f"tracked live on the rig ({n_tmp} template frames)"}
    for k, (t, f) in enumerate(read_frames(video)):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append((t, g))
        corners, ids, rejected = det.detectMarkers(g)
        quads = [c.reshape(4, 2) for c, i in zip(corners, [] if ids is None else ids.ravel())
                 if int(i) == FINGER_TAG_ID]
        if not quads:   # strict decode missed it: try the blurred-tag fallback on rejected squares
            for c in rejected:
                if cv2.contourArea(c.reshape(4, 2).astype(np.float32)) < 150:
                    continue
                m = read_finger_tag(g, c)
                if m and m[0] == FINGER_TAG_ID:
                    quads.append(m[1]); relaxed.add(k)
                    break
        for q in quads[:1]:
            r = solve_pose(obj, cam.normalize(q))
            if r:
                tag[k] = (q.mean(0), r[1])  # pixel centre, position in camera frame (mm)
    relaxed &= set(tag)
    n = len(frames)
    H, W = 70, 90
    ks = sorted(tag)
    xs = np.array([tag[k][0][0] for k in ks]) if ks else np.array([])
    if len(ks) >= 30 and np.ptp(xs) >= 40:     # wide enough range of openings to build our own bank
        lat = np.array([tag[k][1][0] for k in ks])
        a, b = np.polyfit(xs, lat, 1)          # pixel x -> lateral offset (mm, camera x)
        temps, centres, offsets = [], [], []
        ih, iw = frames[0][1].shape
        # spread evenly over the whole opening range (percentiles bunch up where the jaw lingers and
        # leave fully closed / wide open uncovered); crops near the image edge are shifted inside it and
        # the tag's position within the template is kept in `offsets`
        for x in np.linspace(xs.min(), xs.max(), 8):
            k = ks[int(np.argmin(np.abs(xs - x)))]
            cx, cy = tag[k][0].astype(int)
            y0 = int(np.clip(cy - H // 2, 0, ih - H)); x0 = int(np.clip(cx - W // 2, 0, iw - W))
            temps.append(frames[k][1][y0:y0 + H, x0:x0 + W]); centres.append((cx, cy))
            offsets.append((cx - x0, cy - y0))
        # the saved bank is replaced only by one covering at least as wide a range of openings (a demo that
        # never fully closes shouldn't drop the closed templates); otherwise the wider saved bank is used
        saved_ptp = float(np.ptp(np.load(bank_file, allow_pickle=True)["centres"][:, 0])) if bank_file.exists() else -1
        own = np.ptp(xs) >= saved_ptp - 5
        if own:
            np.savez(bank_file, temps=np.array(temps, dtype=np.uint8), centres=np.array(centres),
                     offsets=np.array(offsets), px_to_mm=np.array([a, b]), source=_rel(video))
            bank_note = f"built from this capture ({len(ks)} tag frames, {np.ptp(xs):.0f}px range); saved"
    else:
        own = False
    if not own and bank_file.exists():
        z = np.load(bank_file, allow_pickle=True)
        temps = [np.ascontiguousarray(t, dtype=np.uint8) for t in z["temps"]]
        centres, (a, b) = z["centres"], z["px_to_mm"]
        offsets = z["offsets"] if "offsets" in z.files else [(t.shape[1] / 2, t.shape[0] / 2) for t in temps]
        bank_note = (f"from saved bank ({Path(str(z['source'])).name}, {np.ptp(z['centres'][:, 0]):.0f}px range); "
                     f"this capture has {len(ks)} tag frames"
                     + (f" over {np.ptp(xs):.0f}px" if len(ks) else ""))
    elif not own:
        return [{"t": t, "valid": 0, "x_px": np.nan, "lateral_mm": np.nan, "source": ""} for t, _ in frames], \
            {"tag_frames": len(ks), "note": "too few finger-tag sightings to build templates and no saved bank"}
    cxs = np.array([c[0] for c in centres]); cys = np.array([c[1] for c in centres])
    y_lo = int(max(0, cys.min() - H))
    # one template width of slack on the left: enough for a slightly more closed jaw, short of the
    # fixed-jaw tag (which also looks like a white-ringed square)
    x_lo = int(max(0, cxs.min() - W))
    order = np.argsort(cxs)
    path_x, path_y = cxs[order].astype(float), cys[order].astype(float)
    if tag:   # this capture's own sightings trace the path more densely than the 8 template centres
        tx = np.array([tag[k][0][0] for k in ks]); ty = np.array([tag[k][0][1] for k in ks])
        if np.ptp(tx) >= 40:
            fit = np.polyfit(tx, ty, 2)
            path_x = np.linspace(min(tx.min(), cxs.min()) - W, max(tx.max(), cxs.max()) + W, 64)
            path_y = np.polyval(fit, path_x)
    out = []
    for k, (t, g) in enumerate(frames):
        if k in tag:
            out.append({"t": t, "valid": 1, "x_px": float(tag[k][0][0]), "lateral_mm": float(tag[k][1][0]), "source": "tag"})
            continue
        strip = cv2.GaussianBlur(g[y_lo:, x_lo:], (3, 3), 0)
        best = (-1.0, 0.0)
        for tp, off in zip(temps, offsets):
            r = cv2.matchTemplate(strip, tp, cv2.TM_CCOEFF_NORMED)
            # the moving finger has one degree of freedom: its tag stays on one path through the image
            # (y is a function of x). Only accept matches whose tag centre lies on that path.
            cx = np.arange(r.shape[1]) + x_lo + float(off[0])
            cy = np.arange(r.shape[0])[:, None] + y_lo + float(off[1])
            r[np.abs(cy - np.interp(cx, path_x, path_y)[None, :]) > PATH_TOL_PX] = -1.0
            _, v, _, loc = cv2.minMaxLoc(r)
            if v > best[0]:
                best = (v, loc[0] + x_lo + float(off[0]))
        if best[0] > MATCH_MIN:
            out.append({"t": t, "valid": 1, "x_px": best[1], "lateral_mm": a * best[1] + b, "source": "template"})
        else:
            out.append({"t": t, "valid": 0, "x_px": np.nan, "lateral_mm": np.nan, "source": ""})
    # a template lock that jumps away from its neighbours (wood grain, the box edge) is a misread
    xv = np.array([r["x_px"] for r in out], float)
    dropped = 0
    for k, r in enumerate(out):
        if r["source"] != "template":
            continue
        nb = np.r_[xv[max(0, k - 5):k], xv[k + 1:k + 6]]
        nb = nb[np.isfinite(nb)]
        if len(nb) >= 3 and abs(r["x_px"] - np.median(nb)) > 40:
            out[k] = {"t": r["t"], "valid": 0, "x_px": np.nan, "lateral_mm": np.nan, "source": ""}
            dropped += 1
    info = {"tag_frames": len(tag), "tag_frames_relaxed": len(relaxed), "template_jumps_dropped": dropped, "frames": n, "px_to_mm": [float(a), float(b)], "templates": bank_note}
    return out, info


def lateral_to_angle(lat_mm, closed_mm, open_mm, open_rad):
    """Map the finger's lateral offset to a jaw angle, linear between two reference readings."""
    return (np.asarray(lat_mm) - closed_mm) / (open_mm - closed_mm) * open_rad


# ---------------------------------------------------------------- resample + smooth
def interp_series(t_src, values, valid, t_out, max_gap):
    """Linear interpolation over valid samples; NaN where the nearest valid sample is > max_gap away."""
    v = np.asarray(values, float)
    m = np.asarray(valid, bool) & np.all(np.isfinite(v.reshape(len(v), -1)), axis=1)
    ts = np.asarray(t_src)[m]
    vs = v[m]
    if len(ts) < 2:
        return np.full((len(t_out),) + v.shape[1:], np.nan), np.zeros(len(t_out), bool)
    res = np.stack([np.interp(t_out, ts, vs[:, j]) for j in range(vs.shape[1])], 1) if vs.ndim > 1 \
        else np.interp(t_out, ts, vs)
    idx = np.searchsorted(ts, t_out)
    gap = np.minimum(np.abs(t_out - ts[np.clip(idx - 1, 0, len(ts) - 1)]), np.abs(ts[np.clip(idx, 0, len(ts) - 1)] - t_out))
    ok = (gap <= max_gap) & (t_out >= ts[0]) & (t_out <= ts[-1])
    res[~ok] = np.nan
    return res, ok


def reject_outliers(t, P, valid, win_s=0.4, thresh_mm=25.0):
    """Hampel-style filter on the raw pose stream: drop samples that sit far from the median of
    their neighbours in time. Catches single-tag pose flips (IPPE ambiguity) and misreads."""
    P = np.asarray(P, float) * 1000.0
    keep = np.asarray(valid, bool).copy()
    idx = np.where(keep)[0]
    t = np.asarray(t)
    for i in idx:
        nb = idx[(np.abs(t[idx] - t[i]) <= win_s / 2)]
        if len(nb) < 3:
            continue
        med = np.median(P[nb], axis=0)
        if np.linalg.norm(P[i] - med) > thresh_mm:
            keep[i] = False
    return keep


def smooth(x, win):
    """Centered moving average that ignores NaNs (keeps NaNs where there is no data)."""
    if win <= 1:
        return x
    k = np.ones(win)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        v = x[:, j]
        m = np.isfinite(v)
        num = np.convolve(np.where(m, v, 0), k, "same")
        den = np.convolve(m.astype(float), k, "same")
        out[:, j] = np.where(m, num / np.maximum(den, 1e-9), np.nan)
    return out


def estimate_lag(t_pose, p, t_jaw, x_px, t_grid):
    """Delay of the overhead stream vs the wrist camera, from when motion starts/stops: compare
    gripper speed (overhead) with wrist-image motion. Returns seconds to subtract from overhead."""
    # crude but serviceable: cross-correlate normalized speed signals on the grid
    pp, okp = interp_series(t_pose, p, np.isfinite(np.asarray(p)[:, 0]), t_grid, 0.2)
    if okp.sum() < 30:
        return 0.0, 0.0
    sp = np.linalg.norm(np.gradient(np.nan_to_num(pp), axis=0), axis=1)
    sp[~okp] = 0
    xj, okj = interp_series(t_jaw, np.asarray(x_px)[:, None], np.isfinite(x_px), t_grid, 0.2)
    sj = np.abs(np.gradient(np.nan_to_num(xj[:, 0])))
    sj[~okj] = 0
    a = (sp - sp.mean()) / (sp.std() + 1e-9)
    b = (sj - sj.mean()) / (sj.std() + 1e-9)
    lags = range(-15, 16)
    cc = [np.mean(a[max(0, L):len(a) + min(0, L)] * b[max(0, -L):len(b) + min(0, -L)]) for L in lags]
    best = int(np.argmax(cc))
    return list(lags)[best] / 30.0, float(cc[best])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="capture timestamp, e.g. 2026-11-15T00-06-19-417Z (or a path to either file)")
    ap.add_argument("--umi-lens", default=str(REPO / "calibration/results/innomaker_640x480.json"))
    ap.add_argument("--overhead-lens", default=str(REPO / "calibration/overhead_lens_640x480.json"))
    ap.add_argument("--table-pose", default=str(REPO / "calibration/overhead_table_pose.json"),
                    help="saved overhead->table pose; written when tag 13 is visible, read when it isn't")
    ap.add_argument("--jaw-bank", default=str(REPO / "calibration/finger_templates.npz"),
                    help="saved moving-finger templates; rebuilt from captures with good finger-tag coverage")
    ap.add_argument("--overhead-lag", type=float, default=None,
                    help="seconds the overhead stream trails the wrist camera (default: estimate)")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--max-gap", type=float, default=0.25, help="interpolate across gaps up to this many seconds")
    ap.add_argument("--smooth", type=int, default=5, help="moving-average window on the 30 Hz grid (frames)")
    ap.add_argument("--jaw-closed-mm", type=float, default=None, help="finger lateral offset when fully closed")
    ap.add_argument("--jaw-open-mm", type=float, default=None, help="finger lateral offset at --jaw-open-rad")
    ap.add_argument("--jaw-open-rad", type=float, default=None, help="jaw angle at --jaw-open-mm")
    ap.add_argument("--cube-tags", default=None,
                    help="cube tag corners: a cube_calib.py json, or 'cad' for the CAD model "
                         "(default: calibration/cube_tags.json when good, else CAD)")
    ap.add_argument("--out-suffix", default="", help="write to data/episodes/<ts><suffix>/ (for A/B runs)")
    args = ap.parse_args()
    for f in (args.umi_lens, args.overhead_lens):
        if not Path(f).exists():
            raise SystemExit(f"{f} not found. Calibrate your cameras (calibration/README.md), or start from the reference rig's values: cp -r calibration/example/. calibration/")
    cube_src = "calibration/cube_tags.json" if (_CUBE_FILE.exists() and _cube.get("good")) else "cad"
    if args.cube_tags == "cad":
        CUBE_TAG_CORNERS.update(_CAD_TAG_CORNERS); cube_src = "cad"
    elif args.cube_tags:
        CUBE_TAG_CORNERS.update({int(i): np.array(c, float) for i, c in
                                 json.loads(Path(args.cube_tags).read_text())["corners_gripper_mm"].items()})
        cube_src = args.cube_tags

    ts = Path(args.capture).name
    for pre in ("capture-",):
        ts = ts.removeprefix(pre)
    ts = ts.split("-umi-")[0].split("-overhead-")[0].removesuffix("-rig")
    rig_dir = DATA / f"capture-{ts}-rig"          # recorded on the rig PC (viewer/rig/umi_capture.py): one clock, 30 fps
    if (rig_dir / "umi" / "frames.csv").exists() and (rig_dir / "overhead" / "frames.csv").exists():
        umi_vid, oh_vid = rig_dir / "umi", rig_dir / "overhead"
    else:
        umi_vid = oh_vid = None
    umi_vid = umi_vid or next(DATA.glob(f"capture-{ts}-umi-*.mp4"), None) or next(DATA.glob(f"capture-{ts}-umi-*.webm"), None)
    oh_vid = oh_vid or next((p for p in DATA.glob(f"capture-{ts}-overhead-*") if p.is_dir() and (p / "frames.csv").exists()), None) \
        or next(DATA.glob(f"capture-{ts}-overhead-*.mp4"), None) or next(DATA.glob(f"capture-{ts}-overhead-*.webm"), None)
    if not umi_vid or not oh_vid:
        raise SystemExit(f"need both capture-{ts}-umi-* and capture-{ts}-overhead-* in {DATA}")
    out = DATA / "episodes" / f"{ts}{args.out_suffix}"
    out.mkdir(parents=True, exist_ok=True)

    det = make_detector()
    umi_cam, oh_cam = Camera(Path(args.umi_lens)), Camera(Path(args.overhead_lens))
    print(f"[1/4] overhead: tracking the cube in {oh_vid.name}")
    poses, table_src = track_cube(oh_vid, oh_cam, det, Path(args.table_pose))
    print(f"      table frame from {table_src}")
    print(f"[2/4] wrist: tracking the moving jaw in {umi_vid.name}")
    jaw, jaw_info = track_jaw(umi_vid, umi_cam, det, Path(args.jaw_bank))
    print(f"      finger templates {jaw_info.get('templates', jaw_info.get('note'))}")

    # raw per-source CSVs
    with open(out / "gripper_pose.csv", "w") as f:
        f.write("t,valid,faces,x,y,z,qx,qy,qz,qw,tip_x,tip_y,tip_z,reproj_px\n")
        for r in poses:
            f.write(",".join([f"{r['t']:.4f}", str(r["valid"]), r["faces"]] +
                             [f"{v:.5f}" for v in (*r["p"], *r["q"], *r["tip"])] + [f"{r['err_px']:.2f}"]) + "\n")
    with open(out / "jaw.csv", "w") as f:
        f.write("t,valid,source,finger_x_px,lateral_mm\n")
        for r in jaw:
            f.write(f"{r['t']:.4f},{r['valid']},{r['source']},{r['x_px']:.2f},{r['lateral_mm']:.3f}\n")

    # common clock (wrist camera time), overhead shifted by its lag
    t_pose = np.array([r["t"] for r in poses])
    t_jaw = np.array([r["t"] for r in jaw])
    t_end = min(t_pose.max(), t_jaw.max())
    grid = np.arange(0.0, t_end, 1.0 / args.hz)
    P = np.array([r["p"] for r in poses], float)
    if args.overhead_lag is None:
        lag, corr = estimate_lag(t_pose, P, t_jaw, np.array([r["x_px"] for r in jaw], float), grid)
        if corr >= 0.3:
            lag_note = f"estimated {lag:+.3f}s (corr {corr:.2f})"
        else:
            lag_note = f"estimate unreliable ({lag:+.3f}s, corr {corr:.2f}); using 0. Pass --overhead-lag to set it"
            lag = 0.0
    else:
        lag, lag_note = args.overhead_lag, "given"
    print(f"[3/4] resampling at {args.hz:g} Hz; overhead lag {lag_note}")
    tp = t_pose - lag
    valid_p = np.array([r["valid"] for r in poses], bool)
    n_raw = int(valid_p.sum())
    valid_p = reject_outliers(t_pose, np.array([r['tip'] for r in poses], float), valid_p)
    n_outliers = n_raw - int(valid_p.sum())
    print(f"      rejected {n_outliers} outlier poses (pose flips / misreads)")
    pos, okp = interp_series(tp, P, valid_p, grid, args.max_gap)
    tip, _ = interp_series(tp, np.array([r["tip"] for r in poses], float), valid_p, grid, args.max_gap)
    Q = np.array([r["q"] for r in poses], float)
    for i in range(1, len(Q)):  # keep quaternions on one hemisphere before interpolating
        if np.all(np.isfinite(Q[i])) and np.all(np.isfinite(Q[i - 1])) and np.dot(Q[i], Q[i - 1]) < 0:
            Q[i] = -Q[i]
    quat, _ = interp_series(tp, Q, valid_p, grid, args.max_gap)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    lat, okj = interp_series(t_jaw, np.array([[r["lateral_mm"]] for r in jaw], float),
                             np.array([r["valid"] for r in jaw], bool), grid, args.max_gap)
    pos, tip, quat = smooth(pos, args.smooth), smooth(tip, args.smooth), smooth(quat, args.smooth)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    lat = smooth(lat, args.smooth)[:, 0]

    closed = args.jaw_closed_mm if args.jaw_closed_mm is not None else float(np.nanpercentile(lat, 2))
    if args.jaw_open_mm is not None and args.jaw_open_rad is not None:
        jaw_rad = lateral_to_angle(lat, closed, args.jaw_open_mm, args.jaw_open_rad)
        jaw_note = "calibrated from --jaw-* references"
    else:
        jaw_rad = np.full_like(lat, np.nan)
        jaw_note = "uncalibrated: jaw_opening_mm is relative to the most-closed reading; pass --jaw-* to get radians"

    with open(out / "trajectory.csv", "w") as f:
        f.write("t,pose_valid,x,y,z,qx,qy,qz,qw,tip_x,tip_y,tip_z,jaw_valid,jaw_opening_mm,jaw_rad\n")
        for i, t in enumerate(grid):
            f.write(",".join([f"{t:.4f}", str(int(okp[i]))] + [f"{v:.5f}" for v in (*pos[i], *quat[i], *tip[i])] +
                             [str(int(okj[i])), f"{lat[i] - closed:.3f}", f"{jaw_rad[i]:.4f}"]) + "\n")

    vp = valid_p.mean() if len(valid_p) else 0
    meta = {
        "capture": ts, "umi_video": str(umi_vid.relative_to(REPO)), "overhead_video": str(oh_vid.relative_to(REPO)),
        "umi_lens": _rel(args.umi_lens), "overhead_lens": _rel(args.overhead_lens), "cube_tags": cube_src, "table_frame": table_src,
        "overhead_lag_s": lag, "overhead_lag_note": lag_note, "hz": args.hz, "max_gap_s": args.max_gap,
        "smooth_frames": args.smooth,
        "coverage": {"overhead_frames": len(poses), "cube_pose_frames": int(valid_p.sum()), "cube_pose_pct": round(100 * vp, 1), "outliers_rejected": n_outliers,
                     "wrist_frames": len(jaw), "jaw_frames": int(sum(r["valid"] for r in jaw)),
                     "jaw_from_tag": int(sum(r["source"] == "tag" for r in jaw)),
                     "grid_pose_pct": round(100 * okp.mean(), 1), "grid_jaw_pct": round(100 * okj.mean(), 1)},
        "jaw": {**jaw_info, "closed_lateral_mm": closed, "note": jaw_note},
        "frames": {"pose": "gripper body (SO-101 MJCF 'gripper') in table frame (tag 13), metres, quat xyzw",
                   "tip": "jaw tip point (cad/src/asm.py) in table frame, metres"},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))

    print("[4/4] preview")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for j, n in enumerate("xyz"):
            ax[0].plot(grid, tip[:, j] * 1000, label=f"tip {n}")
        ax[0].set_ylabel("jaw tip, table frame (mm)"); ax[0].legend(loc="upper right"); ax[0].grid(alpha=.3)
        ax[1].plot(grid, lat - closed, color="#c0392b")
        ax[1].set_ylabel("jaw opening proxy (mm)"); ax[1].grid(alpha=.3)
        ax[2].plot(grid, okp.astype(int) + 0.02, label="pose")
        ax[2].plot(grid, okj.astype(int) - 0.02, label="jaw")
        ax[2].set_ylabel("valid"); ax[2].set_xlabel("t (s, wrist-camera clock)"); ax[2].legend(); ax[2].set_ylim(-.2, 1.2)
        fig.suptitle(f"SO-UMI episode {ts}"); fig.tight_layout(); fig.savefig(out / "preview.png", dpi=110)
    except ImportError:
        pass
    c = meta["coverage"]
    print(f"\nWrote {out.relative_to(REPO)}/  (trajectory.csv, gripper_pose.csv, jaw.csv, meta.json, preview.png)")
    print(f"  cube pose: {c['cube_pose_pct']}% of overhead frames;  jaw: {c['jaw_frames']}/{c['wrist_frames']} wrist frames "
          f"({c['jaw_from_tag']} from the tag)")
    print(f"  on the {args.hz:g} Hz grid: pose {c['grid_pose_pct']}%, jaw {c['grid_jaw_pct']}%")


if __name__ == "__main__":
    main()
