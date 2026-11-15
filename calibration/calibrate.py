#!/usr/bin/env python3
"""Calibrate camera intrinsics from a checkerboard video.

Fits both a pinhole (OpenCV radial-tangential) and a fisheye (Kannala-Brandt)
model, picks the better one, and writes an ORB-SLAM3 settings YAML plus a JSON
report and visual check images.

Usage:
  uv run --with "opencv-python<5" --with numpy calibration/calibrate.py data/capture.mp4 \
      [--cols 9 --rows 6 --square-mm 25 --max-views 40 --name innomaker]

Intrinsics are only valid for the resolution the video was recorded at.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def detect(video_path, pattern, max_candidates, min_sharpness):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, total // max_candidates) if total else 5
    found, idx, scanned = [], 0, 0
    size = None
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            scanned += 1
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            ok, corners = cv2.findChessboardCornersSB(
                gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
            )
            if ok and sharp >= min_sharpness:
                found.append({"frame": idx, "corners": corners.reshape(-1, 2), "sharpness": sharp, "image": frame})
        idx += 1
    cap.release()
    return found, size, fps, scanned


def descriptor(corners, size):
    """Position, scale and tilt of the board in the image, normalized."""
    w, h = size
    c = corners.mean(axis=0) / [w, h]
    hull = cv2.convexHull(corners.astype(np.float32))
    area = cv2.contourArea(hull) / (w * h)
    x0, y0, x1, y1 = corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()
    aspect = (x1 - x0) / max(1.0, (y1 - y0))
    return np.array([c[0], c[1], np.sqrt(area) * 2, np.log(aspect) * 0.5])


def select_diverse(found, size, max_views):
    """Greedy farthest-point selection over board pose descriptors."""
    if len(found) <= max_views:
        return found
    desc = np.array([descriptor(f["corners"], size) for f in found])
    chosen = [int(np.argmax([f["sharpness"] for f in found]))]
    dmin = np.linalg.norm(desc - desc[chosen[0]], axis=1)
    while len(chosen) < max_views:
        i = int(np.argmax(dmin))
        chosen.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(desc - desc[i], axis=1))
    return [found[i] for i in sorted(chosen)]


def calibrate_pinhole(obj, img, size, flags=0):
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    errs = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, r, t, K, D)
        errs.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    return rms, K, D.ravel()[:5], errs


def calibrate_fisheye(obj, img, size):
    obj_f = [o.reshape(-1, 1, 3).astype(np.float64) for o in obj]
    img_f = [i.reshape(-1, 1, 2).astype(np.float64) for i in img]
    # Constant location differs across OpenCV versions:
    # 4.x: cv2.fisheye.X or cv2.fisheye_X; 5.x: cv2.X.
    def const(name):
        for ns, attr in ((cv2.fisheye, name), (cv2, f"fisheye_{name}"), (cv2, name)):
            if hasattr(ns, attr):
                return getattr(ns, attr)
        raise AttributeError(name)

    flags = const("CALIB_RECOMPUTE_EXTRINSIC") | const("CALIB_FIX_SKEW")
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(obj_f, img_f, size, None, None, flags=flags, criteria=crit)
    errs = []
    for o, i, r, t in zip(obj_f, img_f, rvecs, tvecs):
        p, _ = cv2.fisheye.projectPoints(o, r, t, K, D)
        errs.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    return rms, K, D.ravel(), errs


def calibrate_pinhole_no_tangent(obj, img, size):
    return calibrate_pinhole(obj, img, size, flags=cv2.CALIB_ZERO_TANGENT_DIST)


def heldout_error(model, K, D, views, objp):
    """Median / p90 reprojection error on views NOT used for fitting.

    Each view's board pose is solved with the candidate intrinsics, so this
    measures how well the intrinsics generalize rather than how well they fit.
    """
    errs = []
    for v in views:
        c = v["corners"].astype(np.float64).reshape(-1, 1, 2)
        if model == "fisheye":
            und = cv2.fisheye.undistortPoints(c, K, D)
        else:
            und = cv2.undistortPoints(c, K, D)
        ok, r, t = cv2.solvePnP(objp, und, np.eye(3), None)
        if not ok:
            continue
        if model == "fisheye":
            p, _ = cv2.fisheye.projectPoints(objp.reshape(-1, 1, 3).astype(np.float64), r, t, K, D)
        else:
            p, _ = cv2.projectPoints(objp, r, t, K, D)
        errs.append(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - c.reshape(-1, 2)) ** 2, axis=1))))
    if not errs:
        return float("inf"), float("inf")
    return float(np.median(errs)), float(np.percentile(errs, 90))


def robust(fn, obj, img, size, label):
    """Calibrate, drop views with error > 3x median, recalibrate."""
    rms, K, D, errs = fn(obj, img, size)
    med = float(np.median(errs))
    keep = [i for i, e in enumerate(errs) if e <= max(3 * med, 0.5)]
    dropped = len(errs) - len(keep)
    if dropped and len(keep) >= 10:
        rms, K, D, errs = fn([obj[i] for i in keep], [img[i] for i in keep], size)
    else:
        keep = list(range(len(errs)))
    print(f"  {label:9s} RMS {rms:.3f}px  views {len(keep)} (dropped {len(errs) + dropped - len(keep) if False else dropped})")
    return {"rms": float(rms), "K": K, "D": D, "per_view": errs, "kept": keep}


def write_orbslam_yaml(path, model, K, D, size, fps):
    w, h = size
    lines = ["%YAML:1.0", "", 'File.version: "1.0"', ""]
    if model == "pinhole":
        lines += ['Camera.type: "PinHole"', ""]
    else:
        lines += ['Camera.type: "KannalaBrandt8"', ""]
    lines += [f"Camera1.fx: {K[0, 0]:.6f}", f"Camera1.fy: {K[1, 1]:.6f}",
              f"Camera1.cx: {K[0, 2]:.6f}", f"Camera1.cy: {K[1, 2]:.6f}", ""]
    if model == "pinhole":
        k1, k2, p1, p2, k3 = D
        lines += [f"Camera1.k1: {k1:.8f}", f"Camera1.k2: {k2:.8f}", f"Camera1.p1: {p1:.8f}",
                  f"Camera1.p2: {p2:.8f}", f"Camera1.k3: {k3:.8f}"]
    else:
        lines += [f"Camera1.k{i + 1}: {d:.8f}" for i, d in enumerate(D)]
    lines += ["", f"Camera.width: {w}", f"Camera.height: {h}", f"Camera.fps: {round(fps)}", "Camera.RGB: 1", "",
              "ORBextractor.nFeatures: 1500", "ORBextractor.scaleFactor: 1.2", "ORBextractor.nLevels: 8",
              "ORBextractor.iniThFAST: 20", "ORBextractor.minThFAST: 7", "",
              "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9",
              "Viewer.PointSize: 2.0", "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0",
              "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7", "Viewer.ViewpointZ: -1.8",
              "Viewer.ViewpointF: 500.0", ""]
    path.write_text("\n".join(lines))


def coverage_image(views, size, path):
    w, h = size
    canvas = np.full((h, w, 3), 255, np.uint8)
    for v in views:
        hull = cv2.convexHull(v["corners"].astype(np.float32)).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [hull], (200, 120, 40))
        canvas = cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0)
        for x, y in v["corners"]:
            cv2.circle(canvas, (int(x), int(y)), 2, (40, 40, 200), -1)
    cv2.imwrite(str(path), canvas)


def undistort_image(frame, model, K, D, path):
    h, w = frame.shape[:2]
    if model == "pinhole":
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
        out = cv2.undistort(frame, K, D, None, newK)
    else:
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (w, h), cv2.CV_16SC2)
        out = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)
    cv2.imwrite(str(path), np.hstack([frame, out]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="one or more checkerboard videos (same camera + resolution)")
    ap.add_argument("--cols", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--max-views", type=int, default=40)
    ap.add_argument("--candidates", type=int, default=400, help="frames to sample from the video")
    ap.add_argument("--min-sharpness", type=float, default=30.0, help="Laplacian variance threshold")
    ap.add_argument("--name", default="innomaker")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    found, size, fps, scanned = [], None, None, 0
    for vi, path in enumerate(args.videos):
        print(f"Scanning {path} for {args.cols}x{args.rows} inner corners…")
        f, s, r, n = detect(path, pattern, args.candidates, args.min_sharpness)
        print(f"  board found in {len(f)}/{n} sampled frames, image {s[0]}x{s[1]}")
        if size is not None and s != size:
            raise SystemExit(f"{path} is {s[0]}x{s[1]} but earlier videos are {size[0]}x{size[1]}; "
                             "calibrate each resolution separately.")
        size, fps, scanned = s, fps or r, scanned + n
        for d in f:
            d["video"] = vi
        found += f
    if len(found) < 10:
        raise SystemExit("Too few detections (<10). Check --cols/--rows, lighting, and focus.")

    views = select_diverse(found, size, args.max_views)
    chosen_ids = {id(v) for v in views}
    holdout = [v for v in found if id(v) not in chosen_ids] or found
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * (args.square_mm / 1000.0)
    obj = [objp] * len(views)
    img = [v["corners"].astype(np.float32) for v in views]
    print(f"Calibrating with {len(views)} diverse views; validating on {len(holdout)} held-out views…")

    candidates = [("pinhole", "pinhole", calibrate_pinhole),
                  ("pinhole_no_tangent", "pinhole", calibrate_pinhole_no_tangent),
                  ("fisheye", "fisheye", calibrate_fisheye)]
    results = {}
    for name, kind, fn in candidates:
        try:
            r = robust(fn, obj, img, size, name)
        except Exception as e:  # fisheye fit can diverge with few/poor views
            print(f"  {name} failed: {e}")
            continue
        r["kind"] = kind
        r["heldout_median"], r["heldout_p90"] = heldout_error(kind, r["K"], r["D"], holdout, objp)
        print(f"  {'':9s} held-out median {r['heldout_median']:.3f}px  p90 {r['heldout_p90']:.3f}px")
        results[name] = r

    # Pick by held-out error (generalization), not by in-sample RMS, which
    # rewards overfitting when parts of the image have no board coverage.
    model_name = min(results, key=lambda m: (results[m]["heldout_median"], results[m]["heldout_p90"]))
    best = results[model_name]
    model = best["kind"]
    K, D = best["K"], best["D"]
    if model == "pinhole":
        D = np.asarray(D, dtype=np.float64).ravel()[:5]
    w, h = size
    hfov = np.degrees(2 * np.arctan(w / (2 * K[0, 0])))
    vfov = np.degrees(2 * np.arctan(h / (2 * K[1, 1])))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_{w}x{h}"
    yaml_path = out / f"{stem}_orbslam3.yaml"
    write_orbslam_yaml(yaml_path, model, K, D, size, fps)
    report = {
        "videos": [str(v) for v in args.videos],
        "image_size": [w, h],
        "fps": fps,
        "pattern_inner_corners": list(pattern),
        "square_mm": args.square_mm,
        "frames_sampled": scanned,
        "frames_with_board": len(found),
        "views_used": len(best["kept"]),
        "model": model,
        "model_variant": model_name,
        "rms_px": best["rms"],
        "heldout_median_px": best["heldout_median"],
        "heldout_p90_px": best["heldout_p90"],
        "K": K.tolist(),
        "D": D.tolist(),
        "approx_fov_deg": {"horizontal": round(float(hfov), 1), "vertical": round(float(vfov), 1)},
        "all_models": {m: {"rms_px": r["rms"], "heldout_median_px": r["heldout_median"],
                           "heldout_p90_px": r["heldout_p90"], "K": r["K"].tolist(),
                           "D": np.asarray(r["D"]).ravel().tolist()} for m, r in results.items()},
        "source_frames": [[views[i]["video"], views[i]["frame"]] for i in best["kept"]],
    }
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2))
    coverage_image([views[i] for i in best["kept"]], size, out / f"{stem}_coverage.png")
    sample = max(views, key=lambda v: v["sharpness"])["image"]
    undistort_image(sample, model, K, D, out / f"{stem}_undistort_check.png")

    print(f"\nModel: {model_name}   RMS {best['rms']:.3f} px   "
          f"held-out median {best['heldout_median']:.3f} px / p90 {best['heldout_p90']:.3f} px")
    print(f"fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}")
    print(f"D={np.array2string(D, precision=5)}")
    print(f"Approx FOV: {hfov:.1f}° x {vfov:.1f}°")
    print(f"\nWrote:\n  {yaml_path}\n  {out / (stem + '.json')}\n  {out / (stem + '_coverage.png')}\n  {out / (stem + '_undistort_check.png')}")


if __name__ == "__main__":
    main()
