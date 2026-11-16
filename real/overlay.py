"""Overlap video of a handheld demo and the real robot replaying it, both filmed by the rig's overhead camera.

    uv run --python 3.12 --with mujoco==3.15.0 --with gymnasium==1.4.0 --with numpy --with pillow \\
        --with "opencv-python<5" --with matplotlib real/overlay.py <episode> --side right \\
        --recording data/<robot-run-capture>-rig --log <umi_replay log.json>

Paths (pi_sim world, projected through the calibrated overhead lens, so they land on the real pixels):
  DEMO   (green)   where the handheld's tool point was: trajectory.csv -> world, as umi/to_robot.py does
  PLAN   (grey)    where the plan puts the robot's TCP (sim IK on plan_<side>.json): DEMO minus IK error
  ROBOT  (magenta) where the real arm's TCP was: forward kinematics of the joints it measured during the run
Time: the replay ran the plan `scale` times slower; robot tick k <-> plan time k / hz / scale.

Writes data/episodes/<episode>/robotrun_<side>_<recording>.mp4 (left: robot run, right: demo, same paths drawn on
both) and .png (top view + height over time + distance from the demo).

What the arm sees: when the robot run also filmed the robot's wrist camera (<recording>/robot_wrist/, recorded by
viewer/rig/umi_capture.py --robot-wrist), it also writes robotrun_<side>_<recording>_wrist.mp4:
    UMI demo wrist | robot wrist | 50/50 overlay
on the same clock as the overhead video (robot time / scale = demo time), like the UMI-vs-sim video. Same camera
model on both, so when the robot follows the demo the two views line up in the overlay.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "umi"))
import to_robot as TR  # noqa: E402
from convert import read_frames  # noqa: E402

GREEN, GREY, MAGENTA = (60, 220, 60), (200, 200, 200), (255, 60, 255)


def demo_tcp(ep: Path, lens: Path, table_pose: Path):
    """Handheld TCP in the world for the plan's frames (same span, gap filling and tip correction as to_robot)."""
    tr = np.genfromtxt(ep / "trajectory.csv", delimiter=",", names=True)
    t, ok = tr["t"], tr["pose_valid"].astype(bool)
    T_wt, _ = TR.table_to_world(lens, table_pose)
    idx = np.where(ok)[0]
    sel = np.arange(idx[0], idx[-1] + 1)
    P = np.stack([np.interp(sel, idx, tr[k][idx]) for k in ("x", "y", "z")], 1)
    Q = np.stack([np.interp(sel, idx, tr[k][idx]) for k in ("qx", "qy", "qz", "qw")], 1)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    tip = np.zeros(3)
    tf = REPO / "calibration" / "handheld_tip.json"
    if tf.exists():
        tj = json.loads(tf.read_text())
        if tj.get("good"):
            tip = (np.array(tj["tip_in_gripper_mm"]) - np.array(tj["cad_tip_mm"])) / 1000.0
    out = []
    for p, (x, y, z, w) in zip(P, Q):
        R_g = T_wt[:3, :3] @ TR.quat_wxyz_to_mat((w, x, y, z))
        out.append(T_wt[:3, :3] @ p + T_wt[:3, 3] + R_g @ tip + R_g @ TR.TCP_IN_GRIPPER)
    return t[sel] - t[sel[0]], np.array(out), float(t[sel[0]])


def fk(arm, Q):
    out = []
    for q in Q:
        arm.set(np.asarray(q, float)[:5])
        out.append(arm.tcp_world())
    return np.array(out)


class Projector:
    def __init__(self, lens_file: Path):
        lens = json.loads(lens_file.read_text())
        self.K, self.D = np.array(lens["K"], float), np.array(lens["D"], float)
        wp = lens["world_pose"]
        R_wc = TR.quat_wxyz_to_mat(wp["quat_wxyz"]) @ np.diag([1, -1, -1])     # OpenCV camera -> world
        self.R_cw, self.pos = R_wc.T, np.array(wp["pos"], float)

    def __call__(self, P):
        Pc = (np.asarray(P, float) - self.pos) @ self.R_cw.T
        rvec, _ = cv2.Rodrigues(np.eye(3))
        uv, _ = cv2.projectPoints(Pc.reshape(-1, 1, 3), rvec, np.zeros(3), self.K, self.D)
        return uv.reshape(-1, 2)


def polyline(img, uv, color, thick, upto=None):
    pts = np.round(uv[:upto] if upto is not None else uv).astype(np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA)


def draw_paths(img, uv_demo, uv_plan, uv_robot, k_demo, k_robot):
    faint = img.copy()
    polyline(faint, uv_demo, GREEN, 1)
    polyline(faint, uv_robot, MAGENTA, 1)
    cv2.addWeighted(faint, 0.5, img, 0.5, 0, img)
    polyline(img, uv_plan, GREY, 1, k_demo + 1)
    polyline(img, uv_demo, GREEN, 2, k_demo + 1)
    if k_robot is not None:
        polyline(img, uv_robot, MAGENTA, 2, k_robot + 1)
        cv2.circle(img, tuple(np.round(uv_robot[k_robot]).astype(int)), 6, MAGENTA, 2, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(uv_demo[k_demo]).astype(int)), 6, GREEN, 2, cv2.LINE_AA)


def frame_index(cam_dir: Path):
    """(t, path) of a recorded camera folder (frames.csv: index,t,t_unix), read lazily (no frames in memory)."""
    rows = (cam_dir / "frames.csv").read_text().splitlines()[1:]
    out = [(float(r.split(",")[1]), float(r.split(",")[2]) if len(r.split(",")) > 2 else None,
            cam_dir / "frames" / f"{int(r.split(',')[0]):06d}.jpg") for r in rows if r.strip()]
    return out


def write_wrist_video(out_mp4, demo_cam: Path, demo_t0: float, robot_cam: Path, t0_unix, scale, t_demo_end,
                      t_rob, dist_mm, side):
    """UMI demo wrist | robot wrist | 50/50 overlay, one row per robot wrist frame."""
    dem = frame_index(demo_cam)
    rob = frame_index(robot_cam)
    if not dem or not rob:
        return None
    dem_t = np.array([d[0] for d in dem]) - demo_t0               # demo time (s since the plan's first frame)
    W, H = 640, 480
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"avc1"), 30, (3 * W, H))
    if not vw.isOpened():
        vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30, (3 * W, H))
    last_k = None
    demo_img = np.zeros((H, W, 3), np.uint8)
    for _, tu, path in rob:
        rimg = cv2.imread(str(path))
        if rimg is None or tu is None:
            continue
        rimg = cv2.resize(rimg, (W, H)) if rimg.shape[:2] != (H, W) else rimg
        tr_ = tu - t0_unix                                         # seconds into the replay
        tp = float(np.clip(tr_ / scale, 0.0, t_demo_end))          # demo time shown on the left
        k = int(np.clip(np.searchsorted(dem_t, tp), 0, len(dem) - 1))
        if k != last_k:
            im = cv2.imread(str(dem[k][2]))
            if im is not None:
                demo_img = cv2.resize(im, (W, H)) if im.shape[:2] != (H, W) else im
            last_k = k
        d_img, r_img = demo_img.copy(), rimg.copy()
        blend = cv2.addWeighted(demo_img, 0.5, rimg, 0.5, 0)
        k_rob = int(np.searchsorted(t_rob, tr_, side="right")) - 1
        phase = "replay" if 0 <= tr_ <= t_rob[-1] else ("before replay" if tr_ < 0 else "after replay (fold)")
        d_txt = f"robot TCP {dist_mm[k_rob]:.0f} mm from the demo" if 0 <= k_rob < len(dist_mm) and phase == "replay" else phase
        legend(d_img, f"UMI DEMO wrist  t={tp:5.1f}s", [("what the handheld saw", GREEN)])
        legend(r_img, f"ROBOT {side} wrist  t={tr_:6.1f}s (x{scale:.2f} slower)", [(d_txt, MAGENTA)])
        legend(blend, "OVERLAY 50/50", [("lined up = the robot sees what you saw", (255, 255, 255))])
        vw.write(np.hstack([d_img, r_img, blend]))
    vw.release()
    return out_mp4


def legend(img, title, lines):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24 + 18 * len(lines)), (0, 0, 0), -1)
    cv2.putText(img, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(img, txt, (8, 36 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("--side", required=True, choices=("left", "right"))
    ap.add_argument("--recording", required=True, help="data/<capture>-rig folder of the robot run (overhead/)")
    ap.add_argument("--log", required=True, help="the replay's log.json (C:\\robot\\real\\runs\\umi_replay_*)")
    ap.add_argument("--lens", default=str(REPO / "calibration/overhead_lens_640x480.json"))
    ap.add_argument("--table-pose", default=str(REPO / "calibration/overhead_table_pose.json"))
    a = ap.parse_args()

    ep = REPO / "data" / "episodes" / a.episode
    rec = Path(a.recording)
    if not rec.is_absolute():
        rec = REPO / rec
    plan = json.loads((ep / f"plan_{a.side}.json").read_text())
    log = json.loads(Path(a.log).read_text())
    if log.get("t0_unix") is None or not log.get("ticks"):
        raise SystemExit(f"{a.log} has no live replay ticks (did the arm start replaying?)")
    scale, hz, t0_unix = float(log["scale"]), float(log["hz"]), float(log["t0_unix"])

    t_demo, P_demo, t_demo_start = demo_tcp(ep, Path(a.lens), Path(a.table_pose))
    arm = TR.ArmModel(a.side)
    P_plan = fk(arm, plan["q"])
    ticks = log["ticks"]
    t_rob = np.array([r["t"] for r in ticks])                     # seconds since t0 (rig clock)
    P_rob = fk(arm, [r["units"] for r in ticks])
    plan_time_of_tick = t_rob / scale
    # demo point at each robot tick's plan time, and the distance between them
    D_at = np.stack([np.interp(plan_time_of_tick, t_demo, P_demo[:, j]) for j in range(3)], 1)
    dist_mm = np.linalg.norm(P_rob - D_at, axis=1) * 1000
    P_at = np.stack([np.interp(plan_time_of_tick, np.asarray(plan["t"]), P_plan[:, j]) for j in range(3)], 1)
    track_mm = np.linalg.norm(P_rob - P_at, axis=1) * 1000
    stats = {"robot_vs_demo_mm": {"median": round(float(np.median(dist_mm)), 1),
                                  "p95": round(float(np.percentile(dist_mm, 95)), 1), "max": round(float(dist_mm.max()), 1)},
             "robot_vs_plan_mm": {"median": round(float(np.median(track_mm)), 1),
                                  "p95": round(float(np.percentile(track_mm, 95)), 1)},
             "ticks": len(ticks), "scale": scale}
    print(json.dumps(stats))

    proj = Projector(Path(a.lens))
    uv_demo, uv_plan, uv_rob = proj(P_demo), proj(P_plan), proj(P_rob)

    # demo overhead frames (the original capture), indexed by demo time
    meta = json.loads((ep / "meta.json").read_text())
    demo_frames = list(read_frames(REPO / meta["overhead_video"]))
    demo_ft = np.array([f[0] for f in demo_frames]) - t_demo_start

    rows = (rec / "overhead" / "frames.csv").read_text().splitlines()[1:]
    frames = [(int(r.split(",")[0]), float(r.split(",")[2])) for r in rows]
    name = rec.name.removesuffix("-rig")
    out_mp4 = ep / f"robotrun_{a.side}_{name}.mp4"
    W, H = 640, 480
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"avc1"), 30, (2 * W, H))
    if not vw.isOpened():
        vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30, (2 * W, H))
    for i, tu in frames:
        img = cv2.imread(str(rec / "overhead" / "frames" / f"{i:06d}.jpg"))
        if img is None:
            continue
        tr_ = tu - t0_unix                                         # seconds into the replay (negative: approach)
        k_rob = int(np.searchsorted(t_rob, tr_, side="right")) - 1
        k_rob = None if k_rob < 0 else min(k_rob, len(t_rob) - 1)
        tp = float(np.clip(tr_ / scale, 0.0, t_demo[-1]))          # plan / demo time
        k_dem = int(np.clip(np.searchsorted(t_demo, tp), 0, len(t_demo) - 1))
        demo_img = demo_frames[int(np.clip(np.searchsorted(demo_ft, tp), 0, len(demo_frames) - 1))][1].copy()
        for im in (img, demo_img):
            draw_paths(im, uv_demo, uv_plan, uv_rob, k_dem, k_rob)
        d_txt = f"robot-demo {dist_mm[k_rob]:.0f} mm" if k_rob is not None else "approaching the start pose"
        phase = "replay" if 0 <= tr_ <= t_rob[-1] else ("before replay" if tr_ < 0 else "after replay (fold)")
        legend(img, f"ROBOT run  t={tr_:6.1f}s ({phase})",
               [("magenta: robot TCP (measured joints)", MAGENTA), ("green: demo tool point", GREEN), (d_txt, (255, 255, 255))])
        legend(demo_img, f"DEMO (handheld)  t={tp:5.1f}s  (robot runs x{scale:.2f} slower)",
               [("green: demo tool point", GREEN), ("grey: plan TCP (sim IK)", GREY), ("magenta: robot TCP", MAGENTA)])
        vw.write(np.hstack([img, demo_img]))
    vw.release()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    c = {"demo": "#2ca02c", "plan": "#888888", "robot": "#d62bd6"}
    ax[0].plot(P_demo[:, 0] * 100, P_demo[:, 1] * 100, c=c["demo"], label="demo (handheld)")
    ax[0].plot(P_plan[:, 0] * 100, P_plan[:, 1] * 100, c=c["plan"], ls="--", label="plan (sim IK)")
    ax[0].plot(P_rob[:, 0] * 100, P_rob[:, 1] * 100, c=c["robot"], label="robot (measured)")
    ax[0].set(xlabel="x (cm)", ylabel="y (cm)", title="top view (pi_sim world)", aspect="equal")
    ax[0].legend()
    ax[1].plot(t_demo, P_demo[:, 2] * 100, c=c["demo"], label="demo")
    ax[1].plot(plan["t"], P_plan[:, 2] * 100, c=c["plan"], ls="--", label="plan")
    ax[1].plot(plan_time_of_tick, P_rob[:, 2] * 100, c=c["robot"], label="robot")
    ax[1].set(xlabel="demo time (s)", ylabel="TCP height (cm)", title="height")
    ax[1].legend()
    ax[2].plot(plan_time_of_tick, dist_mm, c=c["robot"], label="robot vs demo")
    ax[2].plot(plan_time_of_tick, track_mm, c=c["plan"], label="robot vs plan (tracking)")
    ax[2].set(xlabel="demo time (s)", ylabel="mm", title=f"distance: median {stats['robot_vs_demo_mm']['median']} mm "
              f"from the demo, {stats['robot_vs_plan_mm']['median']} mm from the plan")
    ax[2].legend()
    for x in ax:
        x.grid(alpha=0.3)
    fig.suptitle(f"{a.episode} {a.side}: demo vs real robot ({name})")
    fig.tight_layout()
    robot_cam = rec / "robot_wrist"
    demo_cam = REPO / meta.get("umi_video", "")
    if (robot_cam / "frames.csv").exists() and meta.get("umi_video") and (demo_cam / "frames.csv").exists():
        w = write_wrist_video(out_mp4.with_name(out_mp4.stem + "_wrist.mp4"), demo_cam, t_demo_start, robot_cam,
                              t0_unix, scale, float(t_demo[-1]), t_rob, dist_mm, a.side)
        if w:
            print(f"wrote {w.relative_to(REPO)} (what the arm sees: demo wrist | robot wrist | overlay)")
    else:
        print("no wrist comparison: this robot run has no robot_wrist recording (restart viewer/rig/umi_capture.py "
              "with --robot-wrist) or the demo has no umi folder")
    out_png = out_mp4.with_suffix(".png")
    fig.savefig(out_png, dpi=90)
    (out_mp4.with_suffix(".json")).write_text(json.dumps(stats, indent=1))
    print(f"wrote {out_mp4.relative_to(REPO)} and {out_png.relative_to(REPO)}")


if __name__ == "__main__":
    main()
