"""Export a plan for the in-browser replay (viewer/replay.html): no video rendering, no encoding.

Writes
  data/web/meshes.bin + meshes.json      the bench twin's visible meshes, once (shared by every episode)
  data/episodes/<ts>/web/scene.json      visible parts (type, size, colour, mesh), which ones move, the
                                         calibrated overhead camera, your hand's path vs the robot's path, checks
  data/episodes/<ts>/web/poses.bin       float32 [frames, moving parts, 7]: position (m) + quaternion (w, x, y, z)
  data/episodes/<ts>/web/overhead.bin/.csv   undistorted 320 px overhead preview frames (JPEG pack)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "data" / "web"


def _mat2quat(mj, xmat):
    q = np.zeros(4)
    mj.mju_mat2Quat(q, np.ascontiguousarray(xmat, dtype=np.float64).ravel())
    return q


def visible_geoms(m):
    return [g for g in range(m.ngeom) if m.geom_group[g] <= 2 and m.geom_rgba[g][3] > 0]


def export_meshes(arm):
    """Write the visible meshes once; reuse if they're already there for this model."""
    m, mj = arm.m, arm.mj
    vis = visible_geoms(m)
    mesh_ids = sorted({int(m.geom_dataid[g]) for g in vis if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH})
    sig = f"{m.nmesh}-{int(m.mesh_vertnum[mesh_ids].sum())}-{int(m.mesh_facenum[mesh_ids].sum())}"
    idx_file = WEB / "meshes.json"
    if idx_file.exists() and json.loads(idx_file.read_text()).get("signature") == sig:
        return
    WEB.mkdir(parents=True, exist_ok=True)
    index, blobs, off = {}, [], 0
    for mid in mesh_ids:
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float32)
        f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(np.uint32)
        index[str(mid)] = {"v_off": off, "v_n": int(len(v)), "f_off": off + v.nbytes, "f_n": int(len(f))}
        blobs += [v.tobytes(), f.tobytes()]
        off += v.nbytes + f.nbytes
    (WEB / "meshes.bin").write_bytes(b"".join(blobs))
    idx_file.write_text(json.dumps({"signature": sig, "meshes": index}))


def _geom_info(arm, g):
    m, mj = arm.m, arm.mj
    t = int(m.geom_type[g])
    rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    kind = {int(mj.mjtGeom.mjGEOM_PLANE): "plane", int(mj.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mj.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mj.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mj.mjtGeom.mjGEOM_BOX): "box", int(mj.mjtGeom.mjGEOM_MESH): "mesh"}.get(t, "other")
    out = {"id": int(g), "kind": kind, "size": [float(x) for x in m.geom_size[g]],
           "rgba": [round(float(x), 3) for x in rgba], "body": m.body(m.geom_bodyid[g]).name,
           "name": m.geom(g).name or ""}
    if kind == "mesh":
        out["mesh"] = str(int(m.geom_dataid[g]))
    return out


def undistort_preview(capture_dir: Path, lens, out_dir: Path, frames_fallback_every=3):
    """Pack of undistorted 320 px overhead frames (from the rig's live preview, or made from full frames)."""
    K, D = np.array(lens["K"], float), np.array(lens["D"], float)
    W0, H0 = lens["image_size"]
    src = capture_dir / "overhead"
    items = []
    if (src / "preview.csv").exists() and (src / "preview.bin").exists():
        data = (src / "preview.bin").read_bytes()
        for line in (src / "preview.csv").read_text().splitlines()[1:]:
            i, t, o, n = line.split(",")
            items.append((float(t), cv2.imdecode(np.frombuffer(data[int(o):int(o) + int(n)], np.uint8), cv2.IMREAD_COLOR)))
    elif (src / "frames.csv").exists() and (src / "frames").exists():
        for line in (src / "frames.csv").read_text().splitlines()[1::frames_fallback_every]:
            i, t = line.split(",")[:2]
            f = cv2.imread(str(src / "frames" / f"{int(i):06d}.jpg"))
            if f is not None:
                items.append((float(t), cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA)))
    if not items:
        return None
    w, h = items[0][1].shape[1], items[0][1].shape[0]
    s = w / W0
    Ks = K.copy()
    Ks[:2] *= s
    m1, m2 = cv2.initUndistortRectifyMap(Ks, D, None, Ks, (w, h), cv2.CV_16SC2)
    blobs, rows, off = [], [], 0
    for t, f in items:
        ok, buf = cv2.imencode(".jpg", cv2.remap(f, m1, m2, cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 75])
        b = buf.tobytes()
        blobs.append(b)
        rows.append(f"{t:.4f},{off},{len(b)}")
        off += len(b)
    (out_dir / "overhead.bin").write_bytes(b"".join(blobs))
    (out_dir / "overhead.csv").write_text("t,offset,bytes\n" + "\n".join(rows) + "\n")
    return {"width": w, "height": h, "K": Ks.tolist(), "frames": len(rows), "items": items}


def export_episode(arm, ep_dir: Path, capture_dir: Path, side, t_plan, t_offset, Q5, grip, demo_tcp, robot_tcp,
                   tracked, lens, plan_summary, other_pose, finder_cls=None):
    """Everything the browser replay needs for one plan."""
    m, d, mj = arm.m, arm.d, arm.mj
    export_meshes(arm)
    out = ep_dir / "web"
    out.mkdir(parents=True, exist_ok=True)
    preview = undistort_preview(capture_dir, lens, out)

    # props from the (undistorted, 320 px) overhead preview: box once, cube per preview frame
    box_pose, cube_track = (None, 0.0), []
    if preview is not None and finder_cls is not None:
        lens_s = dict(lens)
        lens_s["K"] = preview["K"]
        lens_s["D"] = [0.0] * 5                              # the preview pack is already undistorted
        finder = finder_cls(lens_s)
        boxes = [b for b in (finder.box(f) for _, f in preview["items"][:15]) if b is not None]
        if boxes:
            box_pose = (np.median([b[0] for b in boxes], 0), float(np.median([b[1] for b in boxes])))
        for t, f in preview["items"]:
            c = finder.cube(f)
            if c is not None:
                cube_track.append((t, c))

    vis = visible_geoms(m)
    arm.park_other(other_pose)
    gq = m.jnt_qposadr[m.joint(f"{side}_gripper").id]
    dyn_bodies = (f"{side}_",) if True else ()
    prop_bodies = ("cube", "can")
    dynamic = [g for g in vis if m.body(m.geom_bodyid[g]).name.startswith(dyn_bodies + prop_bodies)
               or m.body(m.geom_bodyid[g]).name in prop_bodies]
    static = [g for g in vis if g not in dynamic]
    # static parts: pose once (other arm parked, props placed at their first pose)
    cube0 = cube_track[0][1] if cube_track else (None, 0.0)
    arm.place_props(cube_xy=cube0[0], cube_yaw=cube0[1], box_xy=box_pose[0], box_yaw=box_pose[1])
    mj.mj_forward(m, d)
    geoms = []
    for g in static:
        gi = _geom_info(arm, g)
        gi["pose"] = [float(x) for x in d.geom_xpos[g]] + [float(x) for x in _mat2quat(mj, d.geom_xmat[g])]
        geoms.append(gi)
    dyn_info = [_geom_info(arm, g) for g in dynamic]

    # moving parts per frame
    ct = np.array([c[0] for c in cube_track]) if cube_track else None
    poses = np.zeros((len(Q5), len(dynamic), 7), np.float32)
    lo, hi = -0.17453, 1.74533
    for k in range(len(Q5)):
        arm.set(Q5[k])
        d.qpos[gq] = lo + float(grip[k]) / 100.0 * (hi - lo)
        if ct is not None:
            j = int(np.argmin(np.abs(ct - (t_plan[k] + t_offset))))
            arm.place_props(cube_xy=cube_track[j][1][0], cube_yaw=cube_track[j][1][1],
                            box_xy=box_pose[0], box_yaw=box_pose[1])
        mj.mj_kinematics(m, d)
        for n, g in enumerate(dynamic):
            poses[k, n, :3] = d.geom_xpos[g]
            poses[k, n, 3:] = _mat2quat(mj, d.geom_xmat[g])
    (out / "poses.bin").write_bytes(poses.tobytes())

    wp = lens["world_pose"]
    cam = {"pos": wp["pos"], "quat_wxyz": wp["quat_wxyz"]}
    if preview is not None:
        cam.update(width=preview["width"], height=preview["height"], K=preview["K"])
    scene = {
        "side": side, "frames": len(Q5), "hz": float(1 / np.median(np.diff(t_plan))) if len(t_plan) > 1 else 30.0,
        "t": [round(float(x), 4) for x in t_plan], "t_offset": float(t_offset),
        "static": geoms, "dynamic": dyn_info, "camera": cam,
        "preview": None if preview is None else {"bin": "overhead.bin", "csv": "overhead.csv"},
        "demo_tcp": np.round(np.asarray(demo_tcp, float), 5).tolist(),
        "robot_tcp": np.round(np.asarray(robot_tcp, float), 5).tolist(),
        "tracked": [bool(x) for x in tracked], "grip": [round(float(x), 1) for x in grip],
        "plan": plan_summary,
        "props": {"box": None if box_pose[0] is None else [float(box_pose[0][0]), float(box_pose[0][1]), box_pose[1]],
                  "cube_frames": len(cube_track)},
    }
    (out / "scene.json").write_text(json.dumps(scene, separators=(",", ":")))
    return out
