#!/usr/bin/env python3
"""SO-UMI episode -> joint plan for one SO-101 arm, checked and previewed in the rig's own sim.

Takes data/episodes/<ts>/trajectory.csv (from umi/convert.py: the handheld's gripper-body pose
in the table frame, tag 13) and produces, for the chosen arm:

    data/episodes/<ts>/plan_<side>.json     joint targets (pi_sim / so101_real units: degrees for the
                                            5 arm joints, 0-100 for the gripper) on a fixed clock,
                                            plus every check result. This is what the rig runner plays.
    data/episodes/<ts>/plan_<side>.mp4      sim replay from the rig's calibrated overhead camera,
                                            next to the real overhead video, for a visual check
    data/episodes/<ts>/plan_<side>.png      tracking error, joint speeds, clearance plots

Chain of frames
  handheld gripper body --(convert.py)--> table frame (tag 13)
  table frame --(overhead camera: rig calibration + tag 13 pose)--> pi_sim world
  pi_sim world --(IK on the rig's MuJoCo model)--> joints

Sim: the bench-matched clamp twin (sim/squeeze_env.py SqueezeEnv): 232 mm layout, real joint
ranges, the grey box (parked out of view), real-matched colours. Same model the rig's safety guard uses.

IK is solved for the gripper BODY (the frame the handheld replicates), steering the rig's
person-confirmed tool point (tcp_confirmed.json) plus the gripper orientation at a lower weight
(the SO-101 has 5 joints, so a full 6-D pose can only be matched as near as it can be), with
dream_kin.py's damped least-squares update.

This script never talks to hardware.

Usage
  uv run --python 3.12 --with mujoco==3.15.0 --with gymnasium==1.4.0 --with numpy --with pillow \
      --with "opencv-python<5" --with matplotlib --with scipy umi/to_robot.py 2026-11-15T00-06-19-417Z --side left
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

SIM_HOME6 = np.array([0.0, -45.8, 45.9, 90.0, -2.8, 100.0])  # so101_real.SIM_HOME (per arm)
TCP_IN_GRIPPER = np.array([-0.0066, -0.0002, -0.0996])         # rig's measured tool point (tcp_confirmed.json, 2026-11-14)
HAND_BODIES = ("gripper", "moving_jaw_so101_v1", "camera_mount", "wrist")
JAW_HINGE_TO_TAG_MM = 54.2   # moving-jaw hinge -> finger-tag spot, in the hinge plane (cad/src/asm.py + finger_tag_patches.json)
JAW_CLOSED_RAD = -0.17       # CAD: jaws touching (cad/src/build.py)
SIM_GRIP_LO, SIM_GRIP_HI = -0.17453, 1.74533   # sim gripper actuator range (rad) = 0..100 units


SIM_CAMERA_RIG = REPO / "sim" / "assets" / "so101_camera_rig.json"


def load_overhead(path=None):
    """The calibrated overhead camera as {K, D, image_size, world_pose, source}.
    Default: the repo's sim/ (sim/assets/so101_camera_rig.json, the same camera the sim renders with).
    A calibration/*.json lens file in the older format is still accepted."""
    p = Path(path) if path else SIM_CAMERA_RIG
    d = json.loads(p.read_text())
    if "cameras" in d:
        o = d["cameras"]["overhead"]
        i = o["intrinsics"]
        return {"K": [[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], "D": o["distortion"],
                "image_size": [o["width"], o["height"]],
                "world_pose": {"pos": o["pose"]["pos"], "quat_wxyz": o["pose"]["quat_wxyz"]},
                "source": str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)}
    return d


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def table_to_world(lens_file: Path, table_pose_file: Path, snap_to_table=True):
    """4x4 transform table frame (tag 13, metres) -> pi_sim world."""
    lens = load_overhead(lens_file)
    wp = lens["world_pose"]
    R_wc = quat_wxyz_to_mat(wp["quat_wxyz"]) @ np.diag([1, -1, -1])  # OpenCV camera -> world
    tp = json.loads(table_pose_file.read_text())
    R_ct, t_ct = np.array(tp["R"]), np.array(tp["t_mm"]) / 1000.0
    T = np.eye(4)
    T[:3, :3] = R_wc @ R_ct
    T[:3, 3] = R_wc @ t_ct + np.array(wp["pos"])
    raw_z = float(T[2, 3])
    if snap_to_table:
        # The tag lies on the table top (world z = 0). Remove the calibration's height error and tilt.
        z = T[:3, 2] / np.linalg.norm(T[:3, 2])
        x = T[:3, 0] - z * (T[:3, 0] @ z)
        R = np.stack([x / np.linalg.norm(x), np.cross(z, x / np.linalg.norm(x)), z], 1)
        flat = np.array([0, 0, 1.0])
        # rotate so the tag normal is exactly world +z, keeping its in-plane heading
        heading = math.atan2(R[1, 0], R[0, 0])
        c, s = math.cos(heading), math.sin(heading)
        T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T[2, 3] = 0.0
        tilt_deg = math.degrees(math.acos(min(1.0, abs(z @ flat))))
    else:
        tilt_deg = 0.0
    return T, {"tag13_world_m": T[:3, 3].round(4).tolist(), "raw_z_m": round(raw_z, 4),
               "snapped_tilt_deg": round(tilt_deg, 2), "camera": lens.get("source", "")}


class ArmModel:
    def __init__(self, side: str):
        # The bench-matched clamp twin (sim/squeeze_env.py SqueezeEnv): the rig's 232 mm layout, real
        # joint ranges (so101_real_ranges.json), the grey box, off-table geometry and real-matched colours.
        # Built the way the rig's real_clamp.py builds its safety model.
        sys.path.insert(0, str(REPO))   # sim/ is this repo's own copy of the bench twin (see sim/README.md)
        import mujoco
        from sim import squeeze_env
        squeeze_env.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}  # clamp-only, slow
        self.mj = mujoco
        self.twin = squeeze_env.SqueezeEnv()
        self.twin.reset(seed=0)
        self.env = self.twin.sim                 # pi_sim RobotEnv underneath: .robot[side], .model
        self.m = self.twin.model                 # real joint ranges applied
        self.d = mujoco.MjData(self.m)
        for joint, xyz in (("cube_joint", (2.0, 2.0, -1.0)), ("can_joint", (2.5, 2.0, -1.0))):   # park the props
            adr = self.m.jnt_qposadr[self.m.joint(joint).id]
            self.d.qpos[adr:adr + 7] = [*xyz, 1, 0, 0, 0]
        if self.m.nmocap:
            self.d.mocap_pos[:] = [2.0, 2.0, -1.0]   # pi_sim's green goal disk: not on the real bench
        self.side = side
        a = self.env.robot[side]
        self.qpos, self.dof, self.jids = a.qpos_ids, a.dof_ids, a.joint_ids
        self.limits = self.m.jnt_range[a.joint_ids].copy()
        self.body = self.m.body(f"{side}_gripper").id
        names = {f"{side}_{b}" for b in HAND_BODIES}
        self.hand_geoms = [g for g in range(self.m.ngeom) if self.m.body(self.m.geom_bodyid[g]).name in names]
        other = "right" if side == "left" else "left"
        self.other_qpos = self.env.robot[other].qpos_ids
        self.park_other(SIM_HOME6[:5])
        def arm_geoms(sd):
            return [g for g in range(self.m.ngeom)
                    if self.m.body(self.m.geom_bodyid[g]).name.startswith(f"{sd}_")
                    and self.m.geom_contype[g] | self.m.geom_conaffinity[g]]
        self.my_geoms, self.other_geoms = arm_geoms(side), arm_geoms(other)
        self._ga, self._gb = np.array(self.my_geoms), np.array(self.other_geoms)
        # bounding-sphere radii summed per (mine, other) pair: centre distance minus this is a lower bound
        # on the pair's true distance, so pairs that can't beat the current best are never measured
        self._rsum = self.m.geom_rbound[self._ga][:, None] + self.m.geom_rbound[self._gb][None, :]
        self._qpos_ix, self._dof_ix = np.asarray(self.qpos), np.asarray(self.dof)
        self._damp = 1e-4 * np.eye(6)
        # The 5-DOF limit, exactly: shoulder lift, elbow and wrist flex bend about parallel axes, so the
        # wrist-roll axis (the gripper's pointing direction) is always perpendicular to the lift axis.
        # Pitch in that plane and twist about the roll axis are reachable; sideways yaw out of it is not.
        self._lift_jid = self.m.joint(f"{side}_shoulder_lift").id
        roll_jid = self.m.joint(f"{side}_wrist_roll").id
        assert self.m.jnt_bodyid[roll_jid] == self.body, "wrist roll must move the gripper body"
        self._roll_axis_b = self.m.jnt_axis[roll_jid].astype(float)
        self.reachable_orient = False    # True: aim only at the reachable part (to_robot.py --reachable-orient)
        # The jaws' pinch line in the gripper frame: perpendicular to the moving jaw's hinge and to the
        # wrist -> fingertip direction (for --grasp-orient).
        self.mj.mj_kinematics(self.m, self.d)
        Rg0 = self.d.xmat[self.body].reshape(3, 3)
        hinge_b = Rg0.T @ self.d.xaxis[self.m.joint(f"{side}_gripper").id]
        p_b = TCP_IN_GRIPPER / np.linalg.norm(TCP_IN_GRIPPER)
        self._pinch_b = np.cross(hinge_b, p_b) / np.linalg.norm(np.cross(hinge_b, p_b))
        self._pan_anchor = self.d.xanchor[self.m.joint(f"{side}_shoulder_pan").id].copy()   # fixed base point

    def grasp_target(self, tcp_world, R_world):
        """grasp_R for a fingertip target, with the shoulder axis this target implies: the wrist-roll origin
        (the gripper body origin) always lies in the vertical plane through the pan axis (checked on the
        model to 0.2 mm), so the plane, and the shoulder axis normal to it, follow from where the wrist must be.
        Two passes settle it (the fingertip sits 6.6 mm off the roll axis)."""
        h = self.d.xaxis[self._lift_jid].copy()
        R_t = R_world
        for _ in range(3):
            w = tcp_world - R_t @ TCP_IN_GRIPPER
            rad = w[:2] - self._pan_anchor[:2]
            if np.linalg.norm(rad) < 0.02:            # wrist over the base: plane undefined, keep the last one
                break
            rad /= np.linalg.norm(rad)
            h = np.array([-rad[1], rad[0], 0.0])
            R_t = self.grasp_R(R_world, h)
        return R_t

    def grasp_R(self, R_world, h):
        """For grasps: the reachable orientation that keeps what decides a grasp. The gripper's tilt is kept as
        far as the arm's plane allows (reachable_R), then the wrist twist is chosen so the jaws close along
        YOUR jaw line as seen from above (which faces the fingers touch). What's given up is only the jaw lean
        (one finger lower than the other). h: this arm's shoulder-lift axis in the world, held fixed.
        Only for a gripper pointing roughly down (a table grasp): full effect up to 35 deg from straight down,
        blended out by 60 deg; more horizontal poses keep the default target (your orientation as is)."""
        pd = R_world @ (TCP_IN_GRIPPER / np.linalg.norm(TCP_IN_GRIPPER))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -pd[2]))))
        w_tilt = min(1.0, max(0.0, (60.0 - tilt) / 25.0))
        if w_tilt <= 0.0:
            return R_world
        R0, _ = self.reachable_R(R_world, h)
        R_full = R0
        cd = R_world @ self._pinch_b
        if math.hypot(cd[0], cd[1]) >= 0.2:           # jaw line not near vertical: keep its heading from above
            r = R0 @ self._roll_axis_b                # twisting about the roll axis stays reachable
            c0 = R0 @ self._pinch_b
            th = np.deg2rad(np.arange(-90.0, 90.25, 0.25))
            C = (np.outer(np.cos(th), c0) + np.outer(np.sin(th), np.cross(r, c0))
                 + np.outer(1 - np.cos(th), r * (r @ c0)))
            want = math.atan2(cd[1], cd[0])
            diff = (np.arctan2(C[:, 1], C[:, 0]) - want + np.pi / 2) % np.pi - np.pi / 2   # jaw lines: mod 180
            k = int(np.argmin(np.abs(diff) + 1e-3 * np.abs(th)))                           # smallest twist that fits
            # a twist this big means the heading barely responds to twist here: fade it out (20 -> 45 deg)
            a = th[k] * min(1.0, max(0.0, (45.0 - abs(math.degrees(th[k]))) / 25.0))
            K = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            R_full = (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K) @ R0
        if w_tilt >= 1.0:
            return R_full
        # partial: rotate from your orientation toward the grasp target by the blend fraction
        Rrel = R_world.T @ R_full
        ang = math.acos(max(-1.0, min(1.0, (np.trace(Rrel) - 1) / 2)))
        if ang < 1e-9:
            return R_world
        ax = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]]) / (2 * math.sin(ang))
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        b = w_tilt * ang
        return R_world @ (np.eye(3) + math.sin(b) * K + (1 - math.cos(b)) * K @ K)

    def reachable_R(self, R_world, h=None):
        """R_world minus only the part this arm can't make: the smallest rotation that tips the gripper's
        pointing direction back into the arm's plane (from the CURRENT pan unless h is given). Pitch and twist
        are kept. Returns (R_target, dropped_yaw_deg)."""
        h = self.d.xaxis[self._lift_jid] if h is None else h
        z = R_world @ self._roll_axis_b
        off = float(z @ h)
        zp = z - off * h
        nz = np.linalg.norm(zp)
        if nz < 1e-6:                                   # pointing along the flex axis: no unique fix
            return R_world, 90.0
        zp /= nz
        v = np.cross(z, zp)
        s, c = np.linalg.norm(v), float(z @ zp)
        if s < 1e-12:
            return R_world, 0.0
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
        return Rm @ R_world, math.degrees(math.asin(min(1.0, abs(off))))

    def place_props(self, cube_xy=None, cube_yaw=0.0, box_xy=None, box_yaw=0.0):
        """Put the red cube / grey box where the overhead camera saw them (world x, y in m); None = off the table."""
        from sim import squeeze_env as SQ
        from sim.scene import CUBE_HALF
        for joint, xy, yaw, z in (("cube_joint", cube_xy, cube_yaw, CUBE_HALF),
                                  ("can_joint", box_xy, box_yaw, SQ.CAN_HEIGHT / 2)):
            adr = self.m.jnt_qposadr[self.m.joint(joint).id]
            if xy is None:
                self.d.qpos[adr:adr + 7] = [2.0 if joint == "cube_joint" else 2.5, 2.0, -1.0, 1, 0, 0, 0]
            else:
                self.d.qpos[adr:adr + 7] = [xy[0], xy[1], z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.mj.mj_kinematics(self.m, self.d)

    def park_other(self, q5_deg):
        """Pose of the arm that is NOT replaying (sim units, deg); used for clearance checks."""
        self.d.qpos[self.other_qpos] = np.deg2rad(np.asarray(q5_deg[:5], float))
        self.mj.mj_kinematics(self.m, self.d)

    def clearance_to_other(self, cap=0.05):
        """Smallest distance (m) between this arm's collision geoms and the other arm's, up to cap."""
        mj, m, d = self.mj, self.m, self.d
        mj.mj_kinematics(m, d)
        pa, pb = d.geom_xpos[self._ga], d.geom_xpos[self._gb]
        lb = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)) - self._rsum   # lower bound per pair
        ia, ib = np.nonzero(lb < cap)
        lbs = lb[ia, ib]
        best = cap
        ft = np.zeros(6)
        for k in np.argsort(lbs):                     # closest-looking pairs first, so best drops fast
            if lbs[k] >= best:
                break                                 # no remaining pair can be closer than best
            dist = mj.mj_geomDistance(m, d, self.my_geoms[ia[k]], self.other_geoms[ib[k]], best, ft)
            if dist < best:
                best = dist
        return float(best)

    def set(self, q5_deg):
        self.d.qpos[self.qpos] = np.deg2rad(np.asarray(q5_deg[:5], float))
        self.mj.mj_kinematics(self.m, self.d)

    def gripper_pose(self):
        return self.d.xpos[self.body].copy(), self.d.xmat[self.body].reshape(3, 3).copy()

    def ik(self, p_tcp_world, R_world, q0_deg, iters=60, rot_w=0.15, max_step_deg=None):
        """Damped least squares (dream_kin.py's update) for the gripper body: the TCP point to
        p_tcp_world, the body orientation toward R_world at weight rot_w.
        max_step_deg: no joint may end more than this far from q0_deg. Frame to frame this keeps the
        solution on the same arm configuration (it misses the target a little instead of flipping
        the elbow/wrist to another branch, which is a violent jump on the real servos)."""
        mj, m, d = self.mj, self.m, self.d
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        q_t = np.zeros(4)
        mj.mju_mat2Quat(q_t, np.ascontiguousarray(R_world).ravel())
        self.set(q0_deg)
        q, rot_err = np.zeros(4), np.zeros(3)
        # everything loop-invariant built once (this loop runs ~100k times per episode)
        qix, dix, body, damp = self._qpos_ix, self._dof_ix, self.body, self._damp
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        if max_step_deg is not None:
            q0r = np.deg2rad(q0_deg)
            lo_s, hi_s = q0r - np.deg2rad(max_step_deg), q0r + np.deg2rad(max_step_deg)
        J, err = np.empty((6, len(dix))), np.empty(6)
        xmat_b = d.xmat[body]
        project = self.reachable_orient
        for _ in range(iters):
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            pb, Rb = d.xpos[body], xmat_b.reshape(3, 3)
            point = pb + Rb @ TCP_IN_GRIPPER
            pos_err = p_tcp_world - point
            if project:      # aim only at what this arm can reach, so the impossible yaw never pulls the pan
                mj.mju_mat2Quat(q_t, np.ascontiguousarray(self.reachable_R(R_world)[0]).ravel())
            mj.mju_mat2Quat(q, xmat_b)
            mj.mju_subQuat(rot_err, q_t, q)
            rot_world = Rb @ rot_err
            if math.sqrt(pos_err @ pos_err) < 0.0002 and math.sqrt(rot_world @ rot_world) < 0.002:
                break
            mj.mj_jac(m, d, jp, jr, point, body)
            J[:3] = jp[:, dix]
            np.multiply(jr[:, dix], rot_w, out=J[3:])
            err[:3] = pos_err
            np.multiply(rot_world, rot_w, out=err[3:])
            dq = J.T @ np.linalg.solve(J @ J.T + damp, err)
            qp = d.qpos[qix] + np.minimum(np.maximum(dq, -0.15), 0.15)
            if max_step_deg is not None:
                qp = np.minimum(np.maximum(qp, lo_s), hi_s)
            d.qpos[qix] = np.minimum(np.maximum(qp, lo), hi)
        mj.mj_kinematics(m, d)
        pb, Rb = self.gripper_pose()
        pos_e = float(np.linalg.norm(p_tcp_world - (pb + Rb @ TCP_IN_GRIPPER)))
        ang_e = float(np.degrees(np.arccos(np.clip((np.trace(Rb.T @ R_world) - 1) / 2, -1, 1))))
        return np.rad2deg(d.qpos[self.qpos]).copy(), pos_e, ang_e

    def hold_torque(self, q5_deg=None):
        """|torque| (Nm) each arm joint needs to hold the current pose against gravity (no contacts, at rest)."""
        mj, m, d = self.mj, self.m, self.d
        if q5_deg is not None:
            self.set(q5_deg)
        saved_q = d.qpos.copy()
        flags = m.opt.disableflags
        m.opt.disableflags = flags | mj.mjtDisableBit.mjDSBL_CONTACT
        d.qvel[:] = 0
        d.qacc[:] = 0
        mj.mj_inverse(m, d)
        tq = np.abs(d.qfrc_inverse[self.dof]).copy()
        m.opt.disableflags = flags
        d.qpos[:] = saved_q
        mj.mj_kinematics(m, d)
        return tq

    def soften_limits(self, margin_deg):
        """Keep IK this far inside every joint's range, so no servo is driven into its hard stop."""
        self.hard_limits = self.limits.copy()
        mrg = np.deg2rad(margin_deg)
        self.limits = np.stack([self.hard_limits[:, 0] + mrg, self.hard_limits[:, 1] - mrg], 1)

    def shoulder_xy(self):
        jid = self.m.joint(f"{self.side}_shoulder_lift").id
        self.mj.mj_kinematics(self.m, self.d)
        return self.d.xanchor[jid][:2].copy()

    def hand_lowest_z(self):
        """Lowest point of the hand: exact mesh vertices for mesh geoms, AABB corners otherwise."""
        m, d = self.m, self.d
        lo = np.inf
        for g in self.hand_geoms:
            Rg = d.geom_xmat[g].reshape(3, 3)
            if m.geom_type[g] == self.mj.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                v = m.mesh_vert[a:a + n]
                lo = min(lo, float((d.geom_xpos[g][2] + v @ Rg[2]).min()))
            else:
                aabb = m.geom_aabb[g]
                c, h = aabb[:3], aabb[3:]
                corners = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                lo = min(lo, float((d.geom_xpos[g][2] + corners @ Rg[2]).min()))
        return lo

    def tcp_world(self):
        pb, Rb = self.gripper_pose()
        return pb + Rb @ TCP_IN_GRIPPER


def clamp_spans(rows):
    """[(start_s, end_s, reason)] for runs of frames whose target was clamped."""
    spans, cur = [], None
    for r in rows:
        why = r["clamp"][0].split(":")[0] if r.get("clamp") else None
        if why and cur and cur[2] == why:
            cur[1] = r["t"]
        else:
            if cur:
                spans.append((round(cur[0], 2), round(cur[1], 2), cur[2]))
            cur = [r["t"], r["t"], why] if why else None
    if cur:
        spans.append((round(cur[0], 2), round(cur[1], 2), cur[2]))
    return [s for s in spans if s[1] - s[0] >= 0.2]


# ---------------------------------------------------------------- props seen by the overhead camera
class PropFinder:
    """Locate the red cube and the grey box in an overhead frame and put them on the table in world
    coordinates: pixel -> undistorted ray from the calibrated overhead camera -> plane at the prop's height."""

    def __init__(self, lens):
        import cv2
        self.cv2 = cv2
        self.K, self.D = np.array(lens["K"], float), np.array(lens["D"], float)
        wp = lens["world_pose"]
        self.R_wc = quat_wxyz_to_mat(wp["quat_wxyz"]) @ np.diag([1, -1, -1])
        self.c_w = np.array(wp["pos"], float)

    def to_table(self, px, z):
        n = self.cv2.undistortPoints(np.asarray(px, np.float64).reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        rays = (self.R_wc @ np.c_[n, np.ones(len(n))].T).T
        s = (z - self.c_w[2]) / rays[:, 2]
        return self.c_w + rays * s[:, None]

    def _rect_pose(self, cnt, z):
        cv2 = self.cv2
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        W = self.to_table(box, z)
        centre = W.mean(0)
        e = W[1] - W[0]
        return centre[:2], math.atan2(e[1], e[0])

    def cube(self, bgr):
        """(xy, yaw) of the red cube's top face, or None."""
        cv2 = self.cv2
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = (((hsv[..., 0] < 10) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 60)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if 80 < cv2.contourArea(c) < 6000]
        if not cnts:
            return None
        from sim.scene import CUBE_HALF
        return self._rect_pose(max(cnts, key=cv2.contourArea), 2 * CUBE_HALF)

    def box(self, bgr):
        """(xy, yaw) of the grey box, or None. The real box reads blue-grey in the overhead camera (blue well
        above red), the wood the opposite; its inside shows up as a compact near-square blob."""
        cv2 = self.cv2
        b, r = bgr[..., 0].astype(int), bgr[..., 2].astype(int)
        v = bgr.max(2).astype(int)
        m = ((b - r > 30) & (v > 60) & (v < 210)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            a = cv2.contourArea(c)
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            if not 800 < a < 20000 or min(w, h) < 1 or max(w, h) / min(w, h) > 1.6 or a / (w * h) < 0.65:
                continue
            corners = cv2.boxPoints(rect)
            W = self.to_table(corners, 0.003)                 # the box floor
            side = float(np.linalg.norm(W[1] - W[0]))
            if 0.04 < side < 0.14:
                score = a * a / (w * h)
                if best is None or score > best[0]:
                    e = W[1] - W[0]
                    best = (score, W.mean(0)[:2], math.atan2(e[1], e[0]))
        return None if best is None else best[1:]


def opening_to_grip_units(open_mm):
    """Finger lateral offset (mm, relative to closed) -> sim gripper units (0 closed .. 100 open).
    Geometry: the finger-tag spot swings on a 54 mm lever about the moving-jaw hinge."""
    dtheta = np.arcsin(np.clip(np.asarray(open_mm, float) / JAW_HINGE_TO_TAG_MM, 0.0, 1.0))
    theta = JAW_CLOSED_RAD + dtheta
    return np.clip((theta - SIM_GRIP_LO) / (SIM_GRIP_HI - SIM_GRIP_LO) * 100.0, 0.0, 100.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="episode id (capture timestamp) under data/episodes/")
    ap.add_argument("--side", default="left", choices=("left", "right"))
    ap.add_argument("--lens", default=None, help="overhead camera calibration (default: the repo's sim/assets/so101_camera_rig.json)")
    ap.add_argument("--table-pose", default=str(REPO / "calibration/overhead_table_pose.json"))
    ap.add_argument("--floor-mm", type=float, default=5.0,
                    help="lowest allowed hand point above the table (was 10 = rig.json table_floor_cm; 5 matches the "
                         "replay's own floor guard, real/replay.py --floor-cm 0.5)")
    ap.add_argument("--midline-mm", type=float, default=None,
                    help="optional blanket rule: keep the TCP this far on its own side of x=0 (off by default; "
                         "the clearance check against the other arm replaces it)")
    ap.add_argument("--other-pose", type=float, nargs=5, default=[-29.0, -91.2, 90.1, -27.5, -87.0],
                    metavar=("PAN", "LIFT", "ELBOW", "WFLEX", "ROLL"),
                    help="where the other arm is parked (sim/so101_real deg); default = its folded rest pose "
                         "as read from the rig")
    ap.add_argument("--clearance-mm", type=float, default=15.0,
                    help="minimum gap to the other arm; closer targets are held at the last safe pose")
    ap.add_argument("--max-ik-mm", type=float, default=10.0, help="flag frames where IK misses the TCP by more")
    ap.add_argument("--max-deg-s", type=float, default=60.0, help="joint speed cap; sets the slow-down factor")
    ap.add_argument("--shift-mm", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"),
                    help="move the whole demo by (dx, dy) mm in the world before IK, e.g. into this arm's half")
    ap.add_argument("--no-tip-calib", action="store_true", help="ignore calibration/handheld_tip.json")
    ap.add_argument("--max-step-deg", type=float, default=20.0,
                    help="largest change of any joint between two demo frames (30 Hz). Only there to stop IK branch "
                         "flips (jumps of 30-180 deg); real fast hand moves need ~6-18 deg/frame, and speed on the "
                         "real arm is limited separately (playback slow-down / --realtime-vmax, rig safety)")
    ap.add_argument("--realtime-vmax", type=float, default=150.0,
                    help="real-time playback: joint speed cap, deg/s (STS3215 no-load ~270); 0 = old slowed-down plan")
    ap.add_argument("--realtime-amax", type=float, default=800.0,
                    help="real-time playback: joint acceleration cap, deg/s^2 (no sudden starts/stops)")
    ap.add_argument("--smooth-s", type=float, default=0.1,
                    help="zero-phase Gaussian smoothing of the joint paths (sigma, seconds); 0 = off")
    ap.add_argument("--max-hold-nm", type=float, default=0.9,
                    help="actuator protection: largest torque any joint may need just to hold a pose against "
                         "gravity (STS3215 stall ~1.9 Nm at 7.4 V; 0.9 = about half). Further out, the target is "
                         "pulled back toward the shoulder")
    ap.add_argument("--limit-margin-deg", type=float, default=5.0,
                    help="actuator protection: stay this far inside every joint's range (no driving into hard stops)")
    ap.add_argument("--reach-tol-mm", type=float, default=8.0,
                    help="if the arm misses the demo by more than this, aim at the furthest reachable point instead")
    ap.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True,
                    help="solve the whole demo at once (umi/trajopt.py): tracking + smoothness with the safety "
                         "limits as heavy penalties / bounds, started from the frame-by-frame plan "
                         "(default on; --no-optimize = frame-by-frame plan only)")
    ap.add_argument("--opt-iters", type=int, default=40, help="--optimize: max solver evaluations")
    ap.add_argument("--grasp-orient", action="store_true",
                    help="aim at the grasp-relevant part of your wrist: tilt as far as the arm's plane allows and the jaw line as seen from above exactly; only the jaw lean (one finger lower) is given up")
    ap.add_argument("--reachable-orient", action="store_true",
                    help="aim only at the part of the demo orientation a 5-DOF arm can reach (pitch + twist; the sideways yaw out of the arm's plane is dropped). Experimental: on 06-26-51 it cut twist error 0.5 -> 0.1 deg but raised the worst TCP miss 62 -> 160 mm")
    ap.add_argument("--export-web", action="store_true",
                    help="write data for the in-browser replay (viewer/replay.html) instead of needing a video")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _shifted")
    args = ap.parse_args()

    ep = DATA / "episodes" / args.episode
    tr = np.genfromtxt(ep / "trajectory.csv", delimiter=",", names=True)
    t = tr["t"]
    pose_ok = tr["pose_valid"].astype(bool)
    hz = 1.0 / np.median(np.diff(t))

    T_wt, frame_info = table_to_world(args.lens, Path(args.table_pose))
    # Real table plane from a table-touch recording (umi/table_touch.py): tag 13's own estimate can be tilted
    # by a degree or two, which is tens of mm at the edge of the work area. Re-level the table frame on it.
    plane_file = REPO / "calibration" / "table_plane.json"
    if plane_file.exists():
        pl = json.loads(plane_file.read_text())
        if pl.get("good"):
            a, b, c = pl["plane"]["a"], pl["plane"]["b"], pl["plane"]["c_mm"] / 1000.0
            nrm = np.array([-a, -b, 1.0])
            nrm /= np.linalg.norm(nrm)
            ax = np.cross(nrm, [0, 0, 1.0])
            ang = math.asin(min(1.0, np.linalg.norm(ax)))
            import cv2 as _cv2
            R_c = _cv2.Rodrigues(ax / (np.linalg.norm(ax) + 1e-12) * ang)[0] if ang > 1e-9 else np.eye(3)
            Cm = np.eye(4)
            Cm[:3, :3] = R_c
            Cm[:3, 3] = -R_c @ np.array([0, 0, c])
            T_wt = T_wt @ Cm
            frame_info["table_plane"] = {"from": pl["capture"], "tilt_deg": pl["tilt_deg"], "height_mm": round(c * 1000, 1)}
            print(f"table plane from touches ({pl['capture']}): re-levelled by {pl['tilt_deg']:.2f} deg, "
                  f"{c * 1000:+.1f} mm at tag 13")
    print(f"table tag 13 -> world: {frame_info}")

    # handheld gripper-body pose in the world
    P_t = np.stack([tr["x"], tr["y"], tr["z"]], 1)
    Q_t = np.stack([tr["qx"], tr["qy"], tr["qz"], tr["qw"]], 1)
    R_wt = T_wt[:3, :3]

    def quat_xyzw_to_mat(q):
        x, y, z, w = q
        return quat_wxyz_to_mat((w, x, y, z))

    # keep only the span where the handheld was tracked; hold the edges, fill short gaps linearly
    idx = np.where(pose_ok)[0]
    if len(idx) < 10:
        raise SystemExit("too few tracked frames in this episode")
    i0, i1 = idx[0], idx[-1]
    sel = np.arange(i0, i1 + 1)
    gaps = sel[~pose_ok[sel]]
    for k in range(3):
        P_t[sel, k] = np.interp(sel, idx, P_t[idx, k])
    for k in range(4):
        Q_t[sel, k] = np.interp(sel, idx, Q_t[idx, k])
    Q_t[sel] /= np.linalg.norm(Q_t[sel], axis=1, keepdims=True)

    arm = ArmModel(args.side)
    arm.reachable_orient = args.reachable_orient
    arm.park_other(args.other_pose)
    arm.soften_limits(args.limit_margin_deg)
    sign = -1.0 if args.side == "left" else 1.0
    q_safe = None
    q = SIM_HOME6[:5].copy()
    rows, flags = [], {"floor_clamped": 0, "midline_clamped": 0, "ik_miss": 0, "clearance_held": 0}
    # Pivot calibration (umi/pivot_calib.py): the real fingertip contact point sits at c_meas in the
    # handheld's cube-derived gripper frame, vs CAD_TIP on identical jaws. The jaws are the same parts
    # as the robot's, so the difference is where the cube really sits on the printed handle: shift the
    # handheld's gripper-body pose by it, so the robot's jaws go where the handheld's jaws really were.
    tip_corr = np.zeros(3)
    tip_file = REPO / "calibration" / "handheld_tip.json"
    if tip_file.exists() and not args.no_tip_calib:
        tj = json.loads(tip_file.read_text())
        if tj.get("good"):
            tip_corr = (np.array(tj["tip_in_gripper_mm"]) - np.array(tj["cad_tip_mm"])) / 1000.0
            print(f"handheld tip calibration: gripper pose shifted by {np.round(tip_corr * 1000, 1).tolist()} mm "
                  f"(gripper frame, from {tj['capture']})")
    gap_set = set(int(g) for g in gaps)
    for n, i in enumerate(sel):
        R_g = R_wt @ quat_xyzw_to_mat(Q_t[i])
        p_g = R_wt @ P_t[i] + T_wt[:3, 3] + R_g @ tip_corr
        tcp = p_g + R_g @ TCP_IN_GRIPPER
        tcp = tcp + np.array([args.shift_mm[0], args.shift_mm[1], 0.0]) / 1000.0
        demo_tcp = tcp.copy()                         # your hand, before any safety adjustment (for --optimize)
        R_demo = R_g                                  # your wrist, as tracked (stats compare against this)
        # --grasp-orient changes only what the whole-demo optimizer aims at (smoothness is part of its cost);
        # the frame-by-frame stage and its safety search keep aiming at your wrist as is.
        R_target = arm.grasp_target(tcp, R_demo) if args.grasp_orient else R_demo
        clamp, wanted_x = [], float(tcp[0])
        if int(i) in gap_set:
            clamp.append("HANDLE NOT SEEN: path filled in between tracked frames")
        # Safety shaping of the TARGET (reported, never hidden):
        if args.midline_mm is not None and sign * tcp[0] < args.midline_mm / 1000.0:  # optional blanket rule
            tcp[0] = sign * args.midline_mm / 1000.0
            flags["midline_clamped"] += 1
            clamp.append(f"HELD BACK - MIDLINE: demo {abs(wanted_x - tcp[0]) * 1000:.0f} mm into the other side")
        q_prev = q.copy()
        step = None if n == 0 else args.max_step_deg
        q, pos_e, ang_e = arm.ik(tcp, R_g, q, iters=150 if n == 0 else 40, max_step_deg=step)
        # ONE RULE for every frame: if the demo pose isn't safe for this arm, use the NEAREST pose that is, and
        # say how far off it is and why. "Safe" = reachable at about the demo's tilt, every joint can hold it
        # (gravity torque), the hand clears the table floor, and it keeps clear of the other arm. (Joint margins
        # are built into the IK.) Search order: the demo pose -> straight back toward the shoulder (the common
        # "too far" case, cheap) -> shells around the demo pose in 14 directions (too close / too high / behind
        # the base / tilt the arm can't make), then refined along the best direction.
        q_seed = q_prev if n else q
        tol = max(args.reach_tol_mm, 12.0)          # a 5-joint arm misses the demo's tilt by a few mm anyway
        floor = args.floor_mm / 1000.0

        def check(tgt, rot_w=0.15, hard_only=False, allow_jump=False):
            # Callers only ask WHETHER a candidate fails, so stop at the first failed test: cheapest first,
            # the other-arm clearance (by far the most expensive) last.
            qq, pe, ae = arm.ik(tgt, R_g, q_seed, iters=50, max_step_deg=None, rot_w=rot_w)
            if pe * 1000 > tol and not hard_only:
                return ["reach/tilt"], qq, pe, ae
            if n and not allow_jump and np.abs(qq - q_seed).max() > args.max_step_deg:
                return ["joint jump"], qq, pe, ae    # another arm configuration: never switch mid-motion
            if arm.hold_torque().max() > args.max_hold_nm:
                return ["servo torque"], qq, pe, ae
            if arm.hand_lowest_z() < floor:
                return ["table"], qq, pe, ae
            if arm.clearance_to_other() * 1000 < args.clearance_mm:
                return ["other arm"], qq, pe, ae
            return [], qq, pe, ae

        why0 = []
        if pos_e * 1000 > tol:
            why0.append("reach/tilt")
        if arm.hold_torque().max() > args.max_hold_nm:
            why0.append("servo torque")
        if arm.hand_lowest_z() < floor:
            why0.append("table")
        if arm.clearance_to_other() * 1000 < args.clearance_mm:
            why0.append("other arm")
        if why0:
            best = None                               # (distance, target, q, pe, ae)
            sh = arm.shoulder_xy()
            # 0) same spot, different tilt: near the base / up high a 5-joint arm can often reach the point but
            #    not with the demo's tilt. Position first, tilt as close as it allows.
            if "reach/tilt" in why0:
                w, qq, pe, ae = check(tcp, rot_w=0.02)
                if not w:
                    best = (0.0, tcp.copy(), qq, pe, ae)
                    why0 = [x if x != "reach/tilt" else "tilt" for x in why0]
            # 1) straight back toward the shoulder, same height (+ a lift if the table is the problem)
            for sc in (np.arange(0.975, 0.39, -0.025) if best is None else []):
                tgt = tcp.copy()
                tgt[:2] = sh + sc * (tcp[:2] - sh)
                if "table" in why0:
                    tgt[2] = max(tgt[2], floor + 0.006)
                w, qq, pe, ae = check(tgt)
                if not w:
                    best = (float(np.linalg.norm(tgt - tcp)), tgt, qq, pe, ae)
                    break
            # 2) shells around the demo pose, all directions; keep the nearest safe point
            back = np.r_[sh - tcp[:2], 0.0]
            back = back / (np.linalg.norm(back) + 1e-9)        # horizontal, toward the shoulder
            up = np.array([0, 0, 1.0])
            side = np.cross(up, back)
            dirs = [back, (back + up) / np.sqrt(2), up, (back + up + side) / np.sqrt(3), (back + up - side) / np.sqrt(3),
                    (back + side) / np.sqrt(2), (back - side) / np.sqrt(2), (up + side) / np.sqrt(2), (up - side) / np.sqrt(2),
                    side, -side, -back, (-back + up) / np.sqrt(2), -up]   # toward you / up first, away / down last
            radii = [0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
            for r_ in radii:
                if best is not None and best[0] <= r_:
                    break                             # nothing at this radius can beat what we have
                found = []
                for dv in dirs:
                    tgt = tcp + dv * r_
                    w, qq, pe, ae = check(tgt)
                    if not w:
                        found.append((r_, tgt, qq, pe, ae, dv))
                if found:
                    r_b, tgt_b, qq_b, pe_b, ae_b, dv_b = found[0]      # most preferred direction
                    lo_r = radii[radii.index(r_) - 1] if radii.index(r_) else 0.0
                    for _ in range(5):                # refine: closest safe point along that direction
                        mid = (lo_r + r_b) / 2
                        w, qq, pe, ae = check(tcp + dv_b * mid)
                        if not w:
                            r_b, tgt_b, qq_b, pe_b, ae_b = mid, tcp + dv_b * mid, qq, pe, ae
                        else:
                            lo_r = mid
                    if best is None or r_b < best[0]:
                        best = (r_b, tgt_b, qq_b, pe_b, ae_b)
                    break
            if best is not None:
                dist, tgt, q, _, ang_e = best
                arm.set(q)
                pos_e = float(np.linalg.norm(tcp - arm.tcp_world()))
                tcp = tgt
                flags["safe_pose_used"] = flags.get("safe_pose_used", 0) + 1
                for w in why0:
                    flags[f"unsafe_{w.replace('/', '_').replace(' ', '_')}"] = flags.get(f"unsafe_{w.replace('/', '_').replace(' ', '_')}", 0) + 1
                clamp.append(f"TILT CHANGED (same spot): {ang_e:.0f} deg from your wrist tilt" if dist == 0.0 else
                             f"NEAREST SAFE POSE: {dist * 1000:.0f} mm from your hand ({', '.join(why0)})")
            else:
                # Nothing matches the demo safely: don't stop. Keep only the hard limits (table, servo torque,
                # other arm, joint margins) and go as close to the demo as those allow, then carry on.
                cands = [tcp.copy(), tcp + np.array([0, 0, 0.03])]
                for sc in (0.9, 0.8, 0.7, 0.6, 0.5):
                    t2 = tcp.copy()
                    t2[:2] = sh + sc * (tcp[:2] - sh)
                    cands += [t2, t2 + np.array([0, 0, 0.03]), t2 + np.array([0, 0, 0.08])]
                poss = None
                for tgt in cands:
                    w, qq, pe, ae = check(tgt, rot_w=0.02, hard_only=True, allow_jump=True)
                    if not w:
                        arm.set(qq)
                        miss = float(np.linalg.norm(tcp - arm.tcp_world()))
                        if poss is None or miss < poss[0]:
                            poss = (miss, tgt, qq, ae)
                if poss is not None:
                    miss, tgt, q, ang_e = poss
                    if n:   # head there at most max_step per frame (no jump), keeping every hard limit on the way
                        stepped = q_seed + np.clip(q - q_seed, -args.max_step_deg, args.max_step_deg)

                        def hard_ok(qx):
                            arm.set(qx)
                            return (arm.hand_lowest_z() >= floor and arm.hold_torque().max() <= args.max_hold_nm
                                    and arm.clearance_to_other() * 1000 >= args.clearance_mm)
                        lo_a, hi_a = 0.0, 1.0
                        if hard_ok(stepped):
                            lo_a = 1.0
                        else:
                            for _ in range(8):
                                mid = (lo_a + hi_a) / 2
                                if hard_ok(q_seed + mid * (stepped - q_seed)):
                                    lo_a = mid
                                else:
                                    hi_a = mid
                        q = q_seed + lo_a * (stepped - q_seed)
                    arm.set(q)
                    miss = float(np.linalg.norm(tcp - arm.tcp_world()))
                    pos_e = miss
                    tcp = tgt
                    flags["closest_possible"] = flags.get("closest_possible", 0) + 1
                    clamp.append(f"CLOSEST POSSIBLE: {miss * 1000:.0f} mm from your hand ({', '.join(why0)}); continuing")
                else:
                    q = q_seed.copy()                 # truly nothing safe nearby: keep the last safe pose
                    arm.set(q)
                    pos_e = float(np.linalg.norm(tcp - arm.tcp_world()))
                    flags["no_safe_pose"] = flags.get("no_safe_pose", 0) + 1
                    clamp.append(f"NO SAFE POSE ({', '.join(why0)}): kept the last safe pose")
        low = arm.hand_lowest_z()
        if low < args.floor_mm / 1000.0:                  # lift the target until the hand clears the floor
            flags["floor_clamped"] += 1
            demo_low, z0 = low, float(tcp[2])
            # Smallest lift that clears the floor (bisection; each try solved from the previous frame's joints,
            # so the arm stays continuous). A 5-joint arm can't keep the demo's exact tilt, so lifting the
            # target by d doesn't lift the hand by d; searching avoids the old overshoot (and joint jumps).
            floor = args.floor_mm / 1000.0

            def lifted(d):
                qq, pe, ae = arm.ik(tcp + np.array([0, 0, d]), R_g, q_prev, iters=40, max_step_deg=None)
                return qq, pe, ae, arm.hand_lowest_z()
            lo_d, hi_d = 0.0, 0.005
            best = None
            while hi_d <= 0.08:
                r_hi = lifted(hi_d)
                if r_hi[3] >= floor:
                    best = (hi_d, r_hi)
                    break
                lo_d, hi_d = hi_d, hi_d * 2
            if best is not None:
                for _ in range(6):
                    mid = (lo_d + best[0]) / 2
                    r_mid = lifted(mid)
                    if r_mid[3] >= floor:
                        best = (mid, r_mid)
                    else:
                        lo_d = mid
                d_lift, (q, pos_e, ang_e, low) = best
            else:
                d_lift, (q, pos_e, ang_e, low) = 0.08, lifted(0.08)
            tcp = tcp + np.array([0, 0, d_lift])
            arm.set(q)
            if step is not None:                                  # the lift may not jump either...
                q_free = q.copy()
                q = np.clip(q, q_prev - step, q_prev + step)
                arm.set(q)
                if arm.hand_lowest_z() < floor:                   # ...unless only a jump clears the floor
                    q = q_free
                    arm.set(q)
                    flags["floor_jump"] = flags.get("floor_jump", 0) + 1
                    clamp.append("BIG JOINT STEP: needed to keep the hand above the floor")
                pos_e = float(np.linalg.norm(tcp - arm.tcp_world()))
                low = arm.hand_lowest_z()
            # The DEMO (as measured) was below the robot's safety floor; the robot itself is raised and
            # never goes below it. Say so plainly, with numbers.
            lift_mm = (float(tcp[2]) - z0) * 1000
            clamp.append(f"DEMO NEAR/BELOW TABLE: your TCP measured at {z0 * 1000:.0f} mm -> "
                         + (f"robot raised {lift_mm:.0f} mm" if lift_mm >= 1 else "robot kept >= 10 mm"))
        gap = arm.clearance_to_other()
        if gap * 1000 < args.clearance_mm and q_safe is not None:   # would come too close to the other arm
            flags["clearance_held"] = flags.get("clearance_held", 0) + 1
            clamp.append(f"HELD BACK - CLEARANCE: {gap * 1000:.0f} mm from the other arm")
            q = q_safe.copy()
            arm.set(q)
            gap = arm.clearance_to_other()
            pos_e = float(np.linalg.norm(tcp - arm.tcp_world()))
            low = arm.hand_lowest_z()
        else:
            q_safe = q.copy()
        if pos_e * 1000 > args.max_ik_mm:
            flags["ik_miss"] += 1
        rows.append({"t": float(t[i] - t[i0]), "q": q.copy(), "pos_err_mm": pos_e * 1000, "ang_err_deg": ang_e,
                     "hand_low_mm": low * 1000, "tcp_target": tcp.copy(), "tcp_reached": arm.tcp_world(),
                     "clamp": clamp, "gap_mm": gap * 1000, "hold_nm": arm.hold_torque().tolist(),
                     "demo_tcp": demo_tcp, "demo_R": R_demo.copy(), "target_R": R_target.copy(), "tracked": int(i) not in gap_set})

    Q5 = np.array([r["q"] for r in rows])
    opt_info = None
    if args.optimize:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import trajopt
        Q5, opt_info = trajopt.optimize(
            arm, Q5, np.array([r["demo_tcp"] for r in rows]), np.array([r["target_R"] for r in rows]),
            [r["tracked"] for r in rows], TCP_IN_GRIPPER, floor_m=args.floor_mm / 1000.0,
            max_hold_nm=args.max_hold_nm, max_nfev=args.opt_iters)
        # Strict re-check with the FULL hand meshes: the optimizer treats the limits as heavy penalties on a
        # vertex subsample, so a frame can end a hair outside. Blend any such frame back toward the (safe)
        # frame-by-frame pose, only as far as needed.
        Q_safe = np.array([r["q"] for r in rows])
        floor_m = args.floor_mm / 1000.0

        def hard_ok(qx):
            # the optimizer's cost has no other-arm term, so clearance is enforced here (the
            # frame-by-frame pose it blends back toward already keeps it)
            arm.set(qx)
            return (arm.hand_lowest_z() >= floor_m and arm.hold_torque().max() <= args.max_hold_nm
                    and arm.clearance_to_other() * 1000 >= args.clearance_mm)
        n_fix = 0
        for kk in range(len(Q5)):
            if hard_ok(Q5[kk]):
                continue
            lo_a, hi_a = 0.0, 1.0                     # a = how much of the optimizer's change to keep
            for _ in range(10):
                mid = (lo_a + hi_a) / 2
                if hard_ok(Q_safe[kk] + mid * (Q5[kk] - Q_safe[kk])):
                    lo_a = mid
                else:
                    hi_a = mid
            Q5[kk] = Q_safe[kk] + lo_a * (Q5[kk] - Q_safe[kk])
            n_fix += 1
        opt_info["frames_pulled_back_to_hard_limits"] = n_fix
        if n_fix:
            print(f"  optimizer: {n_fix} frames pulled back slightly to respect the floor / torque exactly")
        for kk, r in enumerate(rows):
            arm.set(Q5[kk])
            r.update(q=Q5[kk].copy(), tcp_reached=arm.tcp_world(), hand_low_mm=arm.hand_lowest_z() * 1000,
                     pos_err_mm=float(np.linalg.norm(r["tcp_target"] - arm.tcp_world())) * 1000,
                     hold_nm=arm.hold_torque().tolist())
        args.smooth_s = 0.0                           # smoothness is part of the optimization
    raw_step = float(np.abs(np.diff(Q5, axis=0)).max()) if len(Q5) > 1 else 0.0
    n_floor_kept = 0
    if args.smooth_s > 0 and len(Q5) > 5:
        # Zero-phase Gaussian on each joint (no lag). Then re-check the floor: a frame whose smoothed pose
        # would put the hand below it keeps its unsmoothed (floor-checked) pose.
        sig = args.smooth_s * hz
        k = np.arange(-int(3 * sig) - 1, int(3 * sig) + 2)
        w = np.exp(-0.5 * (k / sig) ** 2)
        w /= w.sum()
        pad = len(k) // 2
        Qp = np.pad(Q5, ((pad, pad), (0, 0)), mode="edge")
        Qs = np.stack([np.convolve(Qp[:, j], w, mode="valid") for j in range(5)], 1)
        for kk, r in enumerate(rows):
            arm.set(Qs[kk])
            lo = arm.hand_lowest_z()
            if lo < args.floor_mm / 1000.0 and r["hand_low_mm"] >= lo * 1000:
                Qs[kk] = Q5[kk]
                n_floor_kept += 1
                arm.set(Qs[kk])
                lo = arm.hand_lowest_z()
            r.update(q=Qs[kk].copy(), tcp_reached=arm.tcp_world(), hand_low_mm=lo * 1000,
                     pos_err_mm=float(np.linalg.norm(r["tcp_target"] - arm.tcp_world())) * 1000)
        Q5 = Qs
    rt_dev, rt_floor = 0.0, 0
    if args.realtime_vmax > 0 and len(Q5) > 2:
        # Real-time playback, made gentle: every joint follows the plan under a speed AND acceleration cap
        # (the follower brakes ahead of each target, so no overshoot), run forward and backward in time and
        # averaged, so it neither lags nor leads the demo; only the fastest, most abrupt moments get rounded.
        def follow(Q):
            dt, out, v = 1.0 / hz, np.empty_like(Q), np.zeros(Q.shape[1])
            out[0] = Q[0]
            for kk in range(1, len(Q)):
                err = Q[kk] - out[kk - 1]
                vd = np.sign(err) * np.minimum.reduce([np.full_like(err, args.realtime_vmax),
                                                       np.sqrt(2 * args.realtime_amax * np.abs(err)), np.abs(err) / dt])
                v = v + np.clip(vd - v, -args.realtime_amax * dt, args.realtime_amax * dt)
                out[kk] = out[kk - 1] + v * dt
            return out
        Qr = 0.5 * (follow(Q5) + follow(Q5[::-1])[::-1])
        # Safety beats smoothness: where the limiter's rounding would put the hand under the floor or ask a
        # servo for more holding torque than the cap, blend back toward the (safe) pose it started from, only
        # as far as needed.
        floor = args.floor_mm / 1000.0
        for kk in range(len(Qr)):
            arm.set(Qr[kk])
            if arm.hand_lowest_z() >= floor and arm.hold_torque().max() <= args.max_hold_nm:
                continue
            lo_a, hi_a = 0.0, 1.0                     # a = how much of the limiter's change to keep
            for _ in range(8):
                mid = (lo_a + hi_a) / 2
                arm.set(Q5[kk] + mid * (Qr[kk] - Q5[kk]))
                if arm.hand_lowest_z() >= floor and arm.hold_torque().max() <= args.max_hold_nm:
                    lo_a = mid
                else:
                    hi_a = mid
            Qr[kk] = Q5[kk] + lo_a * (Qr[kk] - Q5[kk])
            flags["realtime_safety_blend"] = flags.get("realtime_safety_blend", 0) + 1
        rt_dev = float(np.abs(Qr - Q5).max())
        for kk, r in enumerate(rows):
            arm.set(Qr[kk])
            lo = arm.hand_lowest_z()
            if lo < args.floor_mm / 1000.0:
                rt_floor += 1
            r.update(q=Qr[kk].copy(), tcp_reached=arm.tcp_world(), hand_low_mm=lo * 1000,
                     pos_err_mm=float(np.linalg.norm(r["tcp_target"] - arm.tcp_world())) * 1000)
        Q5 = Qr
    max_step = float(np.abs(np.diff(Q5, axis=0)).max()) if len(Q5) > 1 else 0.0
    grip = opening_to_grip_units(np.nan_to_num(tr["jaw_opening_mm"][sel], nan=0.0))
    gvalid = tr["jaw_valid"][sel].astype(bool)
    if gvalid.any():  # hold the last valid gripper value through gaps
        last = grip[np.argmax(gvalid)]
        for k in range(len(grip)):
            if gvalid[k]:
                last = grip[k]
            else:
                grip[k] = last
    vel = np.abs(np.diff(Q5, axis=0)) * hz
    peak = float(np.percentile(vel.max(1), 99)) if len(vel) else 0.0
    slow = max(1.0, peak / (args.realtime_vmax * 1.02 if args.realtime_vmax > 0 else args.max_deg_s))
    pe = np.array([r["pos_err_mm"] for r in rows])
    demo_miss = np.array([np.linalg.norm(r["tcp_reached"] - r["demo_tcp"]) * 1000 for r in rows if r["tracked"]])
    # Orientation vs YOUR hand on tracked frames, split into what a 5-DOF arm can't do (sideways yaw out of its
    # plane) and what it can (pitch + twist about the gripper's axis), measured on the final plan.
    ori = []
    qq_ = np.zeros(4)
    for kk, r in enumerate(rows):
        if not r["tracked"]:
            continue
        arm.set(Q5[kk])
        Rb = arm.gripper_pose()[1]
        Rt, yaw_drop = arm.reachable_R(r["demo_R"])
        Rrel = Rt.T @ Rb
        reach_err = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rrel) - 1) / 2))))
        arm.mj.mju_mat2Quat(qq_, np.ascontiguousarray(Rrel).ravel())
        tw = math.degrees(2 * math.atan2(float(qq_[1:] @ arm._roll_axis_b), qq_[0]))
        tw = abs((tw + 180.0) % 360.0 - 180.0)
        ori.append((yaw_drop, reach_err, tw))
    ori = np.array(ori) if ori else np.zeros((0, 3))

    def _stat(v):
        return {"median": round(float(np.median(v)), 1), "p95": round(float(np.percentile(v, 95)), 1)} if len(v) else None
    acc_rms = float(np.sqrt(np.mean(np.diff(Q5, n=2, axis=0) ** 2))) if len(Q5) > 2 else 0.0
    ae = np.array([r["ang_err_deg"] for r in rows])
    low = np.array([r["hand_low_mm"] for r in rows])
    lim = np.rad2deg(getattr(arm, "hard_limits", arm.limits))
    margin = np.minimum(Q5 - lim[:, 0], lim[:, 1] - Q5).min(0)

    plan = {
        "episode": args.episode, "side": args.side,
        "units": "joints: pi_sim / so101_real units (deg, rig joint offsets applied by the runner's Arm.to_raw); gripper 0=closed..100=open",
        "joint_order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        "hz": hz, "recommended_time_scale": round(slow, 2),
        "t": [round(r["t"], 4) for r in rows],
        "q": [[round(float(v), 3) for v in (*r["q"], g)] for r, g in zip(rows, grip)],
        "frames": frame_info, "shift_mm": list(args.shift_mm), "tip_correction_mm": np.round(tip_corr * 1000, 2).tolist(),
        "checks": {
            "frames": len(rows), "gap_frames_filled": int(len(gaps)),
            "miss_vs_demo_mm": {"median": round(float(np.median(demo_miss)), 2), "p95": round(float(np.percentile(demo_miss, 95)), 2),
                                "max": round(float(demo_miss.max()), 2)} if len(demo_miss) else None,
            "joint_accel_rms_deg_per_frame2": round(acc_rms, 3), "optimizer": opt_info,
            "ik_pos_err_mm": {"median": round(float(np.median(pe)), 2), "p95": round(float(np.percentile(pe, 95)), 2), "max": round(float(pe.max()), 2)},
            "ik_ang_err_deg": {"median": round(float(np.median(ae)), 1), "p95": round(float(np.percentile(ae, 95)), 1)},
            "orient_vs_demo_deg": {"sideways_yaw_dropped_5dof": _stat(ori[:, 0]), "reachable_error": _stat(ori[:, 1]),
                                   "twist_error": _stat(ori[:, 2]),
                                   "target": "reachable part (pitch + twist)" if args.reachable_orient else "full demo orientation"},
            "hand_lowest_mm": round(float(low.min()), 1), "floor_mm": args.floor_mm,
            "min_gap_to_other_arm_mm": round(float(min(r["gap_mm"] for r in rows)), 1),
            "clearance_mm": args.clearance_mm, "other_arm_pose_deg": list(args.other_pose),
            "midline_mm": args.midline_mm,
            "joint_limit_margin_deg": [round(float(v), 1) for v in margin],
            "max_hold_torque_nm": [round(float(v), 2) for v in np.max([r["hold_nm"] for r in rows], 0)],
            "hold_torque_cap_nm": args.max_hold_nm, "limit_margin_kept_deg": args.limit_margin_deg,
            "peak_joint_speed_deg_s_p99": round(peak, 1), "speed_cap_deg_s": args.realtime_vmax if args.realtime_vmax > 0 else args.max_deg_s,
            "realtime": args.realtime_vmax > 0, "realtime_vmax_deg_s": args.realtime_vmax,
            "realtime_amax_deg_s2": args.realtime_amax, "realtime_max_change_deg": round(rt_dev, 1),
            "realtime_below_floor_frames": rt_floor,
            "max_joint_step_deg": round(max_step, 2), "max_joint_step_before_smoothing_deg": round(raw_step, 2),
            "step_cap_deg": args.max_step_deg, "smooth_s": args.smooth_s, "smoothing_floor_kept": n_floor_kept,
            **flags,
        },
        "clamped_spans_s": clamp_spans(rows),
        "notes": [
            "Targets were clamped where flagged (midline, floor); see checks.",
            "Gripper units are approximate: finger offset -> jaw angle via CAD lever, sim mapping -10..100 deg.",
            "Collision (arm-arm, mount post, box) and tracking checks are left to the rig's safety.Guard preflight.",
        ],
    }
    (ep / f"plan_{args.side}{args.tag}.json").write_text(json.dumps(plan, indent=1))
    if args.export_web:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import web_export
        meta = json.loads((ep / "meta.json").read_text())
        cap_dir = (REPO / meta["overhead_video"]).parent
        web_dir = web_export.export_episode(
            arm, ep, cap_dir, args.side, np.array(plan["t"]), float(t[i0]), Q5, grip,
            [r["demo_tcp"] for r in rows], [r["tcp_reached"] for r in rows], [r["tracked"] for r in rows],
            load_overhead(args.lens), {"checks": plan["checks"], "clamped_spans_s": plan.get("clamped_spans_s", []),
                                       "recommended_time_scale": plan["recommended_time_scale"]},
            args.other_pose, finder_cls=PropFinder)
        print(f"  browser replay data: {web_dir.relative_to(REPO)}/ (open viewer/replay.html?ep={args.episode}&side={args.side})")
    c = plan["checks"]
    print(f"\nplan_{args.side}{args.tag}.json: {len(rows)} frames @ {hz:.0f} Hz ({rows[-1]['t']:.1f} s demo)")
    print(f"  IK TCP error: median {c['ik_pos_err_mm']['median']} mm, p95 {c['ik_pos_err_mm']['p95']} mm, max {c['ik_pos_err_mm']['max']} mm "
          f"({c['ik_miss']} frames > {args.max_ik_mm:g} mm)")
    if c.get("miss_vs_demo_mm"):
        md = c["miss_vs_demo_mm"]
        print(f"  miss vs YOUR hand (tracked frames): median {md['median']} mm, p95 {md['p95']} mm, max {md['max']} mm; "
              f"joint jerkiness (accel rms) {c['joint_accel_rms_deg_per_frame2']} deg/frame^2")
    print(f"  orientation error: median {c['ik_ang_err_deg']['median']} deg, p95 {c['ik_ang_err_deg']['p95']} deg (5-DOF arm)")
    ov = c["orient_vs_demo_deg"]
    if ov["reachable_error"]:
        print(f"  vs YOUR wrist (tracked frames): sideways yaw the 5-DOF arm can't make: median {ov['sideways_yaw_dropped_5dof']['median']} deg, "
              f"p95 {ov['sideways_yaw_dropped_5dof']['p95']}; error on the reachable part (pitch + twist): median "
              f"{ov['reachable_error']['median']} deg, p95 {ov['reachable_error']['p95']}; twist alone: median "
              f"{ov['twist_error']['median']} deg, p95 {ov['twist_error']['p95']}")
    print(f"  hand lowest point {c['hand_lowest_mm']} mm (floor {args.floor_mm:g}); floor-clamped frames {c['floor_clamped']}; "
          f"midline-clamped {c['midline_clamped']}")
    print(f"  joint-limit margin (deg) {c['joint_limit_margin_deg']} (IK keeps >= {args.limit_margin_deg:g})")
    print(f"  largest holding torque per joint (Nm) {c['max_hold_torque_nm']} (cap {args.max_hold_nm:g}); "
          f"nearest safe pose: {c.get('safe_pose_used', 0)} frames, closest possible (demo out of range): "
          f"{c.get('closest_possible', 0)}, held: {c.get('no_safe_pose', 0)}")
    why_counts = {k[len("unsafe_"):]: v for k, v in c.items() if k.startswith("unsafe_")}
    if why_counts:
        print("  why the demo pose wasn't safe (frames): " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in why_counts.items()))
    print(f"  closest approach to the other arm (parked at {args.other_pose}): {c['min_gap_to_other_arm_mm']} mm "
          f"(need >= {args.clearance_mm:g}); held for clearance: {c.get('clearance_held', 0)} frames")
    for a, b, why in plan["clamped_spans_s"]:
        print(f"  held back ({why.lower()}): {a:.1f}-{b:.1f} s")
    if args.realtime_vmax > 0:
        print(f"  real time: joints capped at {args.realtime_vmax:g} deg/s and {args.realtime_amax:g} deg/s^2; plan changed by at most "
              f"{rt_dev:.1f} deg to fit; {rt_floor} frames below the floor after that")
    print(f"  largest joint step between frames: {max_step:.1f} deg (before smoothing {raw_step:.1f}, cap {args.max_step_deg:g}); "
          f"smoothing {args.smooth_s:g} s, {n_floor_kept} frames kept unsmoothed for the floor")
    print(f"  peak joint speed {c['peak_joint_speed_deg_s_p99']} deg/s at 1x -> play at {slow:.2f}x slower for <= {c['speed_cap_deg_s']:g} deg/s")

    # ---- plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tt = np.array(plan["t"])
        fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        for j, nme in enumerate(plan["joint_order"][:5]):
            ax[0].plot(tt, Q5[:, j], label=nme)
        ax[0].set_ylabel("joint (deg)"); ax[0].legend(fontsize=7, ncol=5); ax[0].grid(alpha=.3)
        ax[1].plot(tt, grip, color="#c0392b"); ax[1].set_ylabel("gripper (0-100)"); ax[1].grid(alpha=.3)
        ax[2].plot(tt, pe, label="TCP pos err (mm)"); ax[2].plot(tt, ae, label="orient err (deg)")
        ax[2].axhline(args.max_ik_mm, ls="--", c="k", lw=.7); ax[2].legend(); ax[2].grid(alpha=.3)
        ax[3].plot(tt, low, label="hand lowest (mm)"); ax[3].axhline(args.floor_mm, ls="--", c="r", lw=.8, label="floor")
        ax[3].legend(); ax[3].grid(alpha=.3); ax[3].set_xlabel("t (s, demo time)")
        fig.suptitle(f"{args.episode}: {args.side} arm plan"); fig.tight_layout(); fig.savefig(ep / f"plan_{args.side}{args.tag}.png", dpi=100)
    except ImportError:
        pass

    # ---- sim video from the rig's overhead camera, next to the real overhead frames
    if not args.no_video:
        try:
            render_video(arm, Q5, grip, rows, ep, args, t[i0])
        except Exception as e:  # rendering needs an OpenGL context
            print(f"  (video skipped: {e})")


def render_video(arm, Q5, grip, rows, ep, args, t_start):
    """Real overhead (undistorted) | sim through the same calibrated camera | 50/50 overlay.
    When the chain is right, the sim gripper lands on the handheld gripper in the overlay."""
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from convert import read_frames
    mj, m, d = arm.mj, arm.m, arm.d
    lens = load_overhead(args.lens)
    K, D = np.array(lens["K"], float), np.array(lens["D"], float)
    W, H = lens["image_size"]
    # The model's "overhead" camera already IS the rig's calibrated camera (it sits on the
    # overhead_camera_mount body at the calibrated pose, with these intrinsics), so leave it as is
    # and only remove the real lens distortion so both images share one ideal pinhole.
    newK = K.copy()
    m1, m2 = cv2.initUndistortRectifyMap(K, D, None, newK, (W, H), cv2.CV_16SC2)
    meta = json.loads((ep / "meta.json").read_text())
    real = read_frames(REPO / meta["overhead_video"])
    nxt = next(real, None)
    renderer = mj.Renderer(m, H, W)
    # 2x2 layout: real | sim (same camera)  /  overlay | side view at table height
    out = cv2.VideoWriter(str(ep / f"plan_{args.side}{args.tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (2 * W, 2 * H))
    side = mj.MjvCamera()
    side.type = mj.mjtCamera.mjCAMERA_FREE
    side.distance, side.elevation = 0.42, -4.0
    side.azimuth = 0.0 if args.side == "right" else 180.0   # look along the table from outside the arm
    names = [m.joint(i).name for i in range(m.njnt)]
    grip_qpos = m.jnt_qposadr[m.joint(f"{args.side}_gripper").id] if f"{args.side}_gripper" in names else None
    real_frame = np.zeros((H, W, 3), np.uint8)
    finder = PropFinder(lens)
    early = [f for _, f in zip(range(45), read_frames(REPO / meta["overhead_video"]))]
    boxes = [b for b in (finder.box(f[1]) for f in early) if b is not None]
    box_pose = (np.median([b[0] for b in boxes], 0), float(np.median([b[1] for b in boxes]))) if boxes else (None, 0.0)
    print(f"  props: grey box {'at ' + str(np.round(box_pose[0] * 1000).astype(int).tolist()) + ' mm' if boxes else 'not found'}"
          f" (seen in {len(boxes)}/{len(early)} early frames); red cube tracked per frame")
    cube_last = (None, 0.0)
    for k, r in enumerate(rows):
        d.qpos[arm.qpos] = np.deg2rad(Q5[k])
        if grip_qpos is not None:
            d.qpos[grip_qpos] = SIM_GRIP_LO + grip[k] / 100.0 * (SIM_GRIP_HI - SIM_GRIP_LO)
        want = (t_start + r["t"]) * 1000.0
        while nxt is not None and nxt[0] * 1000.0 <= want:
            real_frame = nxt[1]
            nxt = next(real, None)
        cube = finder.cube(real_frame) if real_frame.any() else None
        cube_last = cube or cube_last
        arm.place_props(cube_xy=cube_last[0], cube_yaw=cube_last[1], box_xy=box_pose[0], box_yaw=box_pose[1])
        mj.mj_forward(m, d)
        renderer.update_scene(d, camera="overhead")
        sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        real_u = cv2.remap(real_frame, m1, m2, cv2.INTER_LINEAR)
        blend = cv2.addWeighted(real_u, 0.5, sim, 0.5, 0)
        txt = f"t={r['t']:5.1f}s  ik {r['pos_err_mm']:4.1f}mm  grip {grip[k]:4.0f}"
        for img, label in ((real_u, "REAL overhead (handheld demo)"), (sim, f"SIM {args.side} arm, same camera"),
                           (blend, "OVERLAY: sim gripper should sit on the handheld")):
            cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(sim, txt, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for n_c, msg in enumerate(r.get("clamp", [])):
            for img in (sim, blend):
                cv2.rectangle(img, (0, H - 26 - 22 * n_c), (W, H - 4 - 22 * n_c), (0, 0, 160), -1)
                cv2.putText(img, msg, (8, H - 10 - 22 * n_c), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)
        tcp = r["tcp_reached"]
        side.lookat[:] = [tcp[0], tcp[1], 0.06]
        renderer.update_scene(d, camera=side)
        sv = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        low_mm = r["hand_low_mm"]
        col = (0, 200, 0) if low_mm >= args.floor_mm else (0, 0, 255)
        cv2.putText(sv, "SIDE VIEW at table height (table top = z 0)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(sv, f"lowest point of the hand: {low_mm:5.1f} mm above the table", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.putText(sv, f"TCP height {tcp[2] * 1000:5.1f} mm   (robot never goes below {args.floor_mm:g} mm)", (8, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if r.get("clamp"):
            cv2.rectangle(sv, (0, H - 26), (W, H - 4), (0, 0, 160), -1)
            cv2.putText(sv, r["clamp"][0], (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        out.write(np.vstack([np.hstack([real_u, sim]), np.hstack([blend, sv])]))
    out.release()
    print(f"  wrote {ep.relative_to(REPO)}/plan_{args.side}{args.tag}.mp4")


if __name__ == "__main__":
    main()
