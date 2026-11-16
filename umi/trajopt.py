"""Whole-demo trajectory optimization for one SO-101 arm (used by to_robot.py --optimize).

Unknowns: every frame's 5 arm joint angles at once (N x 5). One least-squares cost:

    tracking    TCP close to the demo's TCP (mm), gripper tilt close to the demo's (deg, lower weight:
                a 5-joint arm can't lean sideways)                                    -- frames the handle was seen
    smoothness  joint velocity and acceleration between frames (deg/frame, deg/frame^2)
    floor       hinge: hand's lowest point >= floor (mm), heavy weight                 -- safety
    torque      hinge: gravity holding torque <= cap (Nm), heavy weight                -- safety
    limits      hard bounds: 5 deg inside every joint's range (the arm's soft limits)

Because all frames are solved together, frames before an unreachable stretch "feel" the cost of the jump
that would follow, so the arm eases out early instead of swerving late. Started from the greedy plan;
the caller re-checks every hard limit afterwards.

Each residual depends only on its own frame's joints (tracking, floor, torque) or on 2-3 neighbouring
frames (smoothness), so the Jacobian is banded and scipy's least_squares evaluates it cheaply.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def _rotvec_deg(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    ang = np.arccos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return np.degrees(ang) * v


class _FastLow:
    """Lowest point of the hand from a subsample of each hand mesh's vertices (fast enough to call
    hundreds of thousands of times; the caller re-checks with the full meshes)."""

    def __init__(self, arm, max_verts=80):
        m = arm.m
        self.arm = arm
        self.items = []
        for g in arm.hand_geoms:
            if m.geom_type[g] != arm.mj.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            step = max(1, len(v) // max_verts)
            self.items.append((g, np.ascontiguousarray(v[::step])))

    def bound(self):
        d, m = self.arm.d, self.arm.m
        return min(float(d.geom_xpos[g][2] - m.geom_rbound[g]) for g, _ in self.items)

    def __call__(self):
        d = self.arm.d
        lo = np.inf
        for g, v in self.items:
            lo = min(lo, float((d.geom_xpos[g][2] + v @ d.geom_xmat[g].reshape(3, 3)[2]).min()))
        return lo


def optimize(arm, Q0, demo_tcp, demo_R, tracked, tcp_in_gripper, *, floor_m, max_hold_nm,
             w_pos=1.0, w_rot=0.25, w_vel=0.6, w_acc=3.0, w_floor=60.0, w_torque=400.0, max_nfev=40, log=print):
    """Return (Q, info). Q0: N x 5 initial joints (deg). demo_tcp: N x 3 (m). demo_R: N x 3 x 3.
    tracked: N bools (False = handle not seen: no tracking term, only smoothness/safety)."""
    N = len(Q0)
    lo_lim = np.rad2deg(arm.limits[:, 0])
    hi_lim = np.rad2deg(arm.limits[:, 1])
    x0 = np.clip(np.asarray(Q0, float), lo_lim + 1e-6, hi_lim - 1e-6).ravel()
    low_fn = _FastLow(arm)
    tr = np.asarray(tracked, bool)
    n_tr = int(tr.sum())

    def _target_R(k):
        # A 5-DOF arm can't yaw sideways out of its plane: aim only at the reachable part of the demo's
        # orientation (pitch + twist), from the arm's current pan (arm.set(Q[k]) has just run).
        if getattr(arm, "reachable_orient", False):
            return arm.reachable_R(demo_R[k])[0]
        return demo_R[k]

    # residual layout (rows): [pos 3*n_tr | rot 3*n_tr | vel 5*(N-1) | acc 5*(N-2) | floor N | torque 5N]
    def residuals(x):
        Q = x.reshape(N, 5)
        pos, rot, floor, torque = [], [], np.zeros(N), np.zeros((N, 5))
        for k in range(N):
            arm.set(Q[k])
            if tr[k]:
                pb, Rb = arm.gripper_pose()
                pos.append((pb + Rb @ tcp_in_gripper - demo_tcp[k]) * 1000.0 * w_pos)
                rot.append(_rotvec_deg(_target_R(k).T @ Rb) * w_rot)
            # Cheap, always-valid lower bound first: each hand part's centre height minus its bounding-sphere
            # radius. Only when that bound dips below the floor is the exact (mesh) lowest point computed.
            if low_fn.bound() < floor_m:
                floor[k] = max(0.0, floor_m - low_fn()) * 1000.0 * w_floor
            tq = arm.hold_torque()
            torque[k] = np.maximum(0.0, tq - max_hold_nm) * w_torque
        vel = np.diff(Q, axis=0).ravel() * w_vel
        acc = np.diff(Q, n=2, axis=0).ravel() * w_acc
        parts = [np.ravel(pos), np.ravel(rot), vel, acc, floor, torque.ravel()]
        return np.concatenate([p for p in parts if len(p)])

    # sparsity: which residual rows depend on which frame's variables
    n_res = 6 * n_tr + 5 * (N - 1) + 5 * (N - 2) + N + 5 * N
    S = lil_matrix((n_res, 5 * N), dtype=np.int8)
    row = 0
    tr_idx = np.where(tr)[0]
    for k in tr_idx:                                  # position
        S[row:row + 3, 5 * k:5 * k + 5] = 1
        row += 3
    for k in tr_idx:                                  # rotation
        S[row:row + 3, 5 * k:5 * k + 5] = 1
        row += 3
    for k in range(N - 1):                            # velocity: joint j of frames k, k+1
        for j in range(5):
            S[row, 5 * k + j] = 1
            S[row, 5 * (k + 1) + j] = 1
            row += 1
    for k in range(N - 2):                            # acceleration: frames k, k+1, k+2
        for j in range(5):
            for o in range(3):
                S[row, 5 * (k + o) + j] = 1
            row += 1
    for k in range(N):                                # floor
        S[row, 5 * k:5 * k + 5] = 1
        row += 1
    for k in range(N):                                # torque
        S[row:row + 5, 5 * k:5 * k + 5] = 1
        row += 5
    assert row == n_res

    # Exact Jacobian from MuJoCo (mj_jac at the TCP), instead of finite differences over the whole demo:
    # one pass over the frames per solver step. Floor / torque hinges are differenced only where active.
    mj, m, d = arm.mj, arm.m, arm.d
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    d2r = np.pi / 180.0
    vel_rows = np.arange(5 * (N - 1))
    off_pos, off_rot = 0, 3 * n_tr
    off_vel = 6 * n_tr
    off_acc = off_vel + 5 * (N - 1)
    off_floor = off_acc + 5 * (N - 2)
    off_tq = off_floor + N
    from scipy.sparse import coo_matrix
    # constant smoothness part
    rows_c, cols_c, vals_c = [], [], []
    for k in range(N - 1):
        for j in range(5):
            r_ = off_vel + 5 * k + j
            rows_c += [r_, r_]; cols_c += [5 * k + j, 5 * (k + 1) + j]; vals_c += [-w_vel, w_vel]
    for k in range(N - 2):
        for j in range(5):
            r_ = off_acc + 5 * k + j
            rows_c += [r_] * 3; cols_c += [5 * k + j, 5 * (k + 1) + j, 5 * (k + 2) + j]; vals_c += [w_acc, -2 * w_acc, w_acc]
    tr_pos = {k: n for n, k in enumerate(tr_idx)}

    def jacobian(x):
        Q = x.reshape(N, 5)
        rows, cols, vals = list(rows_c), list(cols_c), list(vals_c)
        for k in range(N):
            arm.set(Q[k])
            mj.mj_comPos(m, d)
            if tr[k]:
                pb, Rb = arm.gripper_pose()
                point = pb + Rb @ tcp_in_gripper
                mj.mj_jac(m, d, jp, jr, point, arm.body)
                Jp = jp[:, arm.dof] * 1000.0 * w_pos * d2r          # mm per degree
                Jr = (_target_R(k).T @ jr[:, arm.dof]) * w_rot       # deg per degree (small-error approximation)
                n = tr_pos[k]
                for a in range(3):
                    for j in range(5):
                        rows += [off_pos + 3 * n + a, off_rot + 3 * n + a]
                        cols += [5 * k + j, 5 * k + j]
                        vals += [Jp[a, j], Jr[a, j]]
            # hinge terms: finite differences only where they're active (or about to be)
            near_floor = low_fn.bound() < floor_m + 0.01
            tq0 = arm.hold_torque()
            near_tq = tq0.max() > max_hold_nm - 0.05
            if near_floor or near_tq:
                f0 = max(0.0, floor_m - low_fn()) if near_floor else 0.0
                for j in range(5):
                    qq = Q[k].copy(); qq[j] += 0.05
                    arm.set(qq)
                    if near_floor:
                        f1 = max(0.0, floor_m - low_fn())
                        if f1 != f0:
                            rows.append(off_floor + k); cols.append(5 * k + j); vals.append((f1 - f0) / 0.05 * 1000.0 * w_floor)
                    if near_tq:
                        t1 = arm.hold_torque()
                        for jj in range(5):
                            g = (max(0.0, t1[jj] - max_hold_nm) - max(0.0, tq0[jj] - max_hold_nm)) / 0.05 * w_torque
                            if g:
                                rows.append(off_tq + 5 * k + jj); cols.append(5 * k + j); vals.append(g)
        return coo_matrix((vals, (rows, cols)), shape=(n_res, 5 * N)).tocsr()

    t0 = time.time()
    r0 = residuals(x0)
    sol = least_squares(residuals, x0, jac=jacobian, bounds=(np.tile(lo_lim, N), np.tile(hi_lim, N)),
                        method="trf", x_scale=1.0, max_nfev=max_nfev, verbose=0)
    Q = sol.x.reshape(N, 5)
    info = {"cost_before": float(0.5 * r0 @ r0), "cost_after": float(sol.cost), "nfev": int(sol.nfev),
            "seconds": round(time.time() - t0, 1), "status": int(sol.status)}
    log(f"  trajectory optimization: cost {info['cost_before']:.0f} -> {info['cost_after']:.0f} "
        f"({info['nfev']} evaluations, {info['seconds']} s)")
    return Q, info
