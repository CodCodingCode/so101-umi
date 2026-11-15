"""Whole-demo trajectory optimization for one SO-101 arm (used by to_robot_fast.py --optimize).

Same problem as trajopt.py (same unknowns, residuals, weights and bounds), solved faster:

  * the hand's lowest point is EXACT and cheap: arm.lowest_point() (to_robot_fast.ArmModel) keeps, per hand
    body, only the convex-hull vertices of all its geoms (the lowest point along any direction is always a
    hull vertex), evaluated with one small matmul per body. trajopt.py used an 80-vertex-per-mesh subsample
    in a Python loop, which was both the main cost (~70 % of the solve) and only approximate (so tens of
    frames had to be pulled back afterwards);
  * floor gradient analytic: mj_jac at the lowest vertex (instead of 5 finite-difference passes);
  * residuals and Jacobian come out of ONE pass over the frames (cached for the accepted point), instead
    of a separate Jacobian pass;
  * the safety hinges aim a hair inside the limits (floor + 0.2 mm, torque cap - 0.005 Nm), so the
    caller's exact re-check rarely has anything to pull back;
  * x_scale="jac" (columns scaled by the Jacobian): better conditioned, more accurate in the same 40 evaluations.

Unknowns: every frame's 5 arm joint angles at once (N x 5). One least-squares cost:

    tracking    TCP close to the demo's TCP (mm), gripper tilt close to the demo's (deg, lower weight)
    smoothness  joint velocity and acceleration between frames (deg/frame, deg/frame^2)
    floor       hinge: hand's lowest point >= floor (mm), heavy weight                 -- safety
    torque      hinge: gravity holding torque <= cap (Nm), heavy weight                -- safety
    limits      hard bounds: 5 deg inside every joint's range (the arm's soft limits)
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix


def _rotvec_deg(R):
    c = np.clip((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2, -1.0, 1.0)
    ang = np.arccos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return np.degrees(ang) * v


def optimize(arm, Q0, demo_tcp, demo_R, tracked, tcp_in_gripper, *, floor_m, max_hold_nm,
             w_pos=1.0, w_rot=0.25, w_vel=0.6, w_acc=3.0, w_floor=60.0, w_torque=400.0, max_nfev=40,
             floor_margin_m=0.0002, torque_margin_nm=0.005, ftol=1e-8, x_scale="jac", log=print):
    """Return (Q, info). Q0: N x 5 initial joints (deg). demo_tcp: N x 3 (m). demo_R: N x 3 x 3.
    tracked: N bools (False = handle not seen: no tracking term, only smoothness/safety)."""
    N = len(Q0)
    lo_lim = np.rad2deg(arm.limits[:, 0])
    hi_lim = np.rad2deg(arm.limits[:, 1])
    x0 = np.clip(np.asarray(Q0, float), lo_lim + 1e-6, hi_lim - 1e-6).ravel()
    tr = np.asarray(tracked, bool)
    tr_idx = np.where(tr)[0]
    n_tr = len(tr_idx)
    tr_pos = np.full(N, -1)
    tr_pos[tr_idx] = np.arange(n_tr)
    floor_t = floor_m + floor_margin_m
    cap_t = max_hold_nm - torque_margin_nm
    mj, m, d = arm.mj, arm.m, arm.d
    dof, body = arm._dof_ix, arm.body
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    d2r = np.pi / 180.0
    tcp_g = np.asarray(tcp_in_gripper, float)
    demo_RT = np.ascontiguousarray(np.transpose(np.asarray(demo_R, float), (0, 2, 1)))
    demo_tcp = np.asarray(demo_tcp, float)
    xmat_b = d.xmat[body]
    xpos_b = d.xpos[body]
    FD = 0.05                                          # deg, torque hinge finite difference

    # ---- residual rows: [pos 3*n_tr | rot 3*n_tr | vel 5*(N-1) | acc 5*(N-2) | floor N | torque 5N]
    off_pos, off_rot = 0, 3 * n_tr
    off_vel = 6 * n_tr
    off_acc = off_vel + 5 * (N - 1)
    off_floor = off_acc + 5 * (N - 2)
    off_tq = off_floor + N
    n_res = off_tq + 5 * N

    # ---- fixed sparsity pattern (values filled per evaluation), in this order:
    #      pos (n_tr,3,5) | rot (n_tr,3,5) | vel const | acc const | floor (N,5) | torque (N,5,5)
    a3, j5 = np.arange(3), np.arange(5)
    r_pos = (off_pos + 3 * np.arange(n_tr)[:, None, None] + a3[None, :, None] + 0 * j5).ravel()
    c_trk = (5 * tr_idx[:, None, None] + 0 * a3[None, :, None] + j5[None, None, :]).ravel()
    r_rot = r_pos - off_pos + off_rot
    kv, jv = np.meshgrid(np.arange(N - 1), j5, indexing="ij")
    r_v = (off_vel + 5 * kv + jv).ravel()
    r_vel = np.r_[r_v, r_v]
    c_vel = np.r_[(5 * kv + jv).ravel(), (5 * (kv + 1) + jv).ravel()]
    v_vel = np.r_[np.full(r_v.size, -w_vel), np.full(r_v.size, w_vel)]
    ka, ja = np.meshgrid(np.arange(N - 2), j5, indexing="ij")
    r_a = (off_acc + 5 * ka + ja).ravel()
    r_acc = np.r_[r_a, r_a, r_a]
    c_acc = np.r_[(5 * ka + ja).ravel(), (5 * (ka + 1) + ja).ravel(), (5 * (ka + 2) + ja).ravel()]
    v_acc = np.r_[np.full(r_a.size, w_acc), np.full(r_a.size, -2 * w_acc), np.full(r_a.size, w_acc)]
    kf = np.arange(N)
    r_fl = np.repeat(off_floor + kf, 5)
    c_fl = (5 * kf[:, None] + j5[None, :]).ravel()
    r_tq = (off_tq + 5 * kf[:, None, None] + j5[None, :, None] + 0 * j5[None, None, :]).ravel()
    c_tq = (5 * kf[:, None, None] + 0 * j5[None, :, None] + j5[None, None, :]).ravel()
    rows = np.r_[r_pos, r_rot, r_vel, r_acc, r_fl, r_tq]
    cols = np.r_[c_trk, c_trk, c_vel, c_acc, c_fl, c_tq]
    const_vals = np.r_[v_vel, v_acc]

    cache = {"x": None}

    def evaluate(x):
        """Residuals AND Jacobian blocks in one pass over the frames."""
        Q = x.reshape(N, 5)
        pos = np.zeros((n_tr, 3))
        rot = np.zeros((n_tr, 3))
        floor = np.zeros(N)
        torque = np.zeros((N, 5))
        Jpos = np.zeros((n_tr, 3, 5))
        Jrot = np.zeros((n_tr, 3, 5))
        Jfl = np.zeros((N, 5))
        Jtq = np.zeros((N, 5, 5))
        for k in range(N):
            arm.set(Q[k])
            tq = arm.hold_torque()                     # mj_inverse also runs mj_comPos, which mj_jac needs
            n = tr_pos[k]
            if n >= 0:
                Rb = xmat_b.reshape(3, 3)
                point = xpos_b + Rb @ tcp_g
                pos[n] = (point - demo_tcp[k]) * (1000.0 * w_pos)
                rot[n] = _rotvec_deg(demo_RT[k] @ Rb) * w_rot
                mj.mj_jac(m, d, jp, jr, point, body)
                Jpos[n] = jp[:, dof] * (1000.0 * w_pos * d2r)          # mm per degree
                Jrot[n] = (demo_RT[k] @ jr[:, dof]) * w_rot             # deg per degree (small-error approx.)
            lowp = arm.lowest_point_below(floor_t)
            if lowp is not None and lowp[0] < floor_t:
                low, lb, lp = lowp
                floor[k] = (floor_t - low) * (1000.0 * w_floor)
                mj.mj_jac(m, d, jp, jr, lp, lb)                         # lowest vertex moves with its body
                Jfl[k] = -jp[2, dof] * (1000.0 * w_floor * d2r)
            h0 = np.maximum(0.0, tq - cap_t)
            torque[k] = h0 * w_torque
            if tq.max() > cap_t - 0.05:                # torque hinge active or about to be: finite differences
                for j in range(5):
                    qq = Q[k].copy()
                    qq[j] += FD
                    t1 = arm.hold_torque(qq)
                    Jtq[k, :, j] = (np.maximum(0.0, t1 - cap_t) - h0) * (w_torque / FD)
        vel = np.diff(Q, axis=0).ravel() * w_vel
        acc = np.diff(Q, n=2, axis=0).ravel() * w_acc
        r = np.concatenate([pos.ravel(), rot.ravel(), vel, acc, floor, torque.ravel()])
        vals = np.concatenate([Jpos.ravel(), Jrot.ravel(), const_vals, Jfl.ravel(), Jtq.ravel()])
        cache["x"], cache["r"], cache["vals"] = x.copy(), r, vals
        return r

    costs = []

    def residuals(x):
        r = evaluate(x)
        costs.append(0.5 * float(r @ r))
        return r

    def jacobian(x):
        if cache["x"] is None or not np.array_equal(cache["x"], x):
            evaluate(x)
        return coo_matrix((cache["vals"], (rows, cols)), shape=(n_res, 5 * N)).tocsr()

    t0 = time.time()
    r0 = evaluate(x0)
    lo_b, hi_b = np.tile(lo_lim, N), np.tile(hi_lim, N)
    sol = least_squares(residuals, x0, jac=jacobian, bounds=(lo_b, hi_b), method="trf", x_scale=x_scale,
                        max_nfev=max_nfev, ftol=ftol, verbose=0)
    x, cost, nfev, status = sol.x, sol.cost, sol.nfev, sol.status
    Q = x.reshape(N, 5)
    info = {"cost_before": float(0.5 * r0 @ r0), "cost_after": float(cost), "nfev": int(nfev),
            "seconds": round(time.time() - t0, 1), "status": int(status)}
    info["_costs"] = costs
    log(f"  trajectory optimization: cost {info['cost_before']:.0f} -> {info['cost_after']:.0f} "
        f"({info['nfev']} evaluations, {info['seconds']} s)")
    return Q, info
