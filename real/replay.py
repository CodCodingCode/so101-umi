"""Replay a SO-UMI joint plan (umi/to_robot.py -> plan_<side>.json) on ONE real SO-101 arm. Runs ON the rig PC
with monitor.py stopped (it holds the COM ports). Launched from the Mac by real/run.py.

    python replay.py plan_right.json              # DRY (default): read the arms, check everything, NO motion
    python replay.py plan_right.json --live       # asks you to type the arm's side, then moves it

It uses the rig's own safety stack, the same as real_clamp.py:
  - safety.Guard on the training sim's collision shapes (this repo's sim/, copied to the rig as sim_mac/) and each servo's real
    calibrated range. Before anything moves it PREFLIGHTS the whole motion: the ramp from where the arm is now to the
    plan's first pose, then every plan frame, with the other arm where it really is. Every tick it re-checks the
    commanded and measured pose, watches tracking and limits how far a command may lead the joint.
  - per-joint torque caps (safety.apply_torque_limits), a hand-height floor guard, the STOP file in the rig folder, Ctrl+C.
  - the plan is slowed so no arm joint moves faster than --max-deg-s (and never faster than the plan's own
    recommended_time_scale).
Only the plan's arm gets torque; the other arm is only read (for the collision check). Any stop folds the arm back
to rest and turns torque off. Logs to <rig folder>\\runs\\umi_replay_<ts>\\.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# The rig folder with the arm drivers (so101_real.py, safety.py, rt_control.py, rig.json). real/run.py uploads this
# script to <that folder>\umi\, so by default it is the parent of this script's folder; SOUMI_RIG_REAL_DIR overrides.
HERE = Path(os.environ.get("SOUMI_RIG_REAL_DIR") or Path(__file__).resolve().parents[1])
SIM = HERE / "sim_mac"                  # the Mac training sim (collision shapes for camera mounts / post / grippers)
sys.path[:0] = [str(SIM / "squeeze"), str(SIM), str(HERE)]
sys.meta_path[:] = [f for f in sys.meta_path if "__editable___pi_sim" not in getattr(f, "__module__", "")]
os.chdir(HERE)

import numpy as np                 # noqa: E402

import rt_control as RT            # noqa: E402
import safety as SF                # noqa: E402
import so101_real as R             # noqa: E402

STOP = HERE / "STOP"
SIDES = ("left", "right")
NJ = 5
REST_UP = np.array([-45.8, 45.9, 90.0])     # shoulder_lift, elbow_flex, wrist_flex of SIM_HOME (the raised pose)


def resample(t, q, scale, hz):
    """Plan frames (t seconds, q [N,6] units) -> commands at `hz` on a clock slowed by `scale`."""
    T = float(t[-1]) * scale
    tt = np.arange(0.0, T + 1e-9, 1.0 / hz)
    src = tt / scale
    return tt, np.stack([np.interp(src, t, q[:, j]) for j in range(q.shape[1])], axis=1)


def linear(a, b, n):
    return [a + (b - a) * f for f in np.linspace(0.0, 1.0, n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--live", action="store_true", help="move the arm (starts right away)")
    ap.add_argument("--confirm", action="store_true", help="with --live: ask for the arm's side to be typed before moving")
    ap.add_argument("--max-deg-s", type=float, default=45.0, help="peak arm-joint speed cap for the replay")
    ap.add_argument("--time-scale", type=float, default=None,
                    help="slow-down factor (default: the larger of the plan's recommendation and the speed cap's need)")
    ap.add_argument("--floor-cm", type=float, default=0.5,
                    help="never command the lowest hand point below this: the arm slides along it and continues")
    ap.add_argument("--max-ik-p95-mm", type=float, default=None,
                    help="optional: refuse plans whose IK misses the demo by more than this (p95). Off by default: "
                         "big misses are the planner's safety net swapping in the nearest safe pose, which is what "
                         "we want the arm to do. Safety (floor, speed, joint limits, safety.Guard) is checked separately.")
    ap.add_argument("--no-gripper", action="store_true", help="keep the jaw where it is (only the 5 arm joints move)")
    a = ap.parse_args()

    plan = json.loads(Path(a.plan).read_text())
    side = plan["side"]
    if side not in SIDES:
        raise SystemExit(f"plan side {side!r}?")
    t = np.asarray(plan["t"], float)
    q = np.asarray(plan["q"], float)
    ck = plan.get("checks", {})
    print(f"plan {plan.get('episode')} {side}: {len(t)} frames, {t[-1]:.1f} s; IK error median "
          f"{ck.get('ik_pos_err_mm', {}).get('median')} mm, p95 {ck.get('ik_pos_err_mm', {}).get('p95')} mm; "
          f"clamped: midline {ck.get('midline_clamped')}, floor {ck.get('floor_clamped')}", flush=True)
    p95 = ck.get("ik_pos_err_mm", {}).get("p95")
    if p95 is not None and p95 > 40.0:
        print(f"NOTE: in the worst 5% of the demo the arm stays {p95} mm from your hand: the planner used the nearest "
              f"safe pose there (safe_pose_used {ck.get('safe_pose_used', 0)}, closest_possible "
              f"{ck.get('closest_possible', 0)} frames). Running it anyway.", flush=True)
    if p95 is not None and a.max_ik_p95_mm is not None and p95 > a.max_ik_p95_mm:
        raise SystemExit(f"REFUSED: the plan's IK misses its targets by {p95} mm (p95 > {a.max_ik_p95_mm} mm), so the "
                         "arm would not follow the demo. Fix the plan (workspace / calibration) first. Nothing moved.")

    # ---- speed: slow the plan until every arm joint stays under the cap
    dt = np.diff(t)
    vel = np.abs(np.diff(q[:, :NJ], axis=0)) / np.maximum(dt, 1e-6)[:, None]
    peak = float(np.percentile(vel.max(axis=1), 99.5)) if len(vel) else 0.0
    need = peak / a.max_deg_s
    scale = a.time_scale if a.time_scale is not None else max(1.0, float(plan.get("recommended_time_scale", 1.0)), need)
    if scale < need:
        raise SystemExit(f"--time-scale {scale} would exceed {a.max_deg_s} deg/s (needs >= {need:.2f}). Nothing moved.")
    cfg = RT.load_config()
    hz = 1.0 / cfg["control"]["nominal_tick_s"]
    tt, qq = resample(t, q, scale, hz)
    qq[:, 5] = np.clip(qq[:, 5], 0.0, 100.0)
    print(f"replay: x{scale:.2f} slower -> {tt[-1]:.1f} s at {hz:.0f} Hz, peak joint speed "
          f"{peak / scale:.0f} deg/s (cap {a.max_deg_s:.0f})", flush=True)

    # ---- arms (both opened: the other arm is read for the collision check, never powered)
    rig = json.loads((HERE / "rig.json").read_text())
    off = rig.get("joint_offsets_deg", {})
    arms = {}
    try:
        for s in SIDES:
            arms[s] = R.Arm(s, rig["arms"][s], off.get(s))
            if arms[s].fingerprint != R.FINGERPRINTS[s]:
                raise SystemExit(f"{rig['arms'][s]} is not the {s} arm")
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"cannot open the arms ({e!r}). Is monitor.py or another runner holding the COM ports?")
    arm = arms[side]

    def read_units():
        return {s: arms[s].to_units(arms[s].read_raw()).astype(float) for s in SIDES}

    # ---- safety model + preflight of the whole motion
    import env as E
    E.SqueezeEnv._measure_pads = lambda self: {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    shadow = E.SqueezeEnv()
    shadow.reset(seed=0)
    try:
        guard = SF.Guard(shadow.model, SF.real_limits(arms))
    except SF.SafetyStop as e:
        raise SystemExit(f"SAFETY: {e}")
    lo, hi = guard.limits[side]
    print(f"{side} real joint ranges (deg): " + " ".join(f"{l:.0f}..{h:.0f}" for l, h in zip(lo, hi)), flush=True)

    u0 = read_units()
    other = [s for s in SIDES if s != side][0]
    start = qq[0].copy()
    if a.no_gripper:
        qq[:, 5] = u0[side][5]
        start[5] = u0[side][5]
    raised = u0[side].copy()
    raised[1:4] = REST_UP
    approach = linear(u0[side], raised, 40) + linear(raised, start, 40)
    motion = [("approach", p) for p in approach] + [("replay", p) for p in qq]
    traj = [{side: p[:NJ], other: u0[other][:NJ]} for _, p in motion]
    bad = guard.preflight(traj)
    if bad is not None:
        i, why = bad
        where = motion[i][0] + ("" if motion[i][0] == "approach" else f" t={tt[i - len(approach)]:.2f}s")
        raise SystemExit(f"SAFETY PREFLIGHT FAILED at {where}: {'; '.join(why)}. Nothing moved.")
    hh = R.HandHeight()
    lows = [hh.lowest(np.r_[(p if side == 'left' else u0['left']), (p if side == 'right' else u0['right'])])[side]
            for _, p in motion]
    print(f"safety preflight OK: {len(motion)} poses within {side}'s real joint ranges and clearances "
          f"(wrist camera mounts >= {SF.MARGIN['cmd']['camera-mount'] * 1000:.0f} mm, camera post >= "
          f"{SF.MARGIN['cmd']['arm-camera'] * 1000:.0f} mm, other arm >= {SF.MARGIN['cmd']['arm-arm'] * 1000:.1f} mm); "
          f"lowest hand point {min(lows) * 100:.1f} cm (floor guard {a.floor_cm} cm)", flush=True)
    print(f"{side} now (deg): {np.round(u0[side], 1).tolist()}  ->  plan start {np.round(start, 1).tolist()}", flush=True)

    run = HERE / "runs" / f"umi_replay_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps({"plan": a.plan, "episode": plan.get("episode"), "side": side,
                                               "scale": scale, "hz": hz, "live": a.live, "start_units": u0[side].tolist(),
                                               "args": vars(a)}, indent=1))
    if not a.live:
        print("DRY: nothing moved. To run it, a person at the rig starts real/run.py with --live.", flush=True)
        for x in arms.values():
            x.close()
        return

    print(f"\nLIVE: the {side} arm will raise, move to the plan's start, and replay {tt[-1]:.0f} s of the demo.\n"
          "Keep a hand on the power switch. Ctrl+C (or the STOP file in the rig folder) stops it and folds the arm to rest.",
          flush=True)
    if a.confirm and input(f"type '{side}' to start: ").strip().lower() != side:
        print("not confirmed: nothing moved", flush=True)
        for x in arms.values():
            x.close()
        return

    one = {side: arm}
    log, fold_log, stream, t0_unix = [], [], None, None
    try:
        STOP.unlink(missing_ok=True)
        arm.prepare()
        arm.goal(arm.read_raw().astype(int))
        SF.apply_torque_limits(one)
        arm.torque(True)
        print("torque on, holding", flush=True)
        ramp = lambda tgt, secs: RT.smooth_ramp(one, {side: tgt}, cfg, secs, check_stop=STOP.exists)  # noqa: E731
        ramp(np.round(arm.to_raw(raised)).astype(int), 3.0)
        ramp(np.round(arm.to_raw(start)).astype(int), 3.0)
        time.sleep(0.5)
        print("at the plan's start pose, replaying", flush=True)
        clock = RT.TickClock(cfg)
        stream = RT.ServoStreamer(one, cfg)
        stream.start({side: arm.read_raw()})
        last_ok = np.asarray(arm.to_units(arm.read_raw()), float)   # last command sent (units)
        floor_ticks, last_floor_print = 0, -1.0
        t0 = time.perf_counter()
        t0_unix = time.time()            # same clock as the rig recorder's frames.csv t_unix (real/overlay.py aligns on it)
        meta = json.loads((run / "meta.json").read_text())
        meta.update(t0_unix=t0_unix, replay_s=float(tt[-1]))
        (run / "meta.json").write_text(json.dumps(meta, indent=1))
        while True:
            if STOP.exists():
                raise KeyboardInterrupt("STOP file")
            tick = time.perf_counter()
            clock.tick()
            k = int((tick - t0) * hz)
            if k >= len(qq):
                break
            u = read_units()
            want = {side: qq[k].copy(), other: u[other].copy()}
            cmd = guard.step(u, want)          # raises SafetyStop; returns the lead-limited command
            def low_of(q):
                return hh.lowest(np.r_[(q if side == "left" else cmd["left"]), (q if side == "right" else cmd["right"])])[side]
            target = np.asarray(cmd[side], float)
            if low_of(target) * 100 < a.floor_cm:
                # Floor: don't freeze. Go as far toward this tick's command as the floor allows (bisection on the
                # straight line from the last command sent); the arm slides along the floor height and continues
                # as soon as the plan rises again.
                lo_a, hi_a = 0.0, 1.0
                for _ in range(8):
                    mid = 0.5 * (lo_a + hi_a)
                    if low_of(last_ok + mid * (target - last_ok)) * 100 >= a.floor_cm:
                        lo_a = mid
                    else:
                        hi_a = mid
                target = last_ok + lo_a * (target - last_ok)
                floor_ticks += 1
                if tick - t0 - last_floor_print > 1.0:
                    last_floor_print = tick - t0
                    print(f"  floor: kept the hand at {a.floor_cm} cm and continued ({floor_ticks} ticks so far)", flush=True)
                # the guard's tracking check must compare the arm with what was really sent
                if guard.prev_cmd is not None:
                    guard.prev_cmd = {**guard.prev_cmd, side: target.copy()}
                cmd = {**cmd, side: target}
            stream.set(side, arm.to_raw(target), clock.period())
            last_ok = target.copy()
            if stream.errors:
                raise RuntimeError(f"servo write failed: {stream.errors[-1]}")
            log.append({"t": round(tick - t0, 3), "k": k, "units": u[side].round(2).tolist(),
                        "cmd": np.asarray(cmd[side]).round(2).tolist()})
            if k % int(hz * 2) == 0:
                err = np.abs(np.asarray(cmd[side])[:NJ] - u[side][:NJ]).max()
                print(f"  t={k / hz:5.1f}s / {tt[-1]:.0f}s  max joint lag {err:4.1f} deg", flush=True)
            clock.sleep_rest(tick)
        time.sleep(0.3)
        print("replay finished", flush=True)
    except SF.SafetyStop as e:
        print(f"SAFETY STOP: {e}", flush=True)
    except BaseException as e:
        print(f"stopping: {e!r}", flush=True)
    finally:
        if stream is not None:
            stream.stop()                      # MUST stop before anything else writes goals
        (run / "log.json").write_text(json.dumps({"t0_unix": t0_unix, "scale": scale, "hz": hz,
                                                  "side": side, "episode": plan.get("episode"), "ticks": log}))
        print(f"log: {run} ({len(log)} ticks)", flush=True)
        STOP.unlink(missing_ok=True)
        print("returning to rest (raise, then fold)", flush=True)
        try:
            uu = arm.to_units(arm.read_raw())
            RT.smooth_ramp(one, {side: arm.to_raw(np.r_[uu[0], REST_UP, uu[4], uu[5]])}, cfg, 3.0, log=fold_log)
            rest = arm.neutral_raw()
            mid = arm.read_raw().copy()
            mid[0], mid[3], mid[4], mid[5] = rest[0], rest[3], rest[4], rest[5]
            RT.smooth_ramp(one, {side: np.round(mid).astype(int)}, cfg, 2.5, log=fold_log)
            RT.smooth_ramp(one, {side: rest}, cfg, 3.0, log=fold_log)
            time.sleep(0.4)
            (run / "fold.json").write_text(json.dumps(fold_log))
        except Exception as e:
            print(f"return to rest failed: {e!r}", flush=True)
        finally:
            arm.torque(False)
            print("rest, torque OFF", flush=True)
            for x in arms.values():
                try:
                    x.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
