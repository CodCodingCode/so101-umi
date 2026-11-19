"""Check GripReflex in the twin: the "closes until it touches, then stops squeezing" failure, without and with it.

    python -m grip.sim_check            # from the so101-umi folder, ~2 min

The left arm lowers around the cube with the jaws open. The command then closes to the angle where both jaws first
touch the cube (what a UMI handheld records), give or take a calibration error, holds, lifts 6 cm, holds, and opens.
At 25 Hz (the reference rig's control rate), with the gripper position rounded to STS3215 encoder steps. Varied: cube width, jaw friction,
cube mass, contact softness, calibration error. Also: free-air closes (the reflex must not grab air), and the hold force
against the gripper torque limit.

Sim caveats: the gripper servo model was fitted on free moves, not squeezes, and MuJoCo's soft contacts are not real
squishy objects. This checks the reflex's logic and the direction of the effect, not real-rig forces.
"""
import itertools
import time

import mujoco
import numpy as np

from grip.reflex import GripReflex
from sim import RobotEnv

HZ = 25
ENC = np.deg2rad(360 / 4096)
WIDTHS_MM = [22, 28, 32, 38]
SOFT_TC = [0.004, 0.01, 0.02]
FRICTION = [0.25, 0.4, 0.6]
MASS_G = [15, 50, 100]
CAL_ERR = [-0.5, 0.0, 0.5, 1.5]            # demo angle minus first-touch angle, gripper units (+ = looser)

env = RobotEnv(task="pick_place", control_mode="ee_delta", control_hz=HZ)
m, d = env.model, env.data
arm = env.robot["left"]
cube, cube_body = m.geom("cube_geom").id, m.body("cube").id
jaws = {m.body("left_gripper").id: 0, m.body("left_moving_jaw_so101_v1").id: 1}
qadr = m.joint("left_gripper").qposadr[0]
gact = arm.gripper_actuator
LO, HI = m.actuator_ctrlrange[gact]
CAP0 = float(m.actuator_forcerange[gact, 1])
MASS0, INERTIA0 = float(m.body_mass[cube_body]), m.body_inertia[cube_body].copy()
m.geom_priority[cube] = 5                  # the cube's friction / softness govern its contacts
CTL = ("_ee_targets", "_orientations", "_joint_targets")
f6 = np.zeros(6)
_scratch = mujoco.MjData(m)


def meas_units():
    q = np.round(d.qpos[qadr] / ENC) * ENC
    return float((q - LO) / (HI - LO) * 100)


def _step(action):
    """env.step without the pick/place episode logic: this check only uses the env as the twin's physics + IK."""
    env._done, env._steps = False, 0
    env.step(action)


def jaw_forces():
    f = np.zeros(2)
    for i in range(d.ncon):
        c = d.contact[i]
        if cube in (c.geom1, c.geom2):
            j = jaws.get(int(m.geom_bodyid[c.geom2 if c.geom1 == cube else c.geom1]))
            if j is not None:
                mujoco.mj_contactForce(m, d, i, f6)
                f[j] += f6[0]
    return f


def set_cube(width_mm, tc=0.004, mu=0.4, mass_g=None, cap=CAP0):
    h = width_mm / 2000
    m.geom_size[cube] = [h, h, h]
    m.geom_rbound[cube] = h * np.sqrt(3)
    m.geom_aabb[cube] = [0, 0, 0, h, h, h]
    # The body's bounding box too (one node: the cube geom at the body origin). MuJoCo 3.15 culls contacts with it, so
    # a cube grown past its compiled 32 mm only met the table 3 mm deep and was flung off.
    m.bvh_aabb[m.body_bvhadr[cube_body]] = [0, 0, 0, h, h, h]
    m.geom_solref[cube] = [tc, 1.0]
    m.geom_friction[cube] = [mu, 5e-3, 5e-4]
    mass = MASS0 if mass_g is None else mass_g / 1000
    m.body_mass[cube_body] = mass
    m.body_inertia[cube_body] = INERTIA0 * mass / MASS0 * (h / 0.016) ** 2
    m.actuator_forcerange[gact] = [-cap, cap]
    # Mass changes need the derived constants (body_invweight0...) recomputed, or the solver blows up (a 50 g cube
    # flew off at 800 m/s). mj_setConst overwrites the data it is given, so use a scratch one.
    mujoco.mj_setConst(m, _scratch)


def save():
    st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
    return st, {k: np.copy(getattr(env, k)) for k in CTL}


def restore(saved):
    st, ctl = saved
    mujoco.mj_setState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m, d)
    for k, v in ctl.items():
        setattr(env, k, v.copy())
    env._steps, env._success_steps, env._done = 0, 0, False


def pregrasp(width_mm, cube_away=False):
    """Jaws open around the cube (or, cube_away, the same pose with the cube moved off): saved state, hand point."""
    env.reset(seed=0, options={"randomize": False})
    set_cube(width_mm)
    h = width_mm / 2000
    d.joint("cube_joint").qpos[2] = h + 0.0005
    mujoco.mj_forward(m, d)
    c = d.body("cube").xpos.copy()
    if cube_away:
        d.joint("cube_joint").qpos[:2] += [0.0, -0.12]
        mujoco.mj_forward(m, d)
    hold = np.r_[c[:2], max(0.012, h + 0.004)]
    for target, n in ((np.r_[c[:2], 0.075], 60), (hold, 60)):
        for _ in range(n):
            _step(env.action_toward(target, opening=1.0))
            if np.linalg.norm(arm.position - target) < 0.004:
                break
        for _ in range(6):
            _step(env.action_toward(target, opening=1.0))
    return save(), arm.position.copy()


def first_touch(saved, hold, width_mm, tc):
    """Gripper units at which both jaws first touch the cube on a slow close (what the handheld would record)."""
    restore(saved)
    set_cube(width_mm, tc)
    cmd = 60.0
    while cmd > 0:
        cmd -= 0.4
        _step(env.action_toward(hold, opening=cmd / 100))
        if (jaw_forces() > 0.05).all():
            return meas_units()
    raise RuntimeError(f"{width_mm} mm cube: the jaws never both touched it")


def run(saved, hold, script, reflex=None):
    """script: list of (n_ticks, cmd_from, cmd_to, dz). Returns per-tick rows: cmd, sent, meas, F_fixed, F_moving, cube_z."""
    restore(saved)
    if reflex:
        reflex.reset()
    rows, sent = [], 100.0
    target = hold.copy()
    for n, c0, c1, dz in script:
        for k in range(n):
            cmd = c0 + (c1 - c0) * min(1.0, (k + 1) / max(1, n // 2))
            meas = meas_units()
            sent = reflex.step(cmd, meas) if reflex else cmd
            target = hold + [0, 0, dz]
            _step(env.action_toward(target, opening=float(np.clip(sent, 0, 100)) / 100))
            rows.append([cmd, sent, meas_units(), *jaw_forces(), d.body("cube").xpos[2]])
    return np.array(rows)


def grasp_script(demo, hold=15):
    #       ticks  from   to    lift (m)
    return [(12, 100.0, demo, 0.0),     # close to the demo angle (0.5 s)
            (hold, demo, demo, 0.0),    # hold (0.6 s): the lift starts soon after the jaw arrives
            (30, demo, demo, 0.06),     # lift 6 cm
            (25, demo, demo, 0.06),     # hold up (1 s)
            (40, demo, 100.0, 0.06)]    # open


def held_up(r, hold=15):
    """Rows of the 1 s hold after the lift."""
    a = 12 + hold + 30
    return r[a:a + 25]


def main():
    t0 = time.time()
    results = []
    for w in WIDTHS_MM:
        saved, hold = pregrasp(w)
        for tc in SOFT_TC:
            touch = first_touch(saved, hold, w, tc)
            for mu, mass, err in itertools.product(FRICTION, MASS_G, CAL_ERR):
                demo = touch + err
                for use in (False, True):
                    set_cube(w, tc, mu, mass)
                    r = run(saved, hold, grasp_script(demo), GripReflex() if use else None)
                    up = held_up(r)
                    results.append(dict(w=w, tc=tc, mu=mu, mass=mass, err=err, reflex=use,
                                        lifted=bool(up[-1, 5] - r[0, 5] > 0.04),
                                        force=float(up[:, 3:5].mean()),
                                        released=bool(r[-1, 1] == r[-1, 0] and r[-1, 2] > 90)))
        print(f"  {w} mm done ({time.time() - t0:.0f} s)", flush=True)

    R = results
    print("\nGrasp to the jaw angle where both jaws touched, then lift 6 cm (25 Hz, encoder-rounded):")
    print(f"  {'':<26}{'reflex off':>12}{'reflex on':>12}")
    def rate(sel, use):
        x = [r["lifted"] for r in R if sel(r) and r["reflex"] == use]
        return f"{np.mean(x):>11.0%} "
    rows = [("all", lambda r: True)]
    rows += [(f"cal err {e:+.1f} units", (lambda e: lambda r: r["err"] == e)(e)) for e in CAL_ERR]
    rows += [(f"cube {g} g", (lambda g: lambda r: r["mass"] == g)(g)) for g in MASS_G]
    rows += [(f"friction {u}", (lambda u: lambda r: r["mu"] == u)(u)) for u in FRICTION]
    rows += [(f"width {w} mm", (lambda w: lambda r: r["w"] == w)(w)) for w in WIDTHS_MM]
    rows += [(f"softness tc {t}", (lambda t: lambda r: r["tc"] == t)(t)) for t in SOFT_TC]
    for name, sel in rows:
        print(f"  {name:<26}{rate(sel, False)}{rate(sel, True)}")
    for use in (False, True):
        f = [r["force"] for r in R if r["reflex"] == use]
        rel = np.mean([r["released"] for r in R if r["reflex"] == use])
        print(f"  reflex {'on ' if use else 'off'}: squeeze while held up median {np.median(f):5.1f} N (p10 {np.percentile(f, 10):4.1f}); "
              f"let go and opened when commanded on {rel:.0%}")
    fails = [r for r in R if r["reflex"] and not r["lifted"]]
    if fails:
        print(f"  reflex on, not lifted ({len(fails)}):", "; ".join(
            f"{r['w']} mm tc {r['tc']} mu {r['mu']} {r['mass']} g err {r['err']:+.1f}" for r in fails[:12]))

    print("\nFree air (cube moved away), command closes part way and holds; the reflex must not grab air:")
    saved, hold = pregrasp(32, cube_away=True)
    set_cube(32)
    for name, script in [("100 -> 40, hold", [(12, 100, 40, 0), (60, 40, 40, 0)]),
                         ("100 -> 40 -> 20 -> 60", [(12, 100, 40, 0), (25, 40, 40, 0), (12, 40, 20, 0), (25, 20, 20, 0), (12, 20, 60, 0), (25, 60, 60, 0)]),
                         ("100 -> 0 (shut)", [(12, 100, 0, 0), (40, 0, 0, 0)])]:
        rf = GripReflex()
        r = run(saved, hold, script, rf)
        dev = r[:, 0] - r[:, 1]
        print(f"  {name:<24} most it closed past the command {dev.max():4.1f} units, ticks pushing {int((dev > 0.5).sum()):>3}, "
              f"final jaw {r[-1, 2]:5.1f} vs command {r[-1, 0]:5.1f}")

    print("\nSqueeze force vs gripper torque limit (reflex on, 32 mm cube, 50 g, friction 0.4, 2 s before the lift:")
    print("  a lower limit also closes the jaw slower):")
    saved, hold = pregrasp(32)
    touch = first_touch(saved, hold, 32, 0.004)
    for frac in (0.25, 0.5, 0.75, 1.0):
        set_cube(32, 0.004, 0.4, 50, cap=CAP0 * frac)
        r = run(saved, hold, grasp_script(touch, hold=50), GripReflex())
        up = held_up(r, hold=50)
        print(f"  torque limit {CAP0 * frac:4.2f} N m ({frac:.0%} of the fitted cap): held-up squeeze {up[:, 3:5].mean():5.1f} N, "
              f"lifted {up[-1, 5] - r[0, 5] > 0.04}")


if __name__ == "__main__":
    main()
