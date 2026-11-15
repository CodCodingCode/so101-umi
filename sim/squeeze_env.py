"""Two-sided pinch: both SO-101 grippers squeeze the red cube from opposite sides and drop it into the grey box.

Target prop (2026-11-10, user): the real open grey box, 10 x 10 cm outside, 3 cm tall. It replaced the earlier can;
the code keeps the can_* / lid_* names for it: can_pos() = box centre, lid_z() = rim height, on_can() = cube
resting inside the box, can_tipped() = box tipped.

The cube sits between the arms. Each gripper's jaws stay shut and the side of the gripper acts as a
pad: the left pad presses the cube's -x face, the right pad its +x face. The arms lift
it together, carry it over the can and lower it onto the lid, then back off sideways.

Why a fixed yaw: pi_sim's `ee_delta` IK leaves yaw free, and in the middle of the table that free
yaw wandered by up to 90 degrees between poses (measured), moving the gripper's inner surface by
1.5-3 cm. Here each arm holds one yaw (closing axis along -x for the left arm, +x for the right),
so the same part of the gripper always faces the cube, at a measured offset from the tool point.

Actions are pi_sim's `ee_delta` (per arm dx, dy, dz in [-1, 1] at 0.4 m/s, gripper opening 0..1),
the same interface the real rig is driven through (hand poses -> pi_sim IK).
"""

from __future__ import annotations

import sys
from pathlib import Path

import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# The real red cube size (PI_SIM_CUBE_SIZE=0.032) is set in sim/__init__.py, since scene.py reads it at import.

from .env import RobotEnv  # noqa: E402
from .scene import ARMS, CUBE_HALF, build_model  # noqa: E402

# ---------------------------------------------------------------- props
# Grey open box (the real one on the bench): outer 10 x 10 cm, 3 cm tall, 4 mm walls, 3 mm floor.
BOX_OUTER, BOX_WALL, BOX_FLOOR = 0.10, 0.004, 0.003
BOX_INNER_HALF = BOX_OUTER / 2 - BOX_WALL
CAN_HEIGHT = 0.03       # box height (rim above the table); name kept for the code that places the prop
CAN_RADIUS = BOX_INNER_HALF   # "over the target" radius used by the planner / RL shaping
CAN_MASS = 0.08
CAN_TIP_COS = np.cos(np.radians(25))

# Spawn regions (x range, y range). Both lie on the centre line, where both arms reach.
CUBE_REGION = ((-0.02, 0.02), (0.00, 0.05))
CAN_REGION = ((-0.02, 0.02), (-0.10, -0.06))
CUBE_YAW_DEG = 10.0     # spawn yaw jitter; the squeeze squares the cube up
# The real cube is 3D printed and filled with hot glue. User, 2026-11-10 (later estimate, replaces 40-60 g): 10-20 g.
# Drawn per episode so nothing learned here depends on one exact mass.
CUBE_MASS_RANGE = (0.010, 0.020)

# Yaw of each arm's jaw-closing axis (radians). Swept all four quarter turns plus diagonals: with the
# closing axis along +y on both arms, the shut moving jaw's flat side meets the cube with a nearly
# horizontal contact normal (|n_z| <= 0.1). Facing the fixed jaw outward instead touched the cube's
# top edge on a slanted hull (|n_z| = 0.34) and pressed it into the table; right arm at -pi/2 is out of
# wrist-roll range (8 mm IK error).
PAD_YAW = {"left": np.pi, "right": 0.0}
# Tool height above the cube's bottom while holding it. The fixed jaw is a wedge (outer face 31 mm in
# from the tool point at the tip, 47 mm at 5 cm up; same in the visual mesh, so the real jaw too). The
# cube's top edge meets each jaw first, and the two jaw tops clear each other only if that contact is
# high enough on the wedge: at 17 mm the 3 cm cube left 1.3 mm per side and the jaws locked on each
# other (15 N); at 7 mm the fingertips are 3 mm off the table and the tops clear by ~6 mm.
GRASP_Z = 0.007

SETTLE_S = 0.4
# 0 = plain MuJoCo friction (user, 2026-11-10). With the real 10-20 g cube the scripted squeeze still lands
# 30/30 into the box without the noslip pass; 10 was needed for the old 40-60 g cube on the can.
NOSLIP_ITERATIONS = 0
# Each real servo's calibrated range (sim degrees, offsets included). The sim joints / actuators / IK are limited to it
# (3 deg inside) so nothing the sim plans can ask a real joint for an angle it cannot reach (2026-11-11: the left wrist
# roll was asked for 163 deg, the real servo stops at 107, and a wrist camera holder got knocked off).
REAL_RANGES = Path(__file__).resolve().parent / "assets" / "so101_real_ranges.json"
REAL_RANGE_MARGIN_DEG = 3.0


def apply_real_ranges(m):
    if os.environ.get("PI_SIM_REAL_RANGES") == "off" or not REAL_RANGES.exists():
        return
    import json
    spec = json.loads(REAL_RANGES.read_text())
    for side, rows in spec["ranges_deg"].items():
        for name, (lo, hi) in zip(spec["joints"], rows):
            j, a = m.joint(f"{side}_{name}").id, m.actuator(f"{side}_{name}").id
            cur = np.rad2deg(m.jnt_range[j])
            new = np.radians([max(cur[0], lo + REAL_RANGE_MARGIN_DEG), min(cur[1], hi - REAL_RANGE_MARGIN_DEG)])
            m.jnt_range[j] = new
            m.actuator_ctrlrange[a] = new


def pad_quaternion(side):
    """Tool orientation: fingers straight down, closing axis at PAD_YAW[side]."""
    psi = PAD_YAW[side]
    close = np.array([np.cos(psi), np.sin(psi), 0.0])
    approach = np.array([0.0, 0.0, -1.0])
    R = np.column_stack([approach, np.cross(close, approach), close])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


# ---------------------------------------------------------------- appearance (matched to the real rig)
# Colours measured from the real overhead camera (outputs/view_matched_sim.py -> matched_sim.mjb), user 2026-11-11.
# Appearance only: no physics changes (the room walls do not collide).
ARM_RGB = np.array([0.557, 0.6, 0.651])      # printed arm plastic as the cameras see it (grey-blue)
CUBE_RGB = np.array([0.518, 0.188, 0.271])   # the real "red" cube
BOX_RGB = np.array([0.149, 0.135, 0.142])    # the real "grey" box (re-measured 2026-11-12)
# 2026-11-12: re-measured against a real overhead frame (squeeze/match_real_colors.py): the real background (white
# floor beyond the table's curved edge) renders ~1.5x brighter than the sim's, the real wood ~8% darker / yellower
ROOM_RGB = np.array([0.54, 0.54, 0.54])      # walls (what the wrist cameras see); the floor is 0.9x this
TABLE_RGB = np.array([0.339, 0.312, 0.286])  # mean colour of the wood bench top in the overhead camera (re-measured)
OFFTABLE_RGB = np.array([0.72, 0.73, 0.70])  # the white floor seen beyond the real table edge
OFFTABLE = Path(__file__).resolve().parent / "assets" / "offtable_cells.json"
_ARM_PLASTIC = np.array([0.92, 0.92, 0.9])   # pi_sim's near-white arm material, replaced by ARM_RGB


def _rgba(rgb):
    return " ".join(f"{v:.3f}" for v in rgb) + " 1"


def _room_xml():
    """Four grey walls 0.95 m out (visual only)."""
    return "".join(
        f'<geom name="room_wall_{n}" type="box" size="{sx} {sy} 1.2" pos="{px} {py} 0.2" rgba="{_rgba(ROOM_RGB)}"'
        f' contype="0" conaffinity="0"/>'
        for n, sx, sy, px, py in (("n", 1.2, 0.02, 0, 0.95), ("e", 0.02, 1.2, 0.95, 0),
                                  ("w", 0.02, 1.2, -0.95, 0), ("s", 1.2, 0.02, 0, -0.95)))


def _offtable_xml():
    """Visual-only floor-coloured patches where the real table is NOT (beyond its curved edge). Removed by default on
    2026-11-12 at the user's request (they look like white paint on the sim table); PI_SIM_OFFTABLE=on brings them back."""
    if os.environ.get("PI_SIM_OFFTABLE") != "on" or not OFFTABLE.exists():   # OFF by default (user: not real)
        return ""
    import json
    spec = json.loads(OFFTABLE.read_text())
    h = spec["cell_m"] / 2 + 0.0005
    return "".join(f'<geom name="offtable_{i}" type="box" size="{h:.4f} {h:.4f} 0.0004" pos="{x:.4f} {y:.4f} 0.0005" '
                   f'rgba="{_rgba(OFFTABLE_RGB)}" contype="0" conaffinity="0" group="0"/>'
                   for i, (x, y) in enumerate(spec["cells"]))


def match_appearance(m):
    """Grey-blue arms, measured cube colour, wood table at the real brightness, grey floor."""
    for i in range(m.nmat):
        if m.mat(i).name.startswith(("left_", "right_")) and np.allclose(m.mat_rgba[i, :3], _ARM_PLASTIC, atol=0.02):
            m.mat_rgba[i, :3] = ARM_RGB
    m.geom_rgba[m.geom("cube_geom").id, :3] = CUBE_RGB
    table = m.geom("table").id
    if m.geom_matid[table] >= 0:   # pi_sim's wood photo (table_wood.png); scale its mean to the real bench's
        t = m.mat_texid[m.geom_matid[table], 1]
        if t < 0:
            t = m.mat_texid[m.geom_matid[table], 0]
        a, w, h, c = m.tex_adr[t], m.tex_width[t], m.tex_height[t], m.tex_nchannel[t]
        mean = m.tex_data[a:a + w * h * c].reshape(-1, c)[:, :3].mean(0) / 255.0
        m.mat_rgba[m.geom_matid[table], :3] = np.clip(TABLE_RGB / mean, 0, 1)
    else:
        m.geom_rgba[table, :3] = TABLE_RGB
    floor = m.geom("floor").id
    m.geom_matid[floor] = -1
    m.geom_rgba[floor] = [*(ROOM_RGB * 0.9), 1]
    m.stat.center[:] = [0.0, -0.05, 0.12]   # default free camera starts inside the walls
    m.light_castshadow[:] = 0               # the real room light is soft: almost no shadows (randomised in training)
    m.stat.extent = 0.35
    return m


def _can_xml():
    """The grey open box as body "can" (origin at mid-height, like the old can): floor + four walls."""
    h, w, t, f = CAN_HEIGHT / 2, BOX_OUTER / 2, BOX_WALL / 2, BOX_FLOOR / 2
    grey = f'rgba="{_rgba(BOX_RGB)}" friction="0.9 0.01 0.001"'
    walls = "".join(
        f'<geom name="box_wall_{i}" type="box" size="{sx} {sy} {h}" pos="{px} {py} 0" mass="{CAN_MASS / 5}" {grey}/>'
        for i, (sx, sy, px, py) in enumerate([(t, w, -w + t, 0), (t, w, w - t, 0), (w - 2 * t, t, 0, -w + t), (w - 2 * t, t, 0, w - t)]))
    return (f'<body name="can" pos="0 -0.08 {h}"><freejoint name="can_joint"/>'
            f'<geom name="can_geom" type="box" size="{w - 2 * t} {w - 2 * t} {f}" pos="0 0 {-h + f}" mass="{CAN_MASS / 5}" {grey}/>'
            f'{walls}</body>')


class _Sim(RobotEnv):
    """pi_sim with the grey box, a wider near edge, and a fixed tool yaw per arm."""

    def _build_model(self, object_shape, object_color, camera_rig):
        return match_appearance(build_model(object_shape, object_color, camera_rig, extra_worldbody=_can_xml() + _room_xml() + _offtable_xml()))

    def _solve_ik(self, arm, position, quaternion, downward=None):
        return super()._solve_ik(arm, position, self.pad_quat[arm.name], downward=False)


class SqueezeEnv(gym.Env):
    """Bimanual SO-101: squeeze the red cube between both grippers and drop it into the grey box."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, control_hz: int = 25, episode_seconds: float = 20.0, seed: int | None = None):
        super().__init__()
        self.sim = _Sim(task="pick_place", control_mode="ee_delta", control_hz=control_hz,
                        episode_seconds=episode_seconds, max_episode_steps=10 ** 9)
        self.sim.pad_quat = {side: pad_quaternion(side) for side in ARMS}
        # pi_sim clips hand targets to y >= -0.08; the can spawns down to -0.10.
        self.sim._workspace_low[:, 1] = -0.13
        self.sim._workspace_low[:, 2] = GRASP_Z - 0.002   # pi_sim's floor is 16 mm; the grip is lower
        self.model, self.data = self.sim.model, self.sim.data
        # MuJoCo's soft contacts let a squeezed object creep: with 8 N on each pad (static friction
        # ~30x the cube's weight) the held cube still slid 7 mm down the jaws while lifting 4 cm and
        # the squeeze relaxed to 0.8 N. The noslip pass removes that creep; real friction does not creep.
        self.model.opt.noslip_iterations = NOSLIP_ITERATIONS
        apply_real_ranges(self.model)
        self.control_hz = control_hz
        self.max_steps = int(round(episode_seconds * control_hz))
        self.settle_steps = max(1, int(round(SETTLE_S * control_hz)))

        m = self.model
        self._cube_gid = m.geom("cube_geom").id
        self._cube_qadr = m.jnt_qposadr[m.joint("cube_joint").id]
        self._cube_bid = m.body("cube").id
        self.cube_mass = float(m.body_mass[self._cube_bid])
        self._cube_dof = m.jnt_dofadr[m.joint("cube_joint").id]
        self._can_qadr = m.jnt_qposadr[m.joint("can_joint").id]
        self._arm_geoms = {side: {g for g in range(m.ngeom)
                                  if m.body(m.geom_bodyid[g]).name.startswith(side)
                                  and (m.geom_contype[g] or m.geom_conaffinity[g])} for side in ARMS}
        self.action_space = spaces.Box(-1.0, 1.0, (8,), np.float32)
        self._rng = np.random.default_rng(seed)
        self.sim.reset(seed=0, options={"randomize": False})
        self.pad = self._measure_pads()
        self.observation_space = spaces.Box(-np.inf, np.inf, self._observe().shape, np.float32)

    # ------------------------------------------------------------ geometry
    def _measure_pads(self):
        """Where each gripper first touches a cube pushed in from the middle, from the model itself.

        The SO-101 gripper widens above its fingertips, so with the tool near the table the first
        contact is the cube's top edge, not a jaw plate (measured; any box-only estimate is ~7 mm off).
        This sweeps a cube toward each posed, shut gripper and records the first contact.
        Returns {side: (inward, dy)}: the cube face touches `inward` metres past the tool point toward
        the middle, and the contact sits `dy` from the tool point in y.
        """
        m, sim = self.model, self.sim
        d = mujoco.MjData(m)
        out = {}
        for side, x in (("left", -0.05), ("right", 0.05)):
            arm = sim.robot[side]
            tool = np.array([x, 0.0, GRASP_Z])
            q, err = sim._solve_ik(arm, tool, None)
            assert err < 1e-3, (side, err)
            d.qpos[:] = self.data.qpos
            d.qpos[arm.qpos_ids] = q
            d.qpos[m.joint(f"{side}_gripper").qposadr[0]] = m.actuator_ctrlrange[arm.gripper_actuator, 0]
            d.qpos[self._can_qadr:self._can_qadr + 3] = [0.4, 0.3, CAN_HEIGHT / 2]
            sign = 1.0 if side == "left" else -1.0
            for inward in np.arange(0.06, -0.03, -0.0002):
                d.qpos[self._cube_qadr:self._cube_qadr + 7] = [x + sign * (inward + CUBE_HALF), 0, CUBE_HALF + 0.0005, 1, 0, 0, 0]
                mujoco.mj_forward(m, d)
                hits = [c.pos for c in d.contact[:d.ncon]
                        if self._cube_gid in (c.geom1, c.geom2) and ({int(c.geom1), int(c.geom2)} & self._arm_geoms[side])]
                if hits:
                    out[side] = (float(inward), float(np.mean(hits, axis=0)[1]))
                    break
            else:
                raise RuntimeError(f"{side} gripper never touched the probe cube")
        return out

    # ------------------------------------------------------------ state
    def cube_pos(self):
        return self.data.xpos[self.model.body("cube").id].copy()

    def cube_yaw(self):
        R = self.data.xmat[self.model.body("cube").id].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def can_pos(self):
        return self.data.xpos[self.model.body("can").id].copy()

    def _can_up(self):
        return self.data.xmat[self.model.body("can").id].reshape(3, 3)[:, 2].copy()

    def lid_z(self):
        return float(self.can_pos()[2] + CAN_HEIGHT / 2 * self._can_up()[2])

    def tcp(self, side):
        return self.sim.robot[side].position

    def touching(self, side):
        """Is this arm in contact with the cube?"""
        geoms = self._arm_geoms[side]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self._cube_gid and g2 in geoms or g2 == self._cube_gid and g1 in geoms:
                return True
        return False

    def squeezed(self):
        """The two-sided pinch: BOTH arms touch the cube."""
        return self.touching("left") and self.touching("right")

    def on_can(self):
        """Resting inside the box: within the walls (box frame), below the rim, slow, box upright, no arm touching."""
        p, c = self.cube_pos(), self.can_pos()
        R = self.data.xmat[self.model.body("can").id].reshape(3, 3)
        local = R.T @ (p - c)
        inside = bool(np.all(np.abs(local[:2]) < BOX_INNER_HALF - 0.6 * CUBE_HALF))
        below_rim = p[2] - CUBE_HALF < self.lid_z() - 0.01
        slow = np.linalg.norm(self.data.qvel[self._cube_dof:self._cube_dof + 3]) < 0.05
        free = not (self.touching("left") or self.touching("right"))
        return bool(inside and below_rim and slow and free and not self.can_tipped())

    def can_tipped(self):
        return bool(self._can_up()[2] < CAN_TIP_COS)

    def _observe(self):
        parts = []
        cube, can = self.cube_pos(), self.can_pos()
        for side in ARMS:
            arm = self.sim.robot[side]
            parts += [self.data.qpos[arm.qpos_ids], self.data.qvel[arm.dof_ids], [arm.gripper_opening]]
            parts += [self.tcp(side), cube - self.tcp(side), [float(self.touching(side))]]
        parts += [cube, self.data.qpos[self._cube_qadr + 3:self._cube_qadr + 7],
                  self.data.qvel[self._cube_dof:self._cube_dof + 3]]
        parts += [can[:2], [self.lid_z()], self._can_up(), [float(self.squeezed()), float(self.on_can())]]
        return np.concatenate([np.asarray(x, np.float64).ravel() for x in parts]).astype(np.float32)

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.sim.reset(seed=int(self._rng.integers(2 ** 31)), options={"randomize": False})
        (x0, x1), (y0, y1) = CUBE_REGION
        cube_xy = self._rng.uniform([x0, y0], [x1, y1])
        (x0, x1), (y0, y1) = CAN_REGION
        can_xy = self._rng.uniform([x0, y0], [x1, y1])
        yaw = np.radians(self._rng.uniform(-CUBE_YAW_DEG, CUBE_YAW_DEG))
        self.cube_mass = float(self._rng.uniform(*CUBE_MASS_RANGE))
        self.model.body_mass[self._cube_bid] = self.cube_mass
        self.model.body_inertia[self._cube_bid] = self.cube_mass * (2 * CUBE_HALF) ** 2 / 6   # solid cube
        d = self.data
        d.qpos[self._cube_qadr:self._cube_qadr + 7] = [*cube_xy, CUBE_HALF + 0.001, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[self._can_qadr:self._can_qadr + 7] = [*can_xy, CAN_HEIGHT / 2, 1, 0, 0, 0]
        d.mocap_pos[0] = [2.0, 2.0, -1.0]   # park pi_sim's goal disk
        d.qvel[:] = 0
        mujoco.mj_forward(self.model, d)
        for _ in range(50):
            mujoco.mj_step(self.model, d)
        self._step = 0
        self._on_can_steps = 0
        self._ever = {k: False for k in ("squeezed", "lifted", "placed", "success")}
        return self._observe(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1, 1).reshape(2, 4).copy()
        a[:, 3] = np.clip(a[:, 3], 0, 1)
        _, _, sim_term, _, _ = self.sim.step(a.ravel())   # sim_term: cube fell off the table
        self._step += 1
        squeezed = self.squeezed()
        lifted = squeezed and self.cube_pos()[2] - CUBE_HALF > 0.01
        on = self.on_can()
        self._on_can_steps = self._on_can_steps + 1 if on else 0
        self._ever["squeezed"] |= squeezed
        self._ever["lifted"] |= lifted
        self._ever["placed"] |= on
        self._ever["success"] |= self._on_can_steps >= self.settle_steps
        tipped = self.can_tipped()
        terminated = bool(tipped or sim_term)
        truncated = self._step >= self.max_steps
        info = {"is_success": self._ever["success"], **{k: v for k, v in self._ever.items() if k != "success"},
                "squeezing": squeezed, "on_can": on, "can_tipped": tipped}
        reward = float(on) + 10.0 * float(self._on_can_steps == self.settle_steps)
        return self._observe(), reward, terminated, truncated, info

    def render(self, camera="overhead"):
        return self.sim.render_camera(camera)

    def close(self):
        self.sim.close()
