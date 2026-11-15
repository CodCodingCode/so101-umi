"""SO-101 forward kinematics for SO-UMI touch calibration, on the rig's own MuJoCo arm model.

FK only on purpose: IK for demos lives in umi/to_robot.py (one solver, not another copy).

The model is pi_sim's vendored TheRobotStudio so101.xml, the same arm model the rig's runners
(so101_real.HandHeight, dream_kin.py, safety.Guard) use, so joint angles here are the rig's units:
degrees = sim joint angle (so101_real already adds the rig.json offsets), gripper 0 (shut) .. 100 (open).

Frames
  base     the arm's `base` body (origin of so101.xml). Each real arm's base pose relative to table tag 13
           comes from touch calibration (umi/touch_calib.py).
  gripper  the `gripper` body: the same frame umi/convert.py tracks on the handheld (its handle replicates the
           robot's horn plate), so a handheld pose maps 1:1 onto the robot's gripper body.

Override the model path with SO101_XML=/path/to/so101.xml.
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
XML = Path(os.environ.get("SO101_XML", REPO / "sim/assets/vendor/robotstudio_so101/so101.xml"))
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# fixed-jaw tip in the gripper frame, metres (cad/src/asm.py; same point as umi/convert.py JAW_TIP)
JAW_TIP = np.array([-7.9, -0.218, -98.13]) / 1000.0
# the rig's person-confirmed tool point: where the CLOSED fingertips meet, gripper frame, metres
# (C:\robot\real\tcp_confirmed.json, 2026-11-14). Touch calibration touches the tag with this point.
TCP = np.array([-0.0066, -0.0002, -0.0996])
# so101_real.SIM_HOME (one arm): upright start pose every rig runner ramps through
SIM_HOME = np.array([0.0, -45.8, 45.9, 90.0, -2.8, 100.0])


def T_from(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def quat_xyzw_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


class Arm:
    def __init__(self, xml: Path = XML):
        if not Path(xml).exists():
            raise SystemExit(f"SO-101 model not found: {xml}\nSet SO101_XML to the rig's so101.xml.")
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.lim = np.array([m.jnt_range[m.joint(j).id] for j in ARM_JOINTS])     # rad
        self.base = m.body("base").id
        self.grip = m.body("gripper").id
        self.site = m.site("gripperframe").id
        self.grip_q = m.jnt_qposadr[m.joint("gripper").id]
        self.grip_range = m.actuator_ctrlrange[m.actuator("gripper").id].copy()  # rad <-> units 0..100
        mujoco.mj_kinematics(m, self.d)
        Rb, pb = self.d.xmat[self.base].reshape(3, 3), self.d.xpos[self.base]
        if not (np.allclose(Rb, np.eye(3)) and np.allclose(pb, 0)):
            raise SystemExit("so101.xml base body is not at the model origin; this module assumes it is")

    def _T_body(self, b):
        return T_from(self.d.xmat[b].reshape(3, 3).copy(), self.d.xpos[b].copy())

    def _set(self, units5):
        self.d.qpos[self.qadr] = np.deg2rad(np.asarray(units5[:5], float))

    def fk(self, units5):
        """T_base_gripper (4x4) for 5 arm joints in degrees."""
        self._set(units5)
        mujoco.mj_kinematics(self.m, self.d)
        return self._T_body(self.grip)

    def tip(self, units5, point=TCP):
        """A gripper-frame point (default: the closed-fingertip TCP) in the base frame (metres)."""
        T = self.fk(units5)
        return T[:3, :3] @ point + T[:3, 3]

    def grip_units(self, angle_rad):
        lo, hi = self.grip_range
        return float(np.clip((angle_rad - lo) / (hi - lo) * 100.0, 0.0, 100.0))


if __name__ == "__main__":
    a = Arm()
    print(f"model {XML}\nSIM_HOME closed-fingertip TCP in base frame (m):", np.round(a.tip(SIM_HOME), 4))
