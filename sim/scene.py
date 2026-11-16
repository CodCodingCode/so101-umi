"""Compose pinned SO-101 Menagerie models into a colored-object tabletop scene."""

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .cameras import CAMERAS, FIXED_CAMERAS, load_camera_rig

ASSETS = Path(__file__).parent / "assets" / "vendor"
ARMS = ("left", "right")
# Object size: PI_SIM_CUBE_SIZE = full cube edge in metres (default 0.03; the real red cube is 0.032).
CUBE_HALF = float(os.environ.get("PI_SIM_CUBE_SIZE", "0.03")) / 2
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
COLORS = {"red": (0.92, 0.24, 0.14, 1), "green": (0.2, 0.72, 0.3, 1),
          "blue": (0.16, 0.4, 0.92, 1), "yellow": (0.95, 0.8, 0.15, 1)}

# Bimanual layout from TheRobotStudio's overhead camera mount (vendor/robotstudio_overhead_mount):
# each arm bolts onto an arm_base plate, and both plates interlock with the camera post's foot.
# The plates' hole trapezoid (63.5 / 55.6 mm wide, 69.8 mm deep) matches the SO-101 base's, so
# the arm's hole centre lands on the plate's: 116.0 mm either side of the post, 23.28 mm in front of it.
ARM_X = 0.116                    # was 0.18 before the mount (0.36 m between arms; now 0.232 m)
ARM_Y = -0.22                    # arm base-body origin; its hole centre is 21.245 mm forward of this
ARM_Z = 0.0072                   # the arms stand on the 7.2 mm arm_base plates
ARM_SHIFT = 0.18 - ARM_X         # how far each arm moved toward the middle; task layouts follow it
# Measured per-arm pedestal corrections (robots/so101/calibration, adopted with the camera refit): optional file
# assets/so101_kinematics.json {"left": {"pedestal_shift_cm": [x, y, z]}, "right": {...}} or PI_SIM_KINEMATICS=<file>.
# Absent -> no shift (the nominal mount layout). The per-joint offsets of the same fit live on the REAL side
# (rig.json joint_offsets_deg: sim = real + offset), not here.
KINEMATICS = Path(os.environ.get("PI_SIM_KINEMATICS", "")) if os.environ.get("PI_SIM_KINEMATICS") else None


def _pedestal_shift(side):
    import json as _json
    path = KINEMATICS or (Path(__file__).parent / "assets" / "so101_kinematics.json")
    if not path.exists():
        return np.zeros(3)
    v = _json.loads(path.read_text()).get(side, {}).get("pedestal_shift_cm")
    return np.zeros(3) if v is None else np.asarray(v, float)[:3] / 100.0
POST_XY = (0.0, ARM_Y + 0.021245 - 0.02328)
MOUNT_QUAT = "0.5 0.5 0.5 0.5"   # STL frame (x workspace, y up, z across) -> world (x across, y workspace, z up)
MOUNT_PARTS = {  # world position of each STL's origin; derivation in vendor/robotstudio_overhead_mount/UPSTREAM.json
    "cam_mount_bottom": (POST_XY[0] - 0.0365, POST_XY[1] - 0.0187, 0.0),
    "cam_mount_middle": (POST_XY[0], POST_XY[1], 0.18825),
    "cam_mount_top": (POST_XY[0], POST_XY[1], 0.18825),
    "left_arm_base": (POST_XY[0] - 0.06255, POST_XY[1] + 0.0752, 0.005),
    "right_arm_base": (POST_XY[0] + 0.16945, POST_XY[1] + 0.0752, 0.005),
}


def _prefixed(path: Path, prefix: str) -> ET.Element:
    root = ET.parse(path).getroot()
    for mesh in root.findall("asset/mesh"):
        mesh.set("name", mesh.get("name", Path(mesh.attrib["file"]).stem))
        mesh.set("file", str(path.parent / "assets" / mesh.attrib["file"]))
    refs = {"name", "class", "childclass", "mesh", "material", "joint", "joint1",
            "joint2", "body1", "body2", "tendon", "target"}
    for elem in root.iter():
        for key in refs & elem.attrib.keys():
            elem.set(key, prefix + elem.attrib[key])
    return root


def _numbers(values):
    return " ".join(map(str, values))


def _configure_camera(camera, spec):
    # MuJoCo principalpixel is a frustum offset, not OpenCV's top-left cx/cy.
    fx, fy, cx, cy = spec.intrinsics
    camera.attrib.clear()
    camera.attrib.update(mode="fixed", pos=_numbers(spec.pos), quat=_numbers(spec.quat),
                         resolution=f"{spec.width} {spec.height}", sensorsize="1 1",
                         focalpixel=f"{fx} {fy}",
                         principalpixel=f"{spec.width / 2 - cx} {spec.height / 2 - cy}", ipd="0")


def _fixed_camera(world, name, spec):
    mount = ET.SubElement(world, "body", name=f"{name}_camera_mount",
                          pos=_numbers(spec.pos), quat=_numbers(spec.quat))
    camera = ET.SubElement(mount, "camera")
    _configure_camera(camera, spec)
    camera.set("name", name)
    camera.set("pos", "0 0 0")
    camera.set("quat", "1 0 0 0")
    # 32x32 mm U20CAM board proxies, with the optical center in front
    # of the lens. They never obstruct their own feed or alter robot contacts.
    common = {"contype": "0", "conaffinity": "0", "group": "2"}
    ET.SubElement(mount, "geom", name=f"{name}_camera_housing", type="box",
                  size="0.016 0.016 0.0008", pos="0 0 0.017", rgba="0.08 0.3 0.16 1", **common)
    ET.SubElement(mount, "geom", name=f"{name}_camera_lens", type="cylinder",
                  size="0.006 0.007", pos="0 0 0.009", rgba="0.08 0.09 0.11 1", **common)
    x, y, z = spec.pos
    if name == "overhead":
        pass  # carried by the TheRobotStudio post meshes (MOUNT_PARTS), added in build_model
    else:
        ET.SubElement(world, "geom", name=f"{name}_camera_stand", type="capsule", size="0.008",
                      fromto=_numbers([x, y, -0.74, x, y, z + 0.035]), rgba="0.22 0.24 0.27 1", **common)


def _add_mount(root):
    """Overhead post and both arm_base plates as visual meshes (no contacts: the table stays the support)."""
    asset, world = root.find("asset"), root.find("worldbody")
    for stl in ("arm_base", "cam_mount_bottom", "cam_mount_middle", "cam_mount_top"):
        ET.SubElement(asset, "mesh", name=f"mount_{stl}", file=str(ASSETS / "robotstudio_overhead_mount" / f"{stl}.stl"),
                      scale="0.001 0.001 0.001")
    colors = {"cam_mount_bottom": "0.92 0.92 0.9 1", "cam_mount_middle": "0.08 0.08 0.09 1",
              "cam_mount_top": "0.08 0.08 0.09 1", "left_arm_base": "0.92 0.92 0.9 1", "right_arm_base": "0.92 0.92 0.9 1"}
    for part, pos in MOUNT_PARTS.items():
        mesh = "mount_arm_base" if part.endswith("arm_base") else f"mount_{part}"
        ET.SubElement(world, "geom", name=f"mount_{part}", type="mesh", mesh=mesh, pos=_numbers(pos),
                      quat=MOUNT_QUAT, rgba=colors[part], contype="0", conaffinity="0", group="1", mass="0")
    # Solid stand-ins so the arms can HIT the camera post and the overhead camera head (the meshes above are
    # visual only; their convex hulls would also fill the foot plate between the arms). Post: 36.6 x 25.4 mm
    # section from the mount CAD, table to 0.56 m; head: the tilted camera plate + module around the lens.
    # contype 2 / conaffinity 1: they collide with the arms (and anything else dynamic), not with the table.
    px, py = POST_XY
    ET.SubElement(world, "geom", name="mount_post_collision", type="box", size="0.0183 0.0127 0.28",
                  pos=_numbers((px, py, 0.28)), contype="2", conaffinity="1", group="3", rgba="1 0.3 0.3 0.4")
    ET.SubElement(world, "geom", name="mount_head_collision", type="box", size="0.024 0.032 0.022",
                  pos=_numbers((px, py + 0.03, 0.54)), contype="2", conaffinity="1", group="3", rgba="1 0.3 0.3 0.4")


def build_model(object_shape="cube", object_color="red", camera_rig=None, extra_worldbody=None) -> mujoco.MjModel:
    """`extra_worldbody`: optional XML string of bodies appended to the worldbody (task props)."""
    if object_shape not in ("cube", "cylinder") or object_color not in COLORS:
        raise ValueError("Choose cube/cylinder and red/green/blue/yellow")
    cameras = load_camera_rig(camera_rig)
    root = ET.fromstring('''<mujoco model="pi_sim_bimanual_so101">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
      <visual><global offwidth="640" offheight="640"/><quality shadowsize="2048"/>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
      <statistic center="0 0 0.15" extent="0.9"/>
      <default><mesh maxhullvert="128"/></default><asset>
        <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.12 0.15 0.19" rgb2="0.17 0.2 0.24" width="256" height="256"/>
        <material name="floor_mat" texture="floor_tex" texrepeat="3 3"/>
      </asset>
      <worldbody>
        <light name="key_light" pos="0 -0.5 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.76" material="floor_mat"/>
        <geom name="table" type="box" size="0.5 0.4 0.025" pos="0 0 -0.025" rgba="0.55 0.38 0.22 1" friction="0.25 0.005 0.0001" priority="1"/>
        <body name="cube" pos="-0.116 0 0.016">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="0.015 0.015 0.015" mass="0.015" friction="1 0.01 0.001" rgba="0.92 0.24 0.14 1"/>
        </body>
        <body name="goal" mocap="true" pos="-0.12 0.06 0.001">
          <geom name="goal_disk_geom" type="cylinder" size="0.04 0.001" contype="0" conaffinity="0" rgba="0.18 0.65 0.40 0.7"/>
          <site name="reach_target" pos="0 0 0.07" size="0.012" rgba="0.18 0.9 0.4 0.55"/>
        </body>
      </worldbody>
      <contact/><tendon/><equality/><actuator/>
    </mujoco>''')
    obj = root.find(".//geom[@name='cube_geom']")
    obj.set("type", "box" if object_shape == "cube" else "cylinder")
    h = CUBE_HALF
    obj.set("size", f"{h} {h} {h}" if object_shape == "cube" else f"{h} {h}")
    root.find(".//body[@name='cube']").set("pos", f"-0.116 0 {h + 0.001}")
    obj.set("rgba", " ".join(map(str, COLORS[object_color])))
    if extra_worldbody:
        root.find("worldbody").extend(list(ET.fromstring(f"<extra>{extra_worldbody}</extra>")))
    for name in FIXED_CAMERAS:
        _fixed_camera(root.find("worldbody"), name, cameras[name])
    # Both bases point into the tabletop (+Y); one wrist camera per follower.
    _add_mount(root)
    for side, x in (("left", -ARM_X), ("right", ARM_X)):
        arm = _prefixed(ASSETS / "robotstudio_so101/so101.xml", f"{side}_")
        _realistic_gripper(arm, side)
        for mat in arm.findall("asset/material"):          # the real arms are white printed plastic, not yellow
            rgba = [float(v) for v in mat.get("rgba", "1 1 1 1").split()]
            if rgba[0] > 0.7 and rgba[1] > 0.6:
                mat.set("rgba", "0.92 0.92 0.9 1")
        camera = arm.find(f".//camera[@name='{side}_wrist_cam']")
        _configure_camera(camera, cameras[f"{side}_wrist"])
        camera.set("name", f"{side}_wrist")
        dx, dy, dz = _pedestal_shift(side)
        wrapper = ET.SubElement(root.find("worldbody"), "body", name=f"{side}_pedestal",
                                pos=f"{x + dx} {ARM_Y + dy} {ARM_Z + dz}", euler=f"0 0 {np.pi / 2}")
        wrapper.append(arm.find("worldbody/body"))
        for section in ("default", "asset", "contact", "tendon", "equality", "actuator"):
            contents = arm.find(section)
            if contents is not None:
                root.find(section).extend([el for el in contents if section != "default" or el.tag == "default"])
    _apply_dynamics(root)
    _table_texture(root)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


TABLE_TEXTURE = Path(__file__).parent / "assets" / "textures" / "table_wood.png"


def _table_texture(root):
    """The real bench top (photo IMG_8852, light oak, grain along x), tiled ~0.40 x 0.30 m per photo.
    PI_SIM_TABLE=plain keeps the flat brown table."""
    if os.environ.get("PI_SIM_TABLE", "wood") != "wood" or not TABLE_TEXTURE.exists():
        return
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="table_wood", type="2d", file=str(TABLE_TEXTURE))
    ET.SubElement(asset, "material", name="table_wood_mat", texture="table_wood", texuniform="true",
                  texrepeat="2.5 3.33", specular="0.15", shininess="0.2")
    table = root.find(".//geom[@name='table']")
    table.set("material", "table_wood_mat")
    table.attrib.pop("rgba", None)


COACD_DIR = ASSETS / "robotstudio_so101" / "assets" / "coacd"
# Plastic-on-plastic (PLA gripper on the 3D-printed cube) is ~0.3-0.5; the stock gripper classes used 1.0.
PAD_FRICTION = float(os.environ.get("PI_SIM_PAD_FRICTION", "0.4"))


def _realistic_gripper(arm, side):
    """Collide with the real jaw shapes: replace each gripper's convex-hull blob (collision_gripper_mesh, which
    puts contact surfaces where the real jaw has empty space) and the hand-placed boxes / spheres with a CoACD
    convex decomposition of the actual fixed-jaw and moving-jaw meshes (24 pieces each), at the visual meshes' poses.
    PI_SIM_GRIPPER_COLLISION=stock keeps the vendor collision geometry."""
    if os.environ.get("PI_SIM_GRIPPER_COLLISION", "coacd") != "coacd" or not COACD_DIR.exists():
        return
    asset = arm.find("asset")
    drop = {f"{side}_collision_gripper", f"{side}_collision_gripper_mesh", f"{side}_collision"}
    for body_name, mesh_name, pos, quat in ((f"{side}_gripper", "wrist_roll_follower_so101_v1", "0 -0.000218 0.00095", "0 1 0 0"),
                                            (f"{side}_moving_jaw_so101_v1", "moving_jaw_so101_v1", "0 0 0.0189", "1 0 0 0")):
        body = arm.find(f".//body[@name='{body_name}']")
        for g in list(body.findall("geom")):
            if g.get("class", "") in drop:
                body.remove(g)
        for f in sorted(COACD_DIR.glob(f"{mesh_name}_*.stl")):
            mesh = f"{side}_coacd_{f.stem}"
            ET.SubElement(asset, "mesh", name=mesh, file=str(f))
            g = ET.SubElement(body, "geom", type="mesh", mesh=mesh, pos=pos, quat=quat,
                              friction=f"{PAD_FRICTION} 5e-3 5e-4", rgba="1 0 0 1")
            g.set("class", f"{side}_collision_gripper")


DYNAMICS_FILE = Path(__file__).parent / "assets" / "so101_dynamics.json"


def _apply_dynamics(root):
    """Servo dynamics fitted to the real rig (assets/so101_dynamics.json), written into each arm's actuators and
    joints before compiling. PI_SIM_DYNAMICS=nominal keeps the stock Menagerie/LeRobot values; PI_SIM_DYNAMICS=<path>
    loads another file. Per joint: kp, kv, forcerange (N m), timeconst (s, first-order command lag), frictionloss,
    armature; the same values are used for both arms."""
    import json
    choice = os.environ.get("PI_SIM_DYNAMICS", "")
    if choice == "nominal":
        return
    path = Path(choice) if choice else DYNAMICS_FILE
    if not path.exists():
        return
    joints = json.loads(path.read_text())["joints"]
    for side in ARMS:
        for name, p in joints.items():
            act = root.find(f"actuator/position[@name='{side}_{name}']")
            jnt = root.find(f".//joint[@name='{side}_{name}']")
            if act is None or jnt is None:
                raise ValueError(f"{path.name}: no actuator/joint {side}_{name}")
            act.set("kp", f"{p['kp']:.6g}")
            act.set("kv", f"{p['kv']:.6g}")
            act.set("forcerange", f"{-p['forcerange']:.6g} {p['forcerange']:.6g}")
            if p.get("timeconst", 0) > 0:
                act.set("timeconst", f"{p['timeconst']:.6g}")
            jnt.set("frictionloss", f"{p['frictionloss']:.6g}")
            jnt.set("armature", f"{p['armature']:.6g}")
