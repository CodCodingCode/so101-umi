"""One-colour 'clay' render of the SO-UMI handheld with MuJoCo's offscreen renderer."""
import sys, json, numpy as np, mujoco
from PIL import Image
PARTS = sys.argv[1]; OUT = sys.argv[2]
W, H = int(sys.argv[3]), int(sys.argv[4]); AZ, EL, DIST = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
COL = "0.80 0.80 0.81 1"
names = ["handle", "fixed_jaw", "moving_jaw", "mount", "camera", "servo"]
import os
TAGS = [n for n in ("tags_white", "tags_black") if os.path.exists(f"{PARTS}/{n}.stl")]
xml = f"""<mujoco>
  <compiler meshdir="{PARTS}"/>
  <visual>
    <global offwidth="{W}" offheight="{H}" fovy="25"/>
    <quality offsamples="16" shadowsize="8192"/>
    <headlight ambient="0.27 0.27 0.27" diffuse="0.25 0.25 0.25" specular="0.05 0.05 0.05"/>
    <map shadowclip="0.6" znear="0.01"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>
    <material name="clay" rgba="{COL}" specular="0.25" shininess="0.35" reflectance="0"/>
    <material name="paper" rgba="0.98 0.98 0.98 1" specular="0.1" shininess="0.1"/>
    <material name="ink" rgba="0.04 0.04 0.04 1" specular="0.1" shininess="0.1"/>
    {''.join(f'<mesh name="{n}" file="{n}.stl"/>' for n in names + TAGS)}
  </asset>
  <worldbody>
    <light directional="true" pos="0.4 0.5 1.0" dir="-0.4 -0.5 -1.0" diffuse="0.68 0.68 0.68" specular="0.2 0.2 0.2" castshadow="true"/>
    <light directional="true" pos="-0.6 -0.3 0.3" dir="0.6 0.3 -0.3" diffuse="0.22 0.22 0.22" specular="0 0 0" castshadow="false"/>
    {''.join(f'<geom type="mesh" mesh="{n}" material="clay" contype="0" conaffinity="0"/>' for n in names)}
    {''.join(f'<geom type="mesh" mesh="{n}" material="{"paper" if "white" in n else "ink"}" contype="0" conaffinity="0" mass="0.001"/>' for n in TAGS)}
  </worldbody>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
b = json.load(open(f"{PARTS}/bounds.json")); c = (np.array(b["lo"]) + np.array(b["hi"])) / 2
cam = mujoco.MjvCamera(); cam.lookat[:] = c; cam.azimuth, cam.elevation, cam.distance = AZ, EL, DIST
r = mujoco.Renderer(m, H, W); r.update_scene(d, cam); img = r.render()
Image.fromarray(img).save(OUT); print("saved", OUT, img.shape)
