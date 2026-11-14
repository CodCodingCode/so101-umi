import trimesh, numpy as np
from trimesh.transformations import quaternion_matrix
import os
from pathlib import Path
# Official SO-101 meshes (TheRobotStudio/SO-ARM100, Apache-2.0). Default: the sibling robots/sim twin.
# Override with SO101_OFFICIAL_DIR=/path/to/so101_official (the folder holding so101_new_calib.xml and assets/).
OFFICIAL = Path(os.environ.get("SO101_OFFICIAL_DIR", Path(__file__).resolve().parents[3] / "sim/so101_twin/assets/so101_official"))
A = str(OFFICIAL / "assets") + "/"
def T(pos,q): M=quaternion_matrix(q); M[:3,3]=pos; return M
def load(n,s=1.0):
    m=trimesh.load(A+n,force='mesh'); m.apply_scale(s); return m
# all in gripper-body frame, meters
def parts(gripper_angle=0.0):
    fol=load("wrist_roll_follower_so101_v1.stl"); fol.apply_transform(T([0,-0.000218214,0.000949706],[0,1,0,0]))
    srv=load("sts3215_03a_v1.stl"); srv.apply_transform(T([0.0077,0.0001,-0.0234],[0.707107,-0.707107,0,0]))
    Mj=T([0.0202,0.0188,-0.0234],[0.707107,0.707107,0,0])
    R=trimesh.transformations.rotation_matrix(gripper_angle,[0,0,1])
    jaw=load("moving_jaw_so101_v1.stl"); jaw.apply_transform(Mj@R@T([0,0,0.0189],[1,0,0,0]))
    Mm=T([-0.0150,0.0240,-0.0315],[0.4997981226682881,-0.5001999599912927,0.4998017973423795,-0.5001999599912927])
    mnt=load("wrist_camera_mount_so101_v1.stl",0.001); mnt.apply_transform(Mm)
    cam=load("wrist_camera_so101_v1.stl"); cam.apply_transform(Mm@T([-0.04941,-0.03813,0.0175],[0.37978693649389456,-0.5964584309361936,-0.37978833153177,-0.596456240026065]))
    tip=np.array([-0.0079,-0.000218121,-0.0981274])
    return dict(follower=fol,servo=srv,jaw=jaw,mount=mnt,camera=cam), tip, Mj
