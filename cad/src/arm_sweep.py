"""Do the loop parts hit the SO-101 arm itself anywhere in the joint ranges? Compare against the stock gripper."""
import sys, os, io, runpy, contextlib, itertools, json, tempfile, numpy as np, mujoco, manifold3d as mf, trimesh
SRC = os.path.dirname(os.path.abspath(__file__))
os.chdir(SRC); sys.path.insert(0, SRC)
from asm import OFFICIAL
TMP = tempfile.mkdtemp(prefix="soumi_armprobe_")   # build.py writes its STLs here, not over the real ones
sys.argv = ["build.py", TMP]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("build.py")
m = mujoco.MjModel.from_xml_path(str(OFFICIAL / "so101_new_calib.xml"))
d = mujoco.MjData(m)
bid = {m.body(i).name: i for i in range(m.nbody)}
arm_bodies = ["base", "shoulder", "upper_arm", "lower_arm", "wrist"]
geoms = []
for gi in range(m.ngeom):
    if m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[gi] == 0: continue
    b = m.body(m.geom_bodyid[gi]).name
    if b not in arm_bodies: continue
    mi = m.geom_dataid[gi]; va, vn = m.mesh_vertadr[mi], m.mesh_vertnum[mi]; fa, fn = m.mesh_faceadr[mi], m.mesh_facenum[mi]
    V = m.mesh_vert[va:va + vn].astype(np.float64); F = m.mesh_face[fa:fa + fn]
    tm = trimesh.Trimesh(V, F, process=True)
    try:
        M = mf.Manifold(mf.Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32)))
        if M.is_empty(): M = None
    except Exception: M = None
    if M is None or M.volume() <= 0:   # fall back to convex hull if the mesh isn't a clean solid
        h = tm.convex_hull; M = mf.Manifold(mf.Mesh(vert_properties=np.asarray(h.vertices, np.float32), tri_verts=np.asarray(h.faces, np.uint32)))
    geoms.append(dict(gi=gi, body=b, mesh=m.mesh(mi).name, M=M, lo=tm.bounds[0], hi=tm.bounds[1]))
print("arm collision meshes:", [(x["body"], x["mesh"]) for x in geoms])
parts = {"fixed loop": g["fixed_loop_part"], "moving loop": g["jaw_loop_part"],
         "stock gripper body": g["fol0"], "stock moving jaw": g["jaw0"]}
pb = {k: (np.array(v.bounding_box().min()) if hasattr(v.bounding_box(), 'min') else None) for k, v in parts.items()}
def bbox(Mn):
    t = g["to_tm"](Mn); return t.bounds
pbox = {k: bbox(v) for k, v in parts.items()}
S = np.diag([1e-3, 1e-3, 1e-3, 1.0])
def pose(q):
    for name, val in q.items(): d.qpos[m.joint(name).qposadr[0]] = val
    mujoco.mj_kinematics(m, d)
def Tbody(b): T = np.eye(4); T[:3, :3] = d.xmat[bid[b]].reshape(3, 3); T[:3, 3] = d.xpos[bid[b]]; return T
def Tgeom(gi): T = np.eye(4); T[:3, :3] = d.geom_xmat[gi].reshape(3, 3); T[:3, 3] = d.geom_xpos[gi]; return T
def hits(q):
    pose(q); G = Tbody("gripper"); out = {}
    for k, P in parts.items():
        Mpart = G @ S @ (g["rot_about_pivot"](q["gripper"]) if "moving" in k else np.eye(4))
        lo, hi = pbox[k]; corners = np.array(list(itertools.product(*zip(lo, hi))))
        for x in geoms:
            A = np.linalg.inv(Tgeom(x["gi"])) @ Mpart
            c = corners @ A[:3, :3].T + A[:3, 3]
            if (c.max(0) < x["lo"]).any() or (c.min(0) > x["hi"]).any(): continue
            v = (P.transform(g["mat"](A)) ^ x["M"]).volume() * 1e9   # m^3 -> mm^3
            if v > 0.5: out.setdefault(k, []).append((x["body"], x["mesh"], round(v, 1)))
    return out
grid = dict(shoulder_lift=[-1.2, 0.0, 1.2], elbow_flex=[-1.69, -0.85, 0.0, 0.85, 1.69],
            wrist_flex=list(np.round(np.linspace(-1.658, 1.658, 13), 3)), wrist_roll=list(np.round(np.linspace(-2.74, 2.84, 13), 3)),
            gripper=[-0.17, 0.45, 1.745])
n = 0; loop_only = []; stock_any = 0
for vals in itertools.product(*grid.values()):
    q = dict(zip(grid.keys(), vals)); q["shoulder_pan"] = 0.0
    h = hits(q); n += 1
    stock = any(k.startswith("stock") for k in h)
    stock_any += stock
    loops = {k: v for k, v in h.items() if "loop" in k}
    if loops and not stock: loop_only.append((q, loops))
print(f"poses checked: {n}; stock gripper already hits the arm in {stock_any}; loops hit while stock doesn't: {len(loop_only)}")
for q, l in loop_only[:25]:
    print({k: round(float(v), 2) for k, v in q.items()}, l)
json.dump([dict(q={k: float(v) for k, v in q.items()}, hits=l) for q, l in loop_only], open(os.path.join(TMP, "arm_sweep.json"), "w"))
print("details:", os.path.join(TMP, "arm_sweep.json"))
