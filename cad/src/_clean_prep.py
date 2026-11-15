"""Export every SO-UMI handheld part in the 'as held' frame (x forward, y left, z up), metres."""
import sys, os, json, numpy as np, trimesh
CAD = sys.argv[1]; OUT = sys.argv[2]; ANGLE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.30
sys.path.insert(0, os.path.join(CAD, "src"))
from asm import parts, T
P, _, Mj = parts(0.0)
def mm(m): m = m.copy(); m.apply_scale(1000); return m
def R(a): return trimesh.transformations.rotation_matrix(a, [0, -1, 0], [20.2, 0, -23.4])
Mj = Mj.copy(); Mj[:3, 3] *= 1000
JT = trimesh.load(f"{CAD}/soumi_moving_jaw_loop.stl"); JT.apply_transform(Mj @ T([0, 0, 18.9], [1, 0, 0, 0])); JT.apply_transform(R(ANGLE))
items = {"handle": trimesh.load(f"{CAD}/soumi_handle.stl"), "fixed_jaw": trimesh.load(f"{CAD}/soumi_fixed_jaw_loop.stl"),
         "moving_jaw": JT, "mount": mm(P["mount"]), "camera": mm(P["camera"]), "servo": mm(P["servo"])}
HOLD = np.eye(4); HOLD[:3, :3] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
lo, hi = [], []
for k, m in items.items():
    m = m.copy(); m.apply_transform(HOLD); m.apply_scale(0.001); m.export(f"{OUT}/{k}.stl")
    lo.append(m.bounds[0]); hi.append(m.bounds[1]); print(k, len(m.faces))
json.dump({"lo": np.min(lo, 0).tolist(), "hi": np.max(hi, 0).tolist()}, open(f"{OUT}/bounds.json", "w")); print(np.min(lo, 0), np.max(hi, 0))
