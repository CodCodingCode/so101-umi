import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import numpy as np, trimesh, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba
from asm import parts, T
OUT = sys.argv[1]
P, _, Mj = parts(0.0)
def mm(m): m = m.copy(); m.apply_scale(1000); return m
H = trimesh.load(f"{OUT}/soumi_handle.stl")
FL = trimesh.load(f"{OUT}/soumi_fixed_jaw_loop.stl")
Mj = Mj.copy(); Mj[:3, 3] *= 1000
JT = trimesh.load(f"{OUT}/soumi_moving_jaw_loop.stl"); JT.apply_transform(Mj @ T([0, 0, 18.9], [1, 0, 0, 0]))
print("bodies:", {n: sum(b.volume > 0 for b in m.split(only_watertight=False)) for n, m in [("handle", H), ("fixed", FL), ("moving", JT)]})
def R(a): return trimesh.transformations.rotation_matrix(a, [0, -1, 0], [20.2, 0, -23.4])
# "as held" frame: forward = -z (jaws), left = -x, up = +y (camera side); proper rotation (det +1)
HOLD = np.eye(4); HOLD[:3, :3] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
def dec(m, n): return m.simplify_quadric_decimation(face_count=n) if len(m.faces) > n else m
fixed = [(dec(FL, 24000), '#f2c31f', 1.0), (dec(mm(P['mount']), 3000), '#f2c31f', 1.0),
         (dec(mm(P['camera']), 3000), '#333333', 1.0), (dec(mm(P['servo']), 3000), '#444444', 1.0), (H, '#4682dc', 1.0)]
JTd = dec(JT, 24000)

def draw(ax, elev, azim, jaw_angles, title, lim, frame=None, labels=('forward (mm)', 'left', 'up')):
    tris, cols = [], []
    items = fixed + [(JTd.copy().apply_transform(R(a)), c, al) for a, c, al in jaw_angles]
    e, a_ = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a_), np.cos(e) * np.sin(a_), np.sin(e)])
    light = view + np.array([0, 0, 0.6]) + 0.3 * np.cross(view, [0, 0, 1]); light /= np.linalg.norm(light)
    for m, c, al in items:
        v = trimesh.transform_points(m.vertices, HOLD if frame is None else frame); t = v[m.faces]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        k = 0.35 + 0.65 * np.abs(n @ light)
        base = np.array(to_rgba(c)); fc = np.clip(base[None, :3] * k[:, None], 0, 1)
        tris.append(t); cols.append(np.hstack([fc, np.full((len(t), 1), al)]))
    ax.add_collection3d(Poly3DCollection(np.concatenate(tris), facecolors=np.concatenate(cols), edgecolor='none'))
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0)); ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, pad=22); ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel(labels[2])

fig = plt.figure(figsize=(19, 7.5))
big = ((-120, 115), (-75, 75), (-90, 95))
draw(fig.add_subplot(131, projection='3d'), 0, 0, [(-0.17, '#e55a3c', 1), (0.45, '#9b59b6', 0.35)],
     'side view, same angle as the first version (raked grip)', ((-60, 60), (-150, 95), (-115, 100)),
     frame=np.eye(4), labels=('x', 'y (camera up)', 'z (jaws point -z)'))
draw(fig.add_subplot(132, projection='3d'), -90, -90, [(-0.17, '#e55a3c', 1), (0.45, '#9b59b6', 0.35)],
     'from below: thumb/index loops, closed (red) vs 48 mm open (purple)', ((-80, 115), (-75, 75), (-150, 95)))
draw(fig.add_subplot(133, projection='3d'), -20, -55, [(0.15, '#e55a3c', 1)],
     '3/4 from below-front-right', ((-80, 115), (-75, 75), (-150, 95)))
plt.tight_layout(); plt.savefig(f"{OUT}/soumi_preview.png", dpi=110); print("saved")
