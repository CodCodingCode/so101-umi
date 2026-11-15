"""Render where the ArUco cube tags (IDs 20-23) go on the SO-UMI handle.

usage: python tag_cube_view.py CAD_DIR      (reads the STLs there, writes CAD_DIR/tag_placement_cube.png)

Placement matches tag_sheet.py. Directions are as you hold it (camera on top, jaws pointing away):
  ID 20  right face (moving-jaw side)  top edge toward the jaws
  ID 21  left face (fixed-jaw side)    top edge toward the jaws
  ID 22  top face (camera side)        top edge toward the jaws
  ID 23  back face (toward you)        top edge up
"""
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import numpy as np, trimesh, cv2, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.colors import to_rgba
from asm import parts, T

OUT = sys.argv[1]
P, _, Mj = parts(0.0)
def mm(m): m = m.copy(); m.apply_scale(1000); return m
H = trimesh.load(f"{OUT}/soumi_handle.stl")
FL = trimesh.load(f"{OUT}/soumi_fixed_jaw_loop.stl")
Mj = Mj.copy(); Mj[:3, 3] *= 1000
JT = trimesh.load(f"{OUT}/soumi_moving_jaw_loop.stl"); JT.apply_transform(Mj @ T([0, 0, 18.9], [1, 0, 0, 0]))
def R(a): return trimesh.transformations.rotation_matrix(a, [0, -1, 0], [20.2, 0, -23.4])
# "as held" frame: x = forward (jaws), y = left, z = up (camera side)
HOLD = np.eye(4); HOLD[:3, :3] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
def dec(m, n): return m.simplify_quadric_decimation(face_count=n) if len(m.faces) > n else m
items = [(dec(FL, 20000), '#f2c31f'), (dec(JT, 20000).apply_transform(R(0.15)), '#e55a3c'),
         (dec(mm(P['mount']), 3000), '#f2c31f'), (dec(mm(P['camera']), 3000), '#333333'),
         (dec(mm(P['servo']), 3000), '#444444'), (H, '#4682dc')]

# cube (gripper frame, mm), from build.py: 36 mm cube centred at (0, -2, 58), 30 mm pockets 0.6 mm deep
C, CC, TAG, STICK = 36.0, np.array([0.0, -2.0, 58.0]), 25.0, 30.0
FACES = [  # id, outward normal, tag "top edge" direction, label   (gripper frame: +x moving jaw, +y camera, -z jaws)
    (20, [1, 0, 0], [0, 0, -1], "ID 20\nright face"),
    (21, [-1, 0, 0], [0, 0, -1], "ID 21\nleft face"),
    (22, [0, 1, 0], [0, 0, -1], "ID 22\ntop face"),
    (23, [0, 0, 1], [0, 1, 0], "ID 23\nback face"),
]
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
def marker_bits(i):
    f = cv2.aruco.generateImageMarker if hasattr(cv2.aruco, 'generateImageMarker') else cv2.aruco.drawMarker
    return f(DICT, i, 6) < 128          # 6x6 cells: 4x4 bits + 1-cell black border

def sticker(tid, n, u):
    """Quads (gripper frame) for one sticker: white margin strips + 6x6 marker cells, plus its top-edge line."""
    n, u = np.array(n, float), np.array(u, float); r = np.cross(u, n)     # image x = r, image up = u
    c = CC + n * (C / 2 + 0.25)
    quads, cols = [], []
    def quad(x0, x1, y0, y1, col):     # x along r, y along u, mm from centre
        quads.append([c + r * x0 + u * y0, c + r * x1 + u * y0, c + r * x1 + u * y1, c + r * x0 + u * y1]); cols.append(col)
    h, s = TAG / 2, STICK / 2
    quad(-s, s, h, s, 'white'); quad(-s, s, -s, -h, 'white'); quad(-s, -h, -h, h, 'white'); quad(h, s, -h, h, 'white')
    bits, cell = marker_bits(tid), TAG / 6
    for i in range(6):
        for j in range(6):
            quad(-h + j * cell, -h + (j + 1) * cell, h - (i + 1) * cell, h - i * cell, 'black' if bits[i, j] else 'white')
    top = [c + r * (-s) + u * (s - 0.6), c + r * s + u * (s - 0.6)]
    return quads, cols, top, c

def draw(ax, elev, azim, lim, title, label_off=30.0):
    e, a = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    light = view + np.array([0, 0, 0.6]); light /= np.linalg.norm(light)
    tris, cols = [], []
    for m, col in items:
        t = trimesh.transform_points(m.vertices, HOLD)[m.faces]
        nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        k = 0.35 + 0.65 * np.abs(nrm @ light)
        tris.append(t); cols.append(np.c_[np.clip(np.array(to_rgba(col))[None, :3] * k[:, None], 0, 1), np.ones(len(t))])
    T_all = np.concatenate(tris); C_all = np.concatenate(cols)
    order = np.argsort(T_all.mean(1) @ view)                                  # far to near
    ax.computed_zorder = False
    ax.add_collection3d(Poly3DCollection(T_all[order], facecolors=C_all[order], edgecolor='none'))
    tq, tc, lines, leaders = [], [], [], []
    for tid, n, u, label in FACES:
        n_hold = HOLD[:3, :3] @ np.array(n, float)
        if n_hold @ view <= 0.05:                                             # face turned away: hidden
            continue
        q, c, top, centre = sticker(tid, n, u)
        tq += [trimesh.transform_points(np.array(x), HOLD) for x in q]; tc += c
        lines.append(trimesh.transform_points(np.array(top), HOLD))
        p0 = trimesh.transform_points(centre[None], HOLD)[0]
        p = p0 + n_hold * label_off
        leaders.append([p0 + n_hold * (C / 2 * 0.0), p])
        ax.text(*p, label, fontsize=9, weight='bold', ha='center', va='center', color='#111',
                bbox=dict(boxstyle='round,pad=0.25', fc='#fff8dc', ec='#999', lw=0.6, alpha=0.95), zorder=10)
    ax.add_collection3d(Poly3DCollection(tq, facecolors=tc, edgecolor='none'))
    ax.add_collection3d(Line3DCollection(lines, colors='#d62728', linewidths=2.2))
    ax.add_collection3d(Line3DCollection([[a + (b - a) * 0.55, b] for a, b in leaders], colors='#777', linewidths=0.8))
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0)); ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off(); ax.set_title(title, fontsize=11, pad=4)

full = ((-95, 105), (-80, 80), (-150, 60))
zoom = ((-115, -5), (-50, 50), (-45, 45))          # around the cube (centre at forward -58, up -2)
fig = plt.figure(figsize=(16, 14))
draw(fig.add_subplot(221, projection='3d'), 25, -140, full,
     "1. The cube is on the back of the handle, toward you\n(view from behind, right side; jaws point away)", label_off=38)
draw(fig.add_subplot(222, projection='3d'), 62, -165, full,
     "2. Roughly what the overhead camera sees\n(gripper held level: the top tag faces up)", label_off=38)
draw(fig.add_subplot(223, projection='3d'), 25, -140, zoom, "3. Close-up, right side: IDs 20, 22, 23", label_off=26)
draw(fig.add_subplot(224, projection='3d'), 25, 140, zoom, "4. Close-up, left side: IDs 21, 22, 23", label_off=26)
fig.suptitle("Where the cube tags go: a 25 mm tag on a 30 mm sticker in each of the 4 pockets.\n"
             "Red line = the tag's top edge (matches the red arrows on the tag sheet).", fontsize=13)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(f"{OUT}/tag_placement_cube.png", dpi=120); print("saved", f"{OUT}/tag_placement_cube.png")
