"""Render where the 8 mm finger tags (ID 0 fixed jaw, ID 1 moving jaw) go, seen from outside the gripper.

usage: python finger_tag_view.py CAD_DIR     (reads the STLs there, writes CAD_DIR/tag_placement_fingers.png)

Spots come from finger_tag_patches.json: the flat top face of each finger blade (the face the wrist
camera sees), ~20 mm back from the tip. The robot's jaws get the same tags in the same spots.
"""
import sys, os, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import numpy as np, trimesh, cv2, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.colors import to_rgba
from asm import parts, T

OUT = sys.argv[1]
TAG, QUIET, ANGLE = 8.0, 8.0 / 6, 0.30          # tag mm, white border mm, jaw angle for the picture (rad)
P, _, Mj = parts(0.0)
def mm(m): m = m.copy(); m.apply_scale(1000); return m
H = trimesh.load(f"{OUT}/soumi_handle.stl")
FL = trimesh.load(f"{OUT}/soumi_fixed_jaw_loop.stl")
Mj = Mj.copy(); Mj[:3, 3] *= 1000
JT = trimesh.load(f"{OUT}/soumi_moving_jaw_loop.stl"); JT.apply_transform(Mj @ T([0, 0, 18.9], [1, 0, 0, 0]))
def R(a): return trimesh.transformations.rotation_matrix(a, [0, -1, 0], [20.2, 0, -23.4])
HOLD = np.eye(4); HOLD[:3, :3] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]   # x forward, y left, z up (camera side)
def dec(m, n): return m.simplify_quadric_decimation(face_count=n) if len(m.faces) > n else m
JA = dec(JT, 24000).apply_transform(R(ANGLE))
items = [(dec(FL, 24000), '#f2c31f'), (JA, '#e55a3c'), (dec(mm(P['mount']), 3000), '#f2c31f'),
         (dec(mm(P['camera']), 3000), '#333333'), (dec(mm(P['servo']), 3000), '#444444'), (H, '#4682dc')]

patches = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "finger_tag_patches.json")))
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
def bits(i):
    f = cv2.aruco.generateImageMarker if hasattr(cv2.aruco, 'generateImageMarker') else cv2.aruco.drawMarker
    return f(DICT, i, 6) < 128

def tag_quads(tid, c, n):
    n = n / np.linalg.norm(n); u = np.array([1.0, 0, 0]); u -= n * (u @ n); u /= np.linalg.norm(u)
    w = np.cross(n, u); c = c + n * 0.3
    quads, cols = [], []
    def quad(x0, x1, y0, y1, col):
        quads.append([c + u * x0 + w * y0, c + u * x1 + w * y0, c + u * x1 + w * y1, c + u * x0 + w * y1]); cols.append(col)
    h, s = TAG / 2, TAG / 2 + QUIET
    quad(-s, s, h, s, 'white'); quad(-s, s, -s, -h, 'white'); quad(-s, -h, -h, h, 'white'); quad(h, s, -h, h, 'white')
    b, cell = bits(tid), TAG / 6
    for i in range(6):
        for j in range(6):
            quad(-h + j * cell, -h + (j + 1) * cell, h - (i + 1) * cell, h - i * cell, 'black' if b[i, j] else 'white')
    return quads, cols

TAGS = []
for nm, tid, label in [('fixed', 0, 'ID 0\nfixed jaw'), ('moving', 1, 'ID 1\nmoving jaw')]:
    p = patches[nm][0]; c = np.array(p['centre'], float); n = np.array(p['normal'], float)
    if nm == 'moving':
        c = trimesh.transform_points(c[None], R(ANGLE))[0]; n = R(ANGLE)[:3, :3] @ n
    mesh = FL if nm == 'fixed' else JA
    side = mesh.vertices[np.abs(mesh.vertices[:, 0] - c[0]) < 6]          # fingertip = furthest-forward point of that finger
    tip = side[np.argmin(side[:, 2])]
    q, col = tag_quads(tid, c, n)
    TAGS.append(dict(q=q, col=col, c=c, n=n / np.linalg.norm(n), tip=tip, label=label))

def draw(ax, elev, azim, lim, title, label_off=22.0, dims=True, lab_shift=None, dim_shift=(0, 0, 10)):
    e, a = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    light = view + np.array([0, 0, 0.6]); light /= np.linalg.norm(light)
    tris, cols = [], []
    for m, colr in items:
        t = trimesh.transform_points(m.vertices, HOLD)[m.faces]
        nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        k = 0.35 + 0.65 * np.abs(nrm @ light)
        tris.append(t); cols.append(np.c_[np.clip(np.array(to_rgba(colr))[None, :3] * k[:, None], 0, 1), np.ones(len(t))])
    T_all = np.concatenate(tris); C_all = np.concatenate(cols); order = np.argsort(T_all.mean(1) @ view)
    ax.computed_zorder = False
    ax.add_collection3d(Poly3DCollection(T_all[order], facecolors=C_all[order], edgecolor='none'))
    tq, tc, lines, leads = [], [], [], []
    for k, t in enumerate(TAGS):
        tq += [trimesh.transform_points(np.array(x), HOLD) for x in t['q']]; tc += t['col']
        c = trimesh.transform_points(t['c'][None], HOLD)[0]; n = HOLD[:3, :3] @ t['n']
        lab = c + (np.array(lab_shift[k], float) if lab_shift else n * label_off + np.array([0, 0, 4]))
        leads.append([c + n * 2, lab])
        ax.text(*lab, t['label'], fontsize=9, weight='bold', ha='center', va='bottom', color='#111',
                bbox=dict(boxstyle='round,pad=0.25', fc='#fff8dc', ec='#999', lw=0.6, alpha=0.95), zorder=10)
        if dims:
            tip = trimesh.transform_points(t['tip'][None], HOLD)[0]
            front = c + np.array([TAG / 2 + QUIET, 0, 0])                         # tag's front edge, along "forward"
            lines.append([c + n * 0.6, np.array([tip[0], c[1], c[2]]) + n * 0.6])
    ax.add_collection3d(Poly3DCollection(tq, facecolors=tc, edgecolor='none'))
    ax.add_collection3d(Line3DCollection(leads, colors='#777', linewidths=0.8))
    if dims:
        ax.add_collection3d(Line3DCollection(lines, colors='#d62728', linewidths=1.6, linestyles='--'))
        mid = np.mean([np.mean(l, 0) for l in lines], 0)
        ax.text(*(mid + np.array(dim_shift, float)), "red dashes: ~20 mm from\ntag centre to fingertip", fontsize=8,
                color='#d62728', ha='center', va='bottom', zorder=10)
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0)); ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off(); ax.set_title(title, fontsize=11, pad=4)

full = ((-80, 115), (-75, 75), (-120, 70))
zoom = ((35, 115), (-45, 45), (-30, 40))       # around the fingertips (forward ~78..98)
fig = plt.figure(figsize=(17, 13))
draw(fig.add_subplot(221, projection='3d'), 38, -150, full,
     "1. Whole handheld, from behind and above\n(jaws point away from you; the camera hides part of ID 0 from here)",
     label_off=30, dims=False)
draw(fig.add_subplot(222, projection='3d'), 50, 25, zoom,
     "2. Close-up from in FRONT of the jaws, above\n(mirror image: the moving jaw is on the left from here)",
     label_off=16, dims=False)
draw(fig.add_subplot(223, projection='3d'), 88, 180, zoom,
     "3. Straight down on the finger tops\n(jaws point up the page; fixed jaw on the left, moving jaw on the right)",
     lab_shift=[(0, 24, 0), (0, -24, 0)], dim_shift=(0, 0, 0))
ax4 = fig.add_subplot(224)
wv = plt.imread(f"{OUT}/tag_placement_wrist_view.png")
hh, ww = wv.shape[:2]
ax4.imshow(wv[int(hh * 0.08):, :ww // 3]); ax4.set_axis_off()
ax4.set_title("4. What the wrist camera sees (jaws closed)", fontsize=11, pad=4)
fig.suptitle("Finger tags: 8 mm, ID 0 on the fixed jaw and ID 1 on the moving jaw, on each finger's top face ~20 mm back from the tip.\n"
             "Optional: the servo already measures the jaw opening. Put the same pair in the same spots on the robot's jaws.",
             fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig(f"{OUT}/tag_placement_fingers.png", dpi=110); print("saved", f"{OUT}/tag_placement_fingers.png")
