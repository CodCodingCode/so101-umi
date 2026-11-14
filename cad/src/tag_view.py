import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import numpy as np, trimesh, json, cv2, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from asm import parts, T
OUT = sys.argv[1]; TAG = float(sys.argv[2])
P, _, _ = parts(0.0)
def mm(m): m = m.copy(); m.apply_scale(1000); return m
fol, jaw, srv = mm(P['follower']), mm(P['jaw']), mm(P['servo'])
def R(a): return trimesh.transformations.rotation_matrix(a, [0, -1, 0], [20.2, 0, -23.4])
Mm = T([-0.0150, 0.0240, -0.0315], [0.4997981226682881, -0.5001999599912927, 0.4998017973423795, -0.5001999599912927])
Mc = Mm @ T([-0.04941, -0.03813, 0.0175], [0.37978693649389456, -0.5964584309361936, -0.37978833153177, -0.596456240026065])
Mc[:3, 3] *= 1000; Mc[:3, 3] += Mc[:3, 2] * 20.0
Ci = np.linalg.inv(Mc); FX = 543.0
def proj(v):
    pc = trimesh.transform_points(np.atleast_2d(v), Ci); z = pc[:, 2]
    return np.c_[FX * pc[:, 0] / z + 320, FX * pc[:, 1] / z + 240], z
patches = json.load(open(f"{OUT}/patches.json"))
# tag squares: centred on each visible patch, in its plane, edges along x and along the finger
tags = {}
for nm in ['fixed', 'moving']:
    p = patches[nm][0]; c = np.array(p['centre']); n = np.array(p['normal']); n /= np.linalg.norm(n)
    u = np.array([1.0, 0, 0]); u -= n * (u @ n); u /= np.linalg.norm(u); w = np.cross(n, u)
    h = TAG / 2; tags[nm] = (c, np.array([c + h * (-u - w), c + h * (u - w), c + h * (u + w), c + h * (-u + w)]))
dict4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
def tag_img(i):
    return cv2.aruco.generateImageMarker(dict4, i, 120) if hasattr(cv2.aruco, 'generateImageMarker') else cv2.aruco.drawMarker(dict4, i, 120)
fig, axs = plt.subplots(1, 3, figsize=(18, 4.9))
light = np.array([0.3, 0.8, 0.5]); light /= np.linalg.norm(light)
report = {}
for ax, a, title in zip(axs, [-0.17, 0.15, 0.45], ['closed', '~25 mm open', '48 mm open']):
    J = jaw.copy().apply_transform(R(a))
    polys, cols, depth = [], [], []
    for m, col in [(fol, (0.95, 0.76, 0.12)), (srv, (0.3, 0.3, 0.3)), (J, (0.9, 0.35, 0.24))]:
        uv, z = proj(m.vertices); t = uv[m.faces]; zf = z[m.faces].mean(1)
        keep = (z[m.faces] > 5).all(1)
        k = 0.35 + 0.65 * np.abs(m.face_normals @ light)
        polys += list(t[keep]); cols += list(np.clip(np.array(col)[None] * k[keep, None], 0, 1)); depth += list(zf[keep])
    order = np.argsort(depth)[::-1]
    ax.add_collection(PolyCollection([polys[i] for i in order], facecolors=[cols[i] for i in order], edgecolors='none'))
    for nm, tid in [('fixed', 0), ('moving', 1)]:
        c, sq = tags[nm]
        if nm == 'moving': sq = trimesh.transform_points(sq, R(a)); c = trimesh.transform_points([c], R(a))[0]
        uv, z = proj(sq)
        H_, _ = cv2.findHomography(np.float32([[0, 0], [120, 0], [120, 120], [0, 120]]), np.float32(uv))
        warped = cv2.warpPerspective(tag_img(tid), H_, (640, 480), flags=cv2.INTER_AREA, borderValue=255)
        mask = cv2.warpPerspective(np.full((120, 120), 255, np.uint8), H_, (640, 480)) > 0
        rgba = np.zeros((480, 640, 4)); rgba[..., :3] = (warped[..., None] / 255.0); rgba[..., 3] = mask
        ax.imshow(rgba, extent=(0, 640, 480, 0), zorder=5, interpolation='nearest')
        side = np.mean([np.linalg.norm(uv[i] - uv[(i + 1) % 4]) for i in range(4)])
        report[f"{title} / ID {tid} ({nm})"] = dict(px_side=round(float(side)), dist_mm=round(float(np.linalg.norm(c - Mc[:3, 3]))))
    ax.set_xlim(0, 640); ax.set_ylim(480, 0); ax.set_aspect('equal'); ax.set_facecolor('#dfe6ee')
    ax.set_title(f"jaws {title}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Simulated wrist-camera view (640x480): 8 mm tags, ID 0 on the fixed jaw, ID 1 on the moving jaw, ~20 mm back from the tips", fontsize=11); plt.tight_layout(); plt.savefig(f"{OUT}/wrist_view_tags.png", dpi=100)
print(json.dumps(report, indent=1)); print({k: (np.round(v[0], 1).tolist()) for k, v in tags.items()})
