"""Tag stickers as thin geometry (white parts + black parts), in the 'as held' frame, metres.
Cube tags 20-23: 25 mm tag on a 30 mm sticker (tag_cube_view.py FACES / umi/convert.py CUBE_FACES).
Finger tags 0/1: 8 mm, on finger_tag_patches.json spots (moving jaw turned by the render's jaw angle)."""
import sys, os, json, numpy as np, trimesh, cv2
CAD = sys.argv[1]; OUT = sys.argv[2]; ANGLE = float(sys.argv[3])
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
def bits(i):
    f = cv2.aruco.generateImageMarker if hasattr(cv2.aruco, "generateImageMarker") else cv2.aruco.drawMarker
    return f(DICT, i, 6) < 128                      # True = black, 6x6 cells incl. border
def slab(c, x_ax, y_ax, n, x0, x1, y0, y1, z0, z1):
    """Box spanning [x0,x1]x[y0,y1] in the tag plane (mm) and [z0,z1] along the normal."""
    pts = [c + x_ax * x + y_ax * y + n * z for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    return trimesh.convex.convex_hull(np.array(pts))
def tag(tid, c, x_ax, y_ax, n, size, quiet):
    """x_ax = marker image right, y_ax = marker image up (top edge), n = outward normal."""
    white, black = [], []; h, s, cell = size / 2, size / 2 + quiet, size / 6
    white.append(slab(c, x_ax, y_ax, n, -s, s, -s, s, 0.0, 0.25))            # sticker
    black.append(slab(c, x_ax, y_ax, n, -h, h, -h, h, 0.25, 0.40))            # marker square
    b = bits(tid)
    for i in range(6):
        for j in range(6):
            if not b[i, j]:
                white.append(slab(c, x_ax, y_ax, n, -h + j * cell, -h + (j + 1) * cell, h - (i + 1) * cell, h - i * cell, 0.40, 0.50))
    return white, black
W, B = [], []
# cube (gripper frame, mm)
C, CC = 36.0, np.array([0.0, -2.0, 58.0])
for tid, n, u in [(20, [1, 0, 0], [0, 0, -1]), (21, [-1, 0, 0], [0, 0, -1]), (22, [0, 1, 0], [0, 0, -1]), (23, [0, 0, 1], [0, 1, 0])]:
    n, u = np.array(n, float), np.array(u, float); r = np.cross(u, n)
    w, b = tag(tid, CC + n * (C / 2 - 0.6), r, u, n, 25.0, 2.5); W += w; B += b      # sticker sits in the 0.6 mm pocket
# fingers (same construction as finger_tag_view.py)
def R(a): return trimesh.transformations.rotation_matrix(a, [0, -1, 0], [20.2, 0, -23.4])
P = json.load(open(os.path.join(CAD, "src", "finger_tag_patches.json")))
for nm, tid in [("fixed", 0), ("moving", 1)]:
    c = np.array(P[nm][0]["centre"], float); n = np.array(P[nm][0]["normal"], float); n /= np.linalg.norm(n)
    if nm == "moving": c = trimesh.transform_points(c[None], R(ANGLE))[0]; n = R(ANGLE)[:3, :3] @ n
    u = np.array([1.0, 0, 0]); u -= n * (u @ n); u /= np.linalg.norm(u); w_ = np.cross(n, u)
    w, b = tag(tid, c - n * 0.1, u, w_, n, 8.0, 8.0 / 6); W += w; B += b
HOLD = np.eye(4); HOLD[:3, :3] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
for name, ms in (("tags_white", W), ("tags_black", B)):
    m = trimesh.util.concatenate(ms); m.apply_transform(HOLD); m.apply_scale(0.001); m.export(f"{OUT}/{name}.stl"); print(name, len(m.faces))
