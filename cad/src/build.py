"""SO-UMI handheld v0.4 ("pinch"). See ../README.md for how it's held.

Parts (all in the SO-101 'gripper' body frame, mm):
  * handle: horn-replica plate + neck + pistol grip (raked 12 deg) + ArUco cube
  * moving jaw + loop: official moving jaw with a finger loop fused under it
  * fixed jaw (follower) + loop: official wrist-roll follower with a finger loop fused under its jaw

Gripper frame: wrist-roll axis = +z (horn face at z=0), jaws point to -z, wrist camera on +y,
moving jaw on +x, jaw pivot axis along y at x=20.2, z=-23.4.
Held like a pistol: camera on top (+y), jaws forward (-z), grip hanging under the body. Thumb and
index go through the two loops (hole axis along z, like rings) and pinch along x to close the jaws.
"""
import sys, json, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, trimesh
import manifold3d as mf
from asm import parts, T

OUT = sys.argv[1]
PIVOT = np.array([20.2, -23.4])  # jaw pivot (x, z)
CLOSED, OPEN48 = -0.17, 0.45     # jaw angle closed / 48 mm open (rad)

# ---------- helpers ----------
def to_mf(m):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                               tri_verts=np.asarray(m.faces, np.uint32)))
def to_tm(M):
    me = M.to_mesh()
    return trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts), process=True)
def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
def cyl_z(r, z0, z1, x=0.0, y=0.0, n=96):
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])
def mat(M4):
    return [list(map(float, r)) for r in np.asarray(M4)[:3, :4]]
def rot_about_pivot(angle):
    """Jaw rotation by +angle (opening) about the pivot axis, mm."""
    return trimesh.transformations.rotation_matrix(angle, [0, -1, 0], [PIVOT[0], 0, PIVOT[1]])
def oval_z(z0, z1, cx, cy, rx, ry, n=96):
    """Elliptic prism along z, ellipse in xy."""
    return mf.CrossSection.circle(1.0, n).scale([rx, ry]).extrude(z1 - z0).translate([cx, cy, z0])
def mm(m):
    m = m.copy(); m.apply_scale(1000.0); return m

P0, tip_m, _ = parts(0.0)
ref = {k: mm(v) for k, v in P0.items()}
tip = tip_m * 1000

# ---------- finger loops (rings; digit points forward through them) ----------
LOOP_Z, LOOP_T = -60.0, 14.0          # loop centre plane and thickness along the digit
RI_X, RI_Y = 12.0, 14.0               # hole: 24 (pinch direction) x 28 mm, fits a thumb
WALL, E = 6.0, 1.8                    # wall, edge round-over
RO_X, RO_Y = RI_X + WALL, RI_Y + WALL
LOOP_Y = -36.0                        # loops hang under the jaws, hidden from the wrist camera
def loop(cx, cy, cz):
    z0, z1 = cz - LOOP_T / 2, cz + LOOP_T / 2
    out = (oval_z(z0, z0 + 0.01, cx, cy, RO_X - E, RO_Y - E) + oval_z(z0 + E, z1 - E, cx, cy, RO_X, RO_Y) +
           oval_z(z1 - 0.01, z1, cx, cy, RO_X - E, RO_Y - E)).hull()
    hole = (oval_z(z0 - 1, z1 + 1, cx, cy, RI_X, RI_Y) +
            (oval_z(z0 - 1, z0, cx, cy, RI_X + E, RI_Y + E) + oval_z(z0 + E, z0 + E + .01, cx, cy, RI_X, RI_Y)).hull() +
            (oval_z(z1, z1 + 1, cx, cy, RI_X + E, RI_Y + E) + oval_z(z1 - E - .01, z1 - E, cx, cy, RI_X, RI_Y)).hull())
    top = box(cx - 9, cx + 9, cy + RI_Y + 0.5, cy + RO_Y - 0.5, z0 + 1, z1 - 1)   # top wall, for joining
    return out, hole, top

# Moving-jaw loop: at mid-travel it sits straight "ahead" of the pivot (same x), so it moves almost
# purely sideways (x = the pinch direction) as the jaw turns: ~22 mm from closed to 48 mm open.
MID = 0.5 * (CLOSED + OPEN48)
M_MID = np.array([PIVOT[0], LOOP_Y, LOOP_Z])
mo, mh, mt = loop(*M_MID)
to0 = mat(rot_about_pivot(-MID))                      # mid pose -> jaw angle-0 frame
mo, mh, mt = mo.transform(to0), mh.transform(to0), mt.transform(to0)
# boss fused to the jaw's lower block (below the horn-screw counterbores, which end at z=-31.5)
boss_m = box(10.5, 29.5, -34, -19.5, -52, -34) + box(10.5, 29.5, -19.5, -13, -52, -44)
bar_m = (box(11, 29, -32, -14, -52, -46) + mt).hull()
jaw_loop_part = (boss_m + bar_m + mo) - mh
jaw0 = to_mf(ref["jaw"])
jaw_trig = jaw0 + jaw_loop_part

# Fixed-jaw loop: mirrored on the other side so the loops are ~6 mm apart when the jaws are closed.
mc = rot_about_pivot(CLOSED - MID) @ np.append(M_MID, 1)      # moving loop centre when closed
F_C = np.array([mc[0] - 2 * RO_X - 6.0, LOOP_Y, LOOP_Z])
fo, fh, ft = loop(*F_C)
boss_f = box(-31, -15, -17, -7.5, -64, -44)                   # fused into the underside of the fixed jaw
bar_f = (box(-31, -15, -24, -10, -62, -48) + ft).hull()
fol0 = to_mf(ref["follower"])
fixed_loop_part = (boss_f + bar_f + fo) - fh
follower_loop = fol0 + fixed_loop_part

# ---------- HANDLE ----------
HORN_HOLES = [(-4.83, -4.95), (-4.83, 4.95), (5.07, -4.95), (5.07, 4.95)]
AX = (0.12, 0.0)
PILOT_D = 2.7                 # M3 self-tapping into PLA/PETG (4.2 for heat-set inserts)
# Pistol grip, same shape and 12 deg rake as v0 (30 x 36 mm ellipse, 105 mm long), but moved
# forward under the gripper body and centred between the two loops so thumb and index reach them.
GX = 0.5 * (F_C[0] + M_MID[0]) - 3.6          # ~centred between the loops, nudged off the moving jaw's boss
GRIP_TOP = np.array([GX, -30.0, -25.0])       # top centre, just under the gripper body (y=-28); clears the tilted moving loop when closed
GRX, GRZ, GRIP_LEN, RAKE = 15.0, 18.0, 105.0, 12.0
g = mf.Manifold.cylinder(GRIP_LEN, 1.0, 1.0, 96).scale([GRX, GRZ, 1.0])     # along +z, ellipse in xy
Rg = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)                    # column axis -> -y
rake = trimesh.transformations.rotation_matrix(np.radians(-RAKE), [1, 0, 0])[:3, :3]  # bottom swings back (+z)
Rt = rake @ Rg
grip = g.transform(np.hstack([Rt, GRIP_TOP[:, None]]).tolist())
bottom = GRIP_TOP + Rt @ np.array([0, 0, GRIP_LEN])
nose = mf.Manifold.sphere(1.0, 64).scale([GRX, 9.0, GRZ]).translate(bottom.tolist())   # rounded bottom
# top strap under the gripper body: ties the grip top back to the neck (and is the "beavertail"
# your thumb-index web rests under). Kept off the moving jaw's side (x <= GX+11).
strap = box(GX - GRX, GX + 11, -40, -29.6, GRIP_TOP[2] - GRZ, 30)
neck = box(-14, 14, -32, 15, 1.2, 40)
post = strap
C = 36.0; cc = np.array([0.0, -2.0, 40.0 + C / 2])                                   # tag cube on the back (as v0)
cube = box(cc[0] - C/2, cc[0] + C/2, cc[1] - C/2, cc[1] + C/2, cc[2] - C/2, cc[2] + C/2)
pk = 30.0; d = 0.6
pockets = (box(cc[0] + C/2 - d, cc[0] + C/2 + 1, cc[1] - pk/2, cc[1] + pk/2, cc[2] - pk/2, cc[2] + pk/2) +
           box(cc[0] - C/2 - 1, cc[0] - C/2 + d, cc[1] - pk/2, cc[1] + pk/2, cc[2] - pk/2, cc[2] + pk/2) +
           box(cc[0] - pk/2, cc[0] + pk/2, cc[1] + C/2 - d, cc[1] + C/2 + 1, cc[2] - pk/2, cc[2] + pk/2) +
           box(cc[0] - pk/2, cc[0] + pk/2, cc[1] - pk/2, cc[1] + pk/2, cc[2] + C/2 - d, cc[2] + C/2 + 1))
zt = box(-20, 20, 4, 6.5, 20, 24) + box(-20, 20, 4, 6.5, 30, 34)      # zip-tie slots (cables)
body = cyl_z(15.0, 1.2, 12.0, *AX) + neck + post + grip + nose + cube
body = body - box(-80, 80, -29.5, 40, -120, 1.2)     # keep clear of the gripper body above z=1.2
handle = body + cyl_z(9.85, 0.0, 2.0, *AX)            # horn-replica disc
handle = handle - pockets - zt
for hx, hy in HORN_HOLES:
    handle = handle - cyl_z(PILOT_D / 2, -1, 11.0, hx, hy, 32)

# ---------- checks ----------
report = {}
cam_hull = to_mf(ref["camera"].convex_hull)
H = handle
fixed = {"follower+loop": follower_loop, "servo": to_mf(ref["servo"]), "mount": to_mf(ref["mount"]),
         "camera": cam_hull, "handle": H}
report["handle_volume_cm3"] = round(H.volume() / 1000, 1)
report["handle_overlap_mm3"] = {k: round((H ^ v).volume(), 2) for k, v in fixed.items() if k != "handle"}
report["fixed_loop_fused_overlap_mm3"] = round((fol0 ^ fixed_loop_part).volume(), 1)
report["moving_loop_fused_overlap_mm3"] = round((jaw0 ^ jaw_loop_part).volume(), 1)
def bodies(M): return int(sum(b.volume > 0 for b in to_tm(M).split(only_watertight=False)))
report["bodies"] = {"jaw+loop": bodies(jaw_trig), "follower+loop": bodies(follower_loop), "handle": bodies(H)}
sweep = {}
# new geometry only: the stock jaw already touches the servo horn in the official meshes (<2 mm^3)
for a in [-0.17, 0.0, 0.15, 0.3, 0.45, 0.6, 0.9, 1.2]:
    M_ = mat(rot_about_pivot(a))
    L = jaw_loop_part.transform(M_)
    hits = {k: round((L ^ v).volume(), 2) for k, v in fixed.items()}
    hits["stock jaw vs fixed loop"] = round((jaw0.transform(M_) ^ fixed_loop_part).volume(), 2)
    c = rot_about_pivot(a - MID) @ np.append(M_MID, 1)
    sweep[f"{a:+.2f}rad"] = {"overlaps_mm3": hits, "moving_loop_centre": np.round(c[:3], 1).tolist(),
                             "loop_centre_spacing_mm": round(float(c[0] - F_C[0]), 1)}
report["sweep"] = sweep
report["fixed_loop_centre"] = F_C.round(1).tolist()
report["grip"] = {"top_centre": GRIP_TOP.round(1).tolist(), "rake_deg": RAKE,
                  "front_of_grip_top_z": round(float(GRIP_TOP[2] - GRZ), 1),
                  "grip_front_to_loop_centre_mm (index reach)": round(float(GRIP_TOP[2] - GRZ - LOOP_Z), 1),
                  "web_to_loop_centre_mm (thumb reach, web at grip back-top)": round(float(np.linalg.norm(
                      np.array([F_C[0], F_C[1], F_C[2]]) - np.array([GX, -40.0, GRIP_TOP[2] + GRZ]))), 1),
                  "grip_back_to_jaw_tip_mm": round(float(GRIP_TOP[2] + GRZ - tip[2]), 1)}

# wrist-camera visibility (640x480, fx~543) with occlusion by the gripper itself
Mm = T([-0.0150, 0.0240, -0.0315], [0.4997981226682881, -0.5001999599912927, 0.4998017973423795, -0.5001999599912927])
Mc = Mm @ T([-0.04941, -0.03813, 0.0175], [0.37978693649389456, -0.5964584309361936, -0.37978833153177, -0.596456240026065])
Mc[:3, 3] *= 1000
Mc[:3, 3] += Mc[:3, 2] * 20.0
occl = trimesh.util.concatenate([ref["follower"], ref["jaw"], ref["servo"], ref["mount"]])
ray = trimesh.ray.ray_triangle.RayMeshIntersector(occl)
def visible(points):
    Ci = np.linalg.inv(Mc); pc = trimesh.transform_points(points, Ci)
    front = pc[:, 2] > 1
    u = 543 * pc[:, 0] / np.maximum(pc[:, 2], 1e-6) + 320
    w = 543 * pc[:, 1] / np.maximum(pc[:, 2], 1e-6) + 240
    idx = np.where(front & (u >= 0) & (u < 640) & (w >= 0) & (w < 480))[0]
    if len(idx) == 0: return 0, 0
    o = np.repeat(Mc[None, :3, 3], len(idx), 0); dv = points[idx] - o
    dist = np.linalg.norm(dv, axis=1); dv /= dist[:, None]
    locs, ri, _ = ray.intersects_location(o, dv, multiple_hits=False)
    blocked = np.zeros(len(idx), bool)
    for L_, r_ in zip(locs, ri):
        if np.linalg.norm(L_ - o[r_]) < dist[r_] - 1.0: blocked[r_] = True
    return int(len(idx)), int((~blocked).sum())
rng = np.random.default_rng(0)
def samp(M, n=3000):
    t = to_tm(M); return t.sample(n, seed=0) if hasattr(t, "sample") else t.vertices
vis = {}
for a in [-0.17, 0.15, 0.45]:
    pts = trimesh.transform_points(to_tm(jaw_loop_part).sample(3000), rot_about_pivot(a))
    vis[f"moving loop @{a:+.2f}"] = visible(pts)
vis["fixed loop"] = visible(to_tm(fixed_loop_part).sample(3000))
vis["handle"] = visible(to_tm(H).sample(6000))
# a fingertip poking 25 mm out the front of each loop (digit is in the loop)
for nm, c in [("fingertip ahead of fixed loop", F_C), ("fingertip ahead of moving loop (open)",
                                                       (rot_about_pivot(OPEN48 - MID) @ np.append(M_MID, 1))[:3])]:
    tipc = trimesh.creation.capsule(height=25, radius=9).apply_translation([0, 0, 0])
    tipc.apply_translation([c[0], c[1], c[2] - LOOP_T / 2 - 12.5])
    vis[nm] = visible(tipc.sample(1500))
report["wrist_cam_points_in_image_(in_view, unoccluded)"] = vis
report["tip_mm"] = np.round(tip, 1).tolist()

# ---------- export ----------
os.makedirs(OUT, exist_ok=True)
H_tm = to_tm(H); H_tm.export(f"{OUT}/soumi_handle.stl")
_, _, Mj = parts(0.0)
Mj_mm = Mj.copy(); Mj_mm[:3, 3] *= 1000
Mjaw = Mj_mm @ T([0, 0, 18.9], [1, 0, 0, 0])
JT = to_tm(jaw_trig); JT_local = JT.copy(); JT_local.apply_transform(np.linalg.inv(Mjaw))
JT_local.export(f"{OUT}/soumi_moving_jaw_loop.stl")
FL = to_tm(follower_loop); FL.export(f"{OUT}/soumi_fixed_jaw_loop.stl")   # gripper frame, mm
def colored(m, rgba):
    m = m.copy(); m.visual.face_colors = rgba; return m
scene = trimesh.Scene()
scene.add_geometry(colored(FL, [255, 209, 31, 255]), node_name="follower_with_loop")
scene.add_geometry(colored(ref["mount"], [255, 209, 31, 255]), node_name="cam_mount")
scene.add_geometry(colored(ref["camera"], [30, 30, 30, 255]), node_name="camera")
scene.add_geometry(colored(ref["servo"], [40, 40, 40, 255]), node_name="servo")
scene.add_geometry(colored(H_tm, [70, 130, 220, 255]), node_name="handle")
scene.add_geometry(colored(JT, [230, 90, 60, 255]), node_name="moving_jaw_with_loop")
scene.export(f"{OUT}/soumi_assembly_preview.glb")
coupon = cyl_z(15.0, 1.2, 7.0, *AX) + cyl_z(9.85, 0.0, 2.0, *AX)
for hx, hy in HORN_HOLES:
    coupon = coupon - cyl_z(PILOT_D / 2, -1, 8.0, hx, hy, 32)
to_tm(coupon).export(f"{OUT}/soumi_fit_test_horn_plate.stl")
json.dump(report, open(f"{OUT}/check_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
