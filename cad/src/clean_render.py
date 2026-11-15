"""One-colour renders of the whole SO-UMI handheld via MuJoCo's offscreen renderer.

usage: python clean_render.py CAD_DIR
writes CAD_DIR/soumi_render_clean.png (no tags, one view) and
       CAD_DIR/soumi_render_tags.png  (cube tags 20-23 + finger tags 0/1 on, two views)
needs: trimesh numpy scipy mujoco pillow opencv-contrib-python
"""
import sys, os, tempfile, subprocess
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
CAD = sys.argv[1]; TMP = tempfile.mkdtemp(prefix="soumi_clean_"); ANGLE = "0.30"   # jaw opening, rad
def run(*a): subprocess.run([PY, *a], check=True)
def crop(p):
    im = Image.open(p); m = (np.asarray(im).astype(int) < 250).any(2); ys, xs = np.where(m)
    assert not (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()), f"{p}: part clipped, raise the distance"
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
def sheet(ims, out):
    H = max(i.height for i in ims); pad = int(0.08 * H)
    s = Image.new("RGB", (sum(i.width for i in ims) + pad * (len(ims) + 1), H + 2 * pad), "white"); x = pad
    for i in ims: s.paste(i, (x, pad + (H - i.height) // 2)); x += i.width + pad
    s.save(out); print("saved", out)
# parts in the "as held" frame (x forward, y left, z up), metres
run(os.path.join(HERE, "_clean_prep.py"), CAD, TMP, ANGLE)
# 1) no tags: front-right three-quarter view
run(os.path.join(HERE, "_clean_mj.py"), TMP, f"{TMP}/clean.png", "2400", "1800", "125", "-20", "0.55")
sheet([crop(f"{TMP}/clean.png")], os.path.join(CAD, "soumi_render_clean.png"))
# 2) with tags: behind-right (IDs 20, 22, 23) and front-left (IDs 21, 22, finger tags)
run(os.path.join(HERE, "_clean_tags.py"), CAD, TMP, ANGLE)
run(os.path.join(HERE, "_clean_mj.py"), TMP, f"{TMP}/a.png", "2400", "1800", "45", "-32", "0.56")
run(os.path.join(HERE, "_clean_mj.py"), TMP, f"{TMP}/b.png", "2400", "1800", "-125", "-32", "0.58")
sheet([crop(f"{TMP}/a.png"), crop(f"{TMP}/b.png")], os.path.join(CAD, "soumi_render_tags.png"))
