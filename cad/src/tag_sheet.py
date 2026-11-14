"""Printable ArUco sheet for SO-UMI (US Letter; content also fits A4). Print at 100% / Actual size.

Page 1: finger tags (8 mm, IDs 0/1, 4 copies each: handheld + robot + spares),
        cube tags (25 mm tag on a 30 mm sticker, IDs 20-23, with which face + which way up),
        a 100 mm scale bar to check the printer didn't rescale.
Page 2: table tag (160 mm, ID 13).
Markers are drawn as vector squares, so edges are exact at any printer resolution.
Usage: python tag_sheet.py OUT.pdf [OUT_preview.png]
"""
import sys
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

W, H = 215.9, 279.4                     # US Letter, mm
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

def bits(tag_id):
    """6x6 cell grid (1-cell black border + 4x4 payload), 1 = white."""
    return (cv2.aruco.generateImageMarker(DICT, tag_id, 6, borderBits=1) > 127).astype(int)

def marker(ax, tag_id, x, y, size, cut=True, label=None, note=None, quiet=None):
    """Draw marker with its outer black square at (x, y) (top-left, mm) and side `size` mm.
    `quiet` = white border kept when cutting (default one cell)."""
    c = size / 6.0
    q = c if quiet is None else quiet
    if cut:                                 # light dashed cut line around the quiet zone
        ax.add_patch(Rectangle((x - q, y - q), size + 2 * q, size + 2 * q, fill=False,
                               ls=(0, (1.5, 1.5)), lw=0.3, ec="#9a9a9a"))
    b = bits(tag_id)
    ax.add_patch(Rectangle((x, y), size, size, fc="black", ec="none"))
    for r in range(6):
        for k in range(6):
            if b[r, k]:
                ax.add_patch(Rectangle((x + k * c, y + r * c), c, c, fc="white", ec="none", lw=0))
    if label:
        ax.text(x + size / 2, y + size + q + 2.2, label, ha="center", va="top", fontsize=6.5)
    if note:
        ax.text(x + size / 2, y + size + q + 5.6, note, ha="center", va="top", fontsize=5.5, color="#444")

def page(pdf, draw):
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.axis("off")
    draw(ax)
    pdf.savefig(fig); return fig

def scale_bar(ax, x, y):
    ax.add_patch(Rectangle((x, y), 100, 1.2, fc="black", ec="none"))
    for i in range(11):
        ax.plot([x + 10 * i, x + 10 * i], [y - (2.5 if i % 5 == 0 else 1.5), y], color="black", lw=0.5)
    ax.text(x, y + 3.5, "This bar must measure exactly 100 mm. If not, reprint at 100% / Actual size (no 'fit to page').",
            ha="left", va="top", fontsize=6.5)

def page1(ax):
    ax.text(15, 14, "SO-UMI ArUco tags  (DICT_4X4_50)", fontsize=14, weight="bold", va="top")
    ax.text(15, 22, "Print at 100% / Actual size on a laser printer, matte paper. Cut on the dashed lines: keep the white border; "
                    "ArUco needs it to detect the tag.", fontsize=7, va="top")
    scale_bar(ax, 15, 33)
    # finger tags
    ax.text(15, 50, "1. Finger tags: 8 mm (optional; the servo already measures width)", fontsize=9.5, weight="bold", va="top")
    ax.text(15, 56, "Top edge of each finger blade (the face the wrist camera sees), ~20 mm back from the tip. "
                    "Put the same pair on the robot's jaws so both look identical.", fontsize=6.5, va="top")
    x0 = 18
    for i, (tid, who) in enumerate([(0, "fixed jaw"), (0, "fixed jaw"), (0, "spare"), (0, "spare"),
                                    (1, "moving jaw"), (1, "moving jaw"), (1, "spare"), (1, "spare")]):
        marker(ax, tid, x0 + i * 23, 64, 8.0, label=f"ID {tid}", note=who)
    ax.text(18, 86, "Handheld: one ID 0 + one ID 1.  Robot: one ID 0 + one ID 1.  ID 0 = fixed jaw, ID 1 = moving jaw.\n"
                    "The finger face is ~10-11 mm wide: if the sticker overhangs, trim the white border to ~0.7 mm.", fontsize=6.5, va="top")
    # cube tags
    ax.text(15, 98, "2. Cube tags: 25 mm tag on a 30 mm sticker, for the 30 mm pockets on the back cube (required)",
            fontsize=9.5, weight="bold", va="top")
    ax.text(15, 104, "Cut exactly on the dashed line (30 mm) so the sticker drops into the pocket. \nDirections are as you hold it "
                     "(camera on top, jaws pointing away). Red arrow = the tag's top edge.", fontsize=6.5, va="top")
    cube = [(20, "ID 20: right face", "(moving-jaw side), top edge toward the jaws"),
            (21, "ID 21: left face", "(fixed-jaw side), top edge toward the jaws"),
            (22, "ID 22: top face", "top edge toward the jaws"),
            (23, "ID 23: back face", "top edge up (camera side)")]
    for i, (tid, lab, note) in enumerate(cube):
        x = 22 + i * 47
        marker(ax, tid, x + 2.5, 116.5, 25.0, label=lab, note=note, quiet=2.5)
        ax.annotate("", xy=(x + 15, 109.2), xytext=(x + 15, 113.0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#c0392b", mutation_scale=6))
    ax.text(15, 166, "Glue flat (spray adhesive or glue stick): a bubble or a curled corner biases the pose estimate. "
                     "Tag size for the tracking code: 25 mm.", fontsize=6.5, va="top")
    # spare cube set
    ax.text(15, 176, "Spare cube set (same IDs, same placement):", fontsize=7.5, va="top")
    for i, (tid, lab, note) in enumerate(cube):
        marker(ax, tid, 22 + i * 47 + 2.5, 186.5, 25.0, label=f"ID {tid}", quiet=2.5)
    ax.text(15, 236, "Table tag (ID 13, 160 mm) is on page 2.  IDs 0-19 are reserved by UMI's own config; "
                     "don't reuse them for anything else.", fontsize=6.5, va="top")
    ax.text(15, 266, "Generated by so-umi/cad/src/tag_sheet.py", fontsize=5.5, color="#777", va="top")

def page2(ax):
    ax.text(15, 14, "Table tag: ID 13, 160 mm (DICT_4X4_50)", fontsize=14, weight="bold", va="top")
    ax.text(15, 22, "Tape flat to the table, inside the overhead camera's view, and don't move it during a session. "
                    "It defines the table/world frame (the same role as UMI's ID 13).", fontsize=7, va="top")
    scale_bar(ax, 15, 34)
    marker(ax, 13, (W - 160) / 2, 58, 160.0, cut=False, label="ID 13  (160 mm black square; leave the paper uncut, its white margin is the border)")
    ax.text(15, 266, "Generated by so-umi/cad/src/tag_sheet.py", fontsize=5.5, color="#777", va="top")

if __name__ == "__main__":
    out = sys.argv[1]
    with PdfPages(out) as pdf:
        f1 = page(pdf, page1); f2 = page(pdf, page2)
    if len(sys.argv) > 2:
        f1.savefig(sys.argv[2], dpi=150)
    # high-res rasters for self-check
    f1.savefig(out.replace(".pdf", "_p1_600dpi.png"), dpi=600); f2.savefig(out.replace(".pdf", "_p2_300dpi.png"), dpi=300)
    print("saved", out)
