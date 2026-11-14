#!/usr/bin/env python3
"""Generate a printable checkerboard PDF at exact physical size (no dependencies).

Default: 10x7 squares (9x6 inner corners), 25 mm squares, landscape A4 + Letter.
Print at 100% / "Actual size" (NOT "fit to page"), then verify the 100 mm
reference bar with a ruler.

Usage: python3 calibration/make_checkerboard.py [--cols 10 --rows 7 --square-mm 25]
"""

import argparse
from pathlib import Path

MM = 72 / 25.4  # PDF points per mm
PAGES = {"a4": (297, 210), "letter": (279.4, 215.9)}  # landscape, mm


def build_pdf(page_w, page_h, cols, rows, sq, label):
    board_w, board_h = cols * sq, rows * sq
    x0 = (page_w - board_w) / 2
    y0 = (page_h - board_h) / 2 + 5  # leave room for the reference bar below
    ops = ["0 g"]
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                ops.append(f"{(x0 + c * sq) * MM:.3f} {(y0 + r * sq) * MM:.3f} "
                           f"{sq * MM:.3f} {sq * MM:.3f} re f")
    # 100 mm reference bar with end ticks, below the board.
    bar_y = y0 - 10
    ops.append(f"{x0 * MM:.3f} {bar_y * MM:.3f} {100 * MM:.3f} {0.8 * MM:.3f} re f")
    for x in (x0, x0 + 100):
        ops.append(f"{x * MM:.3f} {(bar_y - 2) * MM:.3f} {0.4 * MM:.3f} {4.8 * MM:.3f} re f")
    ops.append(f"BT /F1 9 Tf {(x0 + 103) * MM:.3f} {(bar_y - 0.5) * MM:.3f} Td "
               f"({label}) Tj ET")
    content = "\n".join(ops).encode()

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w * MM:.3f} {page_h * MM:.3f}] "
         f"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>").encode(),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=10, help="squares across")
    ap.add_argument("--rows", type=int, default=7, help="squares down")
    ap.add_argument("--square-mm", type=float, default=25.0)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    for name, (w, h) in PAGES.items():
        if args.cols * args.square_mm > w - 20 or args.rows * args.square_mm > h - 30:
            print(f"skip {name}: board too large for page")
            continue
        label = (f"100 mm  |  {args.cols}x{args.rows} squares "
                 f"({args.cols - 1}x{args.rows - 1} inner corners), {args.square_mm:g} mm")
        path = out_dir / f"checkerboard_{args.cols - 1}x{args.rows - 1}_{args.square_mm:g}mm_{name}.pdf"
        path.write_bytes(build_pdf(w, h, args.cols, args.rows, args.square_mm, label))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
