#!/usr/bin/env python3
"""Serve the camera viewer and save captures/snapshots into <repo>/data/.

Usage: python3 viewer/server.py [--port 8000]
"""

import argparse
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VIEWER_DIR = Path(__file__).resolve().parent
DATA_DIR = VIEWER_DIR.parent / "data"
ALLOWED_EXT = {".png", ".jpg", ".webm", ".mp4"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,200}$")


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/save":
            return self.send_error(404)
        name = parse_qs(url.query).get("name", [""])[0]
        if not SAFE_NAME.match(name) or Path(name).suffix.lower() not in ALLOWED_EXT:
            return self.send_error(400, "bad filename")
        length = int(self.headers.get("Content-Length", 0))
        DATA_DIR.mkdir(exist_ok=True)
        dest = DATA_DIR / name
        with open(dest, "wb") as f:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(str(dest.relative_to(VIEWER_DIR.parent)).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    handler = partial(Handler, directory=str(VIEWER_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Viewer: http://localhost:{args.port}  (saving to {DATA_DIR})")
    server.serve_forever()


if __name__ == "__main__":
    main()
