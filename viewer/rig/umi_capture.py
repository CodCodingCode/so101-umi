"""SO-UMI recorder that runs ON the rig PC (dzkchen): wrist (UMI) + overhead cameras on one clock.

Both cameras are opened here, every frame gets a timestamp from the same clock the moment it is
read, and recordings go straight to this machine's disk. Nothing about the data depends on the
network; only the small live preview crosses it.

    python umi_capture.py [--overhead 1 --umi 3 --robot-wrist 2] [--port 8106] [--out C:\\robot\\umi_data]

HTTP (127.0.0.1 only; reach it from the Mac with `ssh -L 8106:127.0.0.1:8106 dzkchen`):
    GET  /cam/<overhead|umi>.mjpg   preview for watching only; ?w=<px>&q=<jpeg quality>&fps=<rate>
                                    (default 320 px, q60, 10 fps: sized for the slow link to the Mac)
    GET  /api/status                cameras (measured fps, last frame age), recording state
    POST /record/start?name=<base>  start writing <out>/<base>/<cam>/frames/*.jpg + frames.csv
    POST /record/stop               stop; returns frame counts / fps per camera
    GET  /recording/<name>/<file>   the SMALL files of a recording: meta.json, <cam>/detections.jsonl (tags and
                                    moving finger per frame, detected live while recording), <cam>/frames.csv,
                                    <cam>/preview.bin + preview.csv (320 px JPEGs for the browser replay)
    GET  /api/recordings            list of recordings on disk

Recording layout: <out>/<base>/{umi,overhead}/frames/000001.jpg ..., frames.csv
(index,t,t_unix: t = seconds since the recording started, same clock for both cameras),
and <out>/<base>/meta.json.

monitor.py holds the rig cameras: stop it before starting this (and restart it afterwards).
This script never touches the arms.
"""

import argparse
import json
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

import numpy as np

SAFE = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
LIVE = {"bank": None}          # finger template bank (temps, centres, px_to_mm), uploaded from the Mac


def make_detector(fast=True):
    """fast=True: OpenCV's 3 threshold passes (enough for 25-45 px cube tags and tag 13; ~4x cheaper, keeps up
    with 30 fps live). fast=False: umi/convert.py's 13-pass tuning for small, slightly blurred tags."""
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if not fast:
        p.adaptiveThreshWinSizeMax = 53
        p.adaptiveThreshWinSizeStep = 4
    p.minMarkerPerimeterRate = 0.01
    p.polygonalApproxAccuracyRate = 0.08
    p.perspectiveRemovePixelPerCell = 8
    p.perspectiveRemoveIgnoredMarginPerCell = 0.2
    p.minDistanceToBorder = 1
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p)


def load_bank(path):
    try:
        z = np.load(path, allow_pickle=True)
        temps = [np.ascontiguousarray(t, dtype=np.uint8) for t in z["temps"]]
        return {"temps": temps, "centres": np.array(z["centres"]), "px_to_mm": [float(v) for v in z["px_to_mm"]]}
    except Exception as e:
        print(f"finger template bank not loaded ({e}); finger tracking uses the tag only", flush=True)
        return None


def finger_x(gray, bank):
    """Moving finger's x (px) by template matching in the strip where it lives, or None."""
    H, W = 70, 90
    cxs, cys = bank["centres"][:, 0], bank["centres"][:, 1]
    y_lo, x_lo = int(max(0, cys.min() - H)), int(max(0, cxs.min() - 2 * W))
    strip = cv2.GaussianBlur(gray[y_lo:, x_lo:], (3, 3), 0)
    small = cv2.resize(strip, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)   # half size: 4x cheaper
    best = (-1.0, 0.0)
    for tp in bank.setdefault("temps_half", [cv2.resize(t, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                                             for t in bank["temps"]]):
        if small.shape[0] < tp.shape[0] or small.shape[1] < tp.shape[1]:
            continue
        r = cv2.matchTemplate(small, tp, cv2.TM_CCOEFF_NORMED)
        _, v, loc_v, loc = cv2.minMaxLoc(r)
        if v > best[0]:
            best = (v, 2 * (loc[0] + tp.shape[1] / 2) + x_lo)
    return best if best[0] > 0.5 else None
CLOCK0_PERF = time.perf_counter()
CLOCK0_UNIX = time.time()


def now():
    """One monotonic clock for every camera, expressed as unix seconds."""
    return CLOCK0_UNIX + (time.perf_counter() - CLOCK0_PERF)


class _NullQueue:
    def put(self, *a, **k): pass
    def put_nowait(self, *a, **k): pass
    def empty(self): return True


class Cam:
    def __init__(self, name, index, width, height, fps, quality):
        self.name, self.index, self.quality = name, index, quality
        self.mode = (width, height, fps)
        self.cap = self._open()
        self.reopens = 0
        self.lock = threading.Lock()
        self.latest = None          # (t_unix, frame)
        self.times = []             # recent read times, for the measured fps
        self.error = None
        self.rec = None             # (dir, csv file, t0, queue) while recording
        self.count = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise OSError(f"cannot open camera {self.name} (index {self.index}); is monitor.py still running?")
        w, h, fps = self.mode
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # fast compressed mode
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap

    def _reopen(self, why):
        """Watchdog: DirectShow cameras sometimes stall after a USB hiccup (no error, ~1 fps, black frames).
        Reopening the device brings them back."""
        print(f"{self.name}: {why}; reopening the camera", flush=True)
        try:
            self.cap.release()
        except Exception:
            pass
        time.sleep(1.0)
        try:
            self.cap = self._open()
            self.reopens += 1
        except OSError as e:
            self.error = str(e)
            time.sleep(2.0)

    def _loop(self):
        slow_since, fail_since = None, None
        while True:
            ok, f = self.cap.read()
            t = now()
            if not ok:
                self.error = "read failed"
                fail_since = fail_since or t
                if t - fail_since > 3.0:
                    self._reopen("no frames for 3 s")
                    fail_since = None
                time.sleep(0.05)
                continue
            fail_since = None
            last = self.latest[0] if self.latest else None
            if last is not None and t - last > 0.25:          # a 30 fps camera delivering < 4 fps
                slow_since = slow_since or t
                if t - slow_since > 5.0:
                    self._reopen(f"stalled ({t - last:.1f} s per frame for 5 s)")
                    slow_since = None
            else:
                slow_since = None
            self.error = None
            with self.lock:
                self.latest = (t, f)
                self.times = [x for x in self.times if t - x < 2.0] + [t]
                if self.rec is not None:
                    self.count += 1
                    self.rec[3].put((self.count, t, f))
                    try:
                        self.rec[4].put_nowait((self.count, t, f))
                    except queue.Full:
                        pass                      # detector far behind: that frame just has no live detection

    def fps(self):
        with self.lock:
            ts = self.times
            return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else 0.0

    def start(self, root: Path, t0):
        d = root / self.name
        (d / "frames").mkdir(parents=True)
        csv = open(d / "frames.csv", "w")
        csv.write("index,t,t_unix\n")
        q = queue.Queue(maxsize=600)
        dq = queue.Queue(maxsize=1200)            # live detection: runs alongside, drains after stop
        threading.Thread(target=self._writer, args=(d, csv, t0, q), daemon=True).start()
        if self.name in ("overhead", "umi"):         # the robot's wrist view needs no live tracking
            self.det_thread = threading.Thread(target=self._detector, args=(d, t0, dq), daemon=True)
            self.det_thread.start()
        else:
            dq = _NullQueue()
        with self.lock:
            self.count = 0
            self.rec = (d, csv, t0, q, dq)

    def _detector(self, d, t0, dq):
        """Tags (all cameras) + moving finger (umi) for every frame, written as one JSON line per frame, and a
        small preview pack (320 px JPEGs) for the browser replay. A few KB per second instead of ~5 MB."""
        det = make_detector(fast=True)
        every = {"overhead": 3, "umi": 6}.get(self.name, 0)   # preview: 10 fps overhead, 5 fps wrist
        self.det_ms = []
        out = open(d / "detections.jsonl", "w")
        pv_bin = open(d / "preview.bin", "wb") if every else None
        pv_csv = open(d / "preview.csv", "w") if every else None
        if pv_csv:
            pv_csv.write("index,t,offset,bytes\n")
        while True:
            item = dq.get()
            if item is None:
                break
            i, t, f = item
            tic = time.perf_counter()
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = det.detectMarkers(g)
            tags = {}
            if ids is not None:
                for c, k in zip(corners, ids.ravel()):
                    q4 = c.reshape(4, 2)
                    if np.linalg.norm(q4 - np.roll(q4, 1, 0), axis=1).mean() >= 6:   # drop specks
                        tags[str(int(k))] = [round(float(v), 2) for v in q4.ravel()]
            rec = {"i": i, "t": round(t - t0, 4), "tags": tags}
            if self.name == "umi" and LIVE["bank"] is not None:
                fx = finger_x(g, LIVE["bank"])
                if fx is not None:
                    rec["finger"] = [round(float(fx[1]), 2), round(float(fx[0]), 3)]
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self.det_ms.append((time.perf_counter() - tic) * 1000)
            if every and i % every == 1:
                h = int(f.shape[0] * 320 / f.shape[1])
                ok, buf = cv2.imencode(".jpg", cv2.resize(f, (320, h), interpolation=cv2.INTER_AREA),
                                       [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ok:
                    off = pv_bin.tell()
                    pv_bin.write(buf.tobytes())
                    pv_csv.write(f"{i},{t - t0:.4f},{off},{len(buf)}\n")
        out.close()
        if pv_bin:
            pv_bin.close()
            pv_csv.close()

    def _writer(self, d, csv, t0, q):
        while True:
            item = q.get()
            if item is None:
                break
            i, t, f = item
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if ok:
                (d / "frames" / f"{i:06d}.jpg").write_bytes(buf.tobytes())
                csv.write(f"{i},{t - t0:.4f},{t:.4f}\n")
        csv.close()

    def detach(self):
        """Stop recording new frames (instant); finalize() then flushes the writer."""
        with self.lock:
            rec, self.rec = self.rec, None
            n = self.count
        return rec, n

    @staticmethod
    def finalize(rec, n):
        if rec is None:
            return None
        rec[3].put(None)
        rec[4].put(None)
        while not rec[3].empty() or not rec[4].empty():
            time.sleep(0.05)
        time.sleep(0.3)
        rows = (rec[0] / "frames.csv").read_text().splitlines()[1:]
        ts = [float(r.split(",")[1]) for r in rows]
        return {"frames": len(ts), "read": n,
                "fps": round((len(ts) - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 2 else 0.0,
                "first_s": round(ts[0], 3) if ts else None, "last_s": round(ts[-1], 3) if ts else None,
                "max_gap_ms": round(max((b - a) * 1000 for a, b in zip(ts, ts[1:])), 1) if len(ts) > 2 else None}


class Rig:
    def __init__(self, cams, out: Path):
        self.cams, self.out = cams, out
        self.current = None
        self.lock = threading.Lock()

    def start(self, base):
        with self.lock:
            if self.current:
                raise RuntimeError("already recording")
            root = self.out / base
            root.mkdir(parents=True, exist_ok=False)
            t0 = now()
            for c in self.cams.values():
                c.start(root, t0)
            self.current = {"name": base, "dir": str(root), "t0_unix": t0}
            return dict(self.current)

    def stop(self):
        with self.lock:
            cur, self.current = self.current, None
        if not cur:
            return {"error": "not recording"}
        detached = {n: c.detach() for n, c in self.cams.items()}   # all cameras stop at the same instant
        cur["duration_s"] = round(now() - cur["t0_unix"], 2)
        cur["cameras"] = {n: {"index": self.cams[n].index, **(Cam.finalize(*detached[n]) or {})} for n in self.cams}
        for n, c in self.cams.items():
            ms = getattr(c, "det_ms", None)
            if ms:
                cur["cameras"][n]["live_detect_ms_mean"] = round(float(np.mean(ms)), 1)
        (Path(cur["dir"]) / "meta.json").write_text(json.dumps(cur, indent=2))
        return cur


def make_handler(rig: Rig, preview_fps, preview_w):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            p = urlparse(self.path).path
            if p == "/api/status":
                return self._json(200, {
                    "t_unix": now(), "recording": rig.current,
                    "cameras": {n: {"index": c.index, "fps": round(c.fps(), 1), "error": c.error, "reopens": c.reopens,
                                    "age_ms": round((now() - c.latest[0]) * 1000) if c.latest else None,
                                    "frames": c.count if rig.current else 0}
                                for n, c in rig.cams.items()}})
            m = re.fullmatch(r"/recording/([A-Za-z0-9._-]+)/(meta\.json|(?:overhead|umi|robot_wrist)/"
                             r"(?:detections\.jsonl|frames\.csv|preview\.bin|preview\.csv))", p)
            if m:
                f = rig.out / m.group(1) / m.group(2)
                if not f.exists():
                    return self._json(404, {"error": f"{m.group(2)} not found (recorded before live detection?)"})
                b = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            if p == "/api/recordings":
                return self._json(200, sorted(d.name for d in rig.out.iterdir() if d.is_dir()) if rig.out.exists() else [])
            m = re.fullmatch(r"/cam/(\w+)\.mjpg", p)
            if m and m.group(1) in rig.cams:
                cam = rig.cams[m.group(1)]
                q = parse_qs(urlparse(self.path).query)
                pw = max(160, min(640, int(q.get("w", [preview_w])[0])))       # preview width (px)
                pq = max(30, min(95, int(q.get("q", [60])[0])))                # preview JPEG quality
                pf = max(1.0, min(30.0, float(q.get("fps", [preview_fps])[0])))  # preview rate
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    last = None
                    while True:
                        with cam.lock:
                            lt = cam.latest
                        if lt and lt[0] != last:
                            last = lt[0]
                            f = lt[1]
                            h = int(f.shape[0] * pw / f.shape[1])
                            img = f if pw == f.shape[1] else cv2.resize(f, (pw, h), interpolation=cv2.INTER_AREA)
                            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, pq])
                            b = buf.tobytes()
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                        time.sleep(1.0 / pf)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return
            self.send_error(404)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/record/start":
                base = parse_qs(u.query).get("name", [""])[0]
                if not SAFE.match(base):
                    return self._json(400, {"error": "bad name"})
                try:
                    return self._json(200, rig.start(base))
                except Exception as e:
                    return self._json(409, {"error": str(e)})
            if u.path == "/record/stop":
                return self._json(200, rig.stop())
            self.send_error(404)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overhead", type=int, default=1, help="DirectShow index of the overhead camera")
    ap.add_argument("--umi", type=int, default=3, help="DirectShow index of the UMI (handheld wrist) camera")
    ap.add_argument("--robot-wrist", type=int, default=2,
                    help="DirectShow index of the robot arm's wrist camera (rig.json right_wrist = 2), recorded as "
                         "robot_wrist so a robot run can be compared with the demo's wrist view; -1 = don't open it")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality of recorded frames")
    ap.add_argument("--port", type=int, default=8106)
    ap.add_argument("--preview-fps", type=float, default=10.0)
    ap.add_argument("--preview-width", type=int, default=320)
    ap.add_argument("--out", default=r"C:\robot\umi_data")
    ap.add_argument("--finger-bank", default=str(Path(__file__).with_name("finger_templates.npz")),
                    help="moving-finger templates (calibration/finger_templates.npz from the Mac) for live gripper tracking")
    a = ap.parse_args()
    if Path(a.finger_bank).exists():
        LIVE["bank"] = load_bank(a.finger_bank)
    cams = {"overhead": Cam("overhead", a.overhead, a.width, a.height, a.fps, a.quality),
            "umi": Cam("umi", a.umi, a.width, a.height, a.fps, a.quality)}
    if a.robot_wrist >= 0:
        try:
            cams["robot_wrist"] = Cam("robot_wrist", a.robot_wrist, a.width, a.height, a.fps, a.quality)
        except OSError as e:                     # a missing wrist camera must not stop UMI recording
            print(f"robot_wrist: {e}; recording without it", flush=True)
    time.sleep(2.0)
    for n, c in cams.items():
        print(f"{n}: index {c.index}, {c.fps():.1f} fps, error {c.error}", flush=True)
    rig = Rig(cams, Path(a.out))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(rig, a.preview_fps, a.preview_width))
    srv.daemon_threads = True
    print(f"umi_capture: http://127.0.0.1:{a.port}  recording to {a.out}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
