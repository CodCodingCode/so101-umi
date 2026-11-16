#!/usr/bin/env python3
"""Serve the camera viewer, save captures/snapshots into <repo>/data/, and relay
the rig's overhead camera so the page can show and record it.

Usage:
  python3 viewer/server.py                       # UMI camera only (+ overhead if the tunnel is already up)
  python3 viewer/server.py --tunnel              # also open the SSH tunnel to the rig's monitor.py
  python3 viewer/server.py --rig                 # record on the rig PC (viewer/rig/umi_capture.py), use studio.html

The rig PC's ssh host, paths and ports come from .env (see .env.example); --tunnel/--rig HOST overrides the host.
The overhead feed comes from monitor.py on the rig PC (it holds the rig cameras),
reached through `ssh -L 8104:127.0.0.1:8104 <host>`. It is relayed at
/overhead.mjpg for the page to display.

Recording the overhead happens HERE, not in the browser: POST /record/start?name=<base>
saves every JPEG exactly as it arrives, with its arrival time, into data/<base>/
(frames/000001.jpg ..., frames.csv), until POST /record/stop. The stream is ~15 fps with
uneven spacing (rig + SSH tunnel), so re-recording it in the browser at a fixed rate made
playback speed wobble; true per-frame timestamps don't.
"""

import argparse
import atexit
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import sys

import umi_jobs as UJ   # noqa: E402  (viewer/umi_jobs.py: Generate UMI trajectory / Run real)

VIEWER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VIEWER_DIR.parent / "umi"))
import rig_link as L    # noqa: E402  (rig settings from .env)
DATA_DIR = VIEWER_DIR.parent / "data"
ALLOWED_EXT = {".png", ".jpg", ".webm", ".mp4"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
RELAYS = {"/overhead.mjpg": "overhead"}  # path -> rig camera name


def mjpeg_parts(resp):
    """Yield JPEG bytes from a multipart/x-mixed-replace response, using each part's
    Content-Length (falls back to scanning for the JPEG end marker)."""
    buf = b""
    while True:
        while b"\r\n\r\n" not in buf:
            chunk = resp.read1(65536)
            if not chunk:
                return
            buf += chunk
        head, _, buf = buf.partition(b"\r\n\r\n")
        m = re.search(rb"Content-Length:\s*(\d+)", head, re.I)
        if m:
            n = int(m.group(1))
            while len(buf) < n:
                chunk = resp.read1(65536)
                if not chunk:
                    return
                buf += chunk
            jpg, buf = buf[:n], buf[n:]
        else:
            while (end := buf.find(b"\xff\xd9")) < 0:
                chunk = resp.read1(65536)
                if not chunk:
                    return
                buf += chunk
            jpg, buf = buf[:end + 2], buf[end + 2:]
        start = jpg.find(b"\xff\xd8")
        if start >= 0:
            yield jpg[start:]


class OverheadRecorder:
    """Keeps one connection to the overhead feed open all the time (so recording starts on the
    very next frame, not after a ~2 s connect through the tunnel) and, while recording, writes
    each JPEG with its arrival time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.url = None
        self.out = None          # Path while recording
        self.csv = None
        self.t0 = 0.0
        self.info = {}
        self.connected = False
        self.latest = None       # (arrival time, jpeg bytes) of the newest frame, for the browser relay
        self.cond = threading.Condition()

    def ensure_reader(self, url):
        with self.lock:
            if self.url:
                return
            self.url = url
        threading.Thread(target=self._read_forever, daemon=True).start()

    def _read_forever(self):
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=5) as up:
                    self.connected = True
                    for jpg in mjpeg_parts(up):
                        now = time.time()
                        with self.cond:
                            self.latest = (now, jpg)
                            self.cond.notify_all()
                        with self.lock:
                            if self.out is not None:
                                i = self.info["frames"] + 1
                                (self.out / "frames" / f"{i:06d}.jpg").write_bytes(jpg)
                                self.csv.write(f"{i},{now - self.t0:.4f},{now:.4f},{len(jpg)}\n")
                                self.info["frames"] = i
            except Exception as e:
                with self.lock:
                    if self.out is not None:
                        self.info["error"] = f"feed dropped: {e}"
            self.connected = False
            time.sleep(1.0)  # reconnect

    def frames(self, timeout=5.0):
        """Yield each new frame (jpeg bytes) as the reader receives it."""
        last = None
        while True:
            with self.cond:
                if not self.cond.wait_for(lambda: self.latest is not None and self.latest[0] != last, timeout):
                    return
                last, jpg = self.latest
            yield jpg

    def start(self, url, base):
        self.ensure_reader(url)
        with self.lock:
            if self.out is not None:
                raise RuntimeError("already recording")
            if not self.connected:
                raise RuntimeError("overhead feed not connected (tunnel / monitor.py down?)")
            out = DATA_DIR / base
            (out / "frames").mkdir(parents=True, exist_ok=False)
            self.csv = open(out / "frames.csv", "w")
            self.csv.write("index,t,t_unix,bytes\n")
            self.t0 = time.time()
            self.info = {"dir": str(out.relative_to(VIEWER_DIR.parent)), "frames": 0, "t0_unix": self.t0, "error": None}
            self.out = out
            return dict(self.info)

    def stop(self):
        with self.lock:
            out, info = self.out, dict(self.info)
            self.out = None
            if self.csv:
                self.csv.close()
                self.csv = None
        if out is None:
            return {"error": "not recording"}
        if info.get("frames", 0) > 1:
            ts = [float(l.split(",")[1]) for l in (out / "frames.csv").read_text().splitlines()[1:]]
            info["first_frame_s"] = round(ts[0], 3)
            info["duration_s"] = round(ts[-1] - ts[0], 2)
            info["fps"] = round((len(ts) - 1) / max(ts[-1] - ts[0], 1e-6), 1)
        (out / "meta.json").write_text(__import__("json").dumps(info, indent=2))
        return info


RECORDER = OverheadRecorder()


# ---------------------------------------------------------------- rig recorder (viewer/rig/umi_capture.py on the rig PC)
RIG = {"host": None, "port": L.CAPTURE_PORT}
FETCH = {}   # name -> {"state", "bytes", "total", "error", "dir"}


def rig_url(path):
    return f"http://127.0.0.1:{RIG['port']}{path}"


LIVE_FILES = ("meta.json", "overhead/detections.jsonl", "overhead/frames.csv", "overhead/preview.bin",
              "overhead/preview.csv", "umi/detections.jsonl", "umi/frames.csv", "umi/preview.bin", "umi/preview.csv")


def fetch_live(name):
    """Get just the rig's live tracking + preview for a recording (a few hundred KB over the rig's HTTP link,
    ~1-2 s) into data/<name>-rig/. True if it had live tracking; False for older recordings (copy frames)."""
    dest = DATA_DIR / f"{name}-rig"
    got = {}
    for rel in LIVE_FILES:
        try:
            with urllib.request.urlopen(rig_url(f"/recording/{name}/{rel}"), timeout=60) as up:
                got[rel] = up.read()
        except Exception:
            if rel.endswith(("detections.jsonl", "frames.csv", "meta.json")):
                return False
    for rel, b in got.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_bytes(b)
    return True


def fetch_recording(name, only=None):
    """Copy <SOUMI_RIG_DATA_DIR>\\<name> from the rig into data/<name>-rig/ (tar over ssh, base64 text,
    because the rig's login banner breaks scp)."""
    import base64, tarfile, io
    st = FETCH[name] = {"state": "running", "bytes": 0, "total": None, "error": None, "dir": None}
    remote = (
        "import io,os,sys,tarfile,base64\n"
        f"d=r'{L.DATA_DIR}\\{name}'\n"
        f"only={only!r}\n"
        "tot=sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(d) for f in fs)\n"
        "print('@@TOTAL',tot,flush=True)\n"
        "class W:\n"
        "  def write(self,b):\n"
        "    sys.stdout.write(base64.b64encode(b).decode()+'\\n'); return len(b)\n"
        "  def flush(self): sys.stdout.flush()\n"
        "t=tarfile.open(fileobj=W(),mode='w|',bufsize=1<<20)\n"
        "n=os.path.basename(d)\n"
        "if only:\n"
        "  t.add(os.path.join(d,'meta.json'),arcname=n+'/meta.json')\n"
        "  for o in (only if isinstance(only,(list,tuple)) else [only]):\n"
        "    if os.path.isdir(os.path.join(d,o)): t.add(os.path.join(d,o),arcname=n+'/'+o)\n"
        "else: t.add(d,arcname=n)\n"
        "t.close()\n"
        "print('@@END',flush=True)\n"
    )
    try:
        p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", RIG["host"],
                              f"& '{L.PY}' -u -"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        p.stdin.write(remote); p.stdin.close()
        buf = io.BytesIO(); started = False
        for line in p.stdout:
            line = line.strip()
            if line.startswith("@@TOTAL"):
                st["total"] = int(line.split()[1]); started = True; continue
            if line == "@@END":
                break
            if started and line:
                b = base64.b64decode(line); buf.write(b); st["bytes"] += len(b)
        p.wait(timeout=30)
        buf.seek(0)
        dest = DATA_DIR / f"{name}-rig"
        with tarfile.open(fileobj=buf, mode="r|") as t:
            t.extractall(DATA_DIR / ".rig_tmp", filter="data")
        (DATA_DIR / ".rig_tmp" / name).rename(dest)
        (DATA_DIR / ".rig_tmp").rmdir()
        st.update(state="done", dir=str(dest.relative_to(VIEWER_DIR.parent)))
    except Exception as e:
        st.update(state="error", error=str(e))



# ---------------------------------------------------------------- robot run: film the overhead while real/replay.py runs
ROBOTRUN = {"state": "idle"}   # idle | filming | processing | done | error
OVERLAY_CMD = ["uv", "run", "-q", "--python", "3.12", "--with", "mujoco==3.15.0", "--with", "gymnasium==1.4.0",
               "--with", "numpy", "--with", "pillow", "--with", "opencv-python<5", "--with", "matplotlib"]


def list_plans():
    out = []
    for d in sorted((DATA_DIR / "episodes").glob("*"), reverse=True):
        sides = [s for s in ("left", "right") if (d / f"plan_{s}.json").exists()]
        if d.is_dir() and sides and SAFE_NAME.match(d.name) and ".bak" not in d.name:
            out.append({"episode": d.name, "sides": sides})
    return out


def robotrun_process(name, episode, side, t_start, t_stop):
    """After filming: copy the overhead frames to the Mac, fetch the replay's log, build the overlap video."""
    import json as _json
    import sys as _sys
    st = ROBOTRUN
    try:
        st.update(state="processing", step="copying the overhead + robot wrist recording to the Mac")
        fetch_recording(name, only=("overhead", "robot_wrist"))
        if FETCH[name]["state"] != "done":
            raise RuntimeError(f"copy failed: {FETCH[name]['error']}")
        st.update(step="finding the replay's log on the rig")
        _sys.path.insert(0, str(VIEWER_DIR.parent / "umi"))
        import rig_link
        run_dir = rig_link.latest_replay_run(episode, side)
        if not run_dir:
            raise RuntimeError(f"no live replay of {episode} {side} found on the rig. Was real/run.py --live run?")
        local = DATA_DIR / f"{name}-rig" / "replay_log.json"
        rig_link.download(run_dir + "\\log.json", local)
        t0 = _json.loads(local.read_text()).get("t0_unix")
        rm = _json.loads((DATA_DIR / f"{name}-rig" / "meta.json").read_text())     # rig clock, same as t0
        if t0 is None or not (rm["t0_unix"] <= t0 <= rm["t0_unix"] + rm["duration_s"]):
            raise RuntimeError(f"the newest replay ({run_dir}) did not start while this was filming")
        st.update(step="drawing the trajectories (overlay video)")
        r = subprocess.run(OVERLAY_CMD + [str(VIEWER_DIR.parent / "real" / "overlay.py"), episode, "--side", side,
                                          "--recording", str(DATA_DIR / f"{name}-rig"), "--log", str(local)],
                           cwd=VIEWER_DIR.parent, capture_output=True, text=True, timeout=1200)
        if r.returncode != 0:
            raise RuntimeError("overlay failed: " + (r.stderr or r.stdout).strip()[-800:])
        base = f"episodes/{episode}/robotrun_{side}_{name}"
        stats = _json.loads((DATA_DIR / f"{base}.json").read_text())
        st.update(step="saving the comparison episode")
        saved = UJ.save_comparison(episode, side, name, run_dir, local, DATA_DIR / base, stats)
        wv = f"/{saved}/wrist.mp4" if (VIEWER_DIR.parent / saved / "wrist.mp4").exists() else None
        st.update(state="done", step=None, video=f"/{saved}/comparison.mp4", plot=f"/{saved}/comparison.png", wrist_video=wv,
                  stats=stats, saved=saved)
    except Exception as e:
        st.update(state="error", step=None, error=str(e))


class Handler(SimpleHTTPRequestHandler):
    upstream = f"http://127.0.0.1:{L.MONITOR_PORT}"

    def end_headers(self):
        # Never cache the viewer, so edits show up on a normal reload.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "/overhead.mjpg" not in (str(args[0]) if args else ""):
            super().log_message(fmt, *args)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/rig/"):
            q = urlparse(self.path).query
            return self.rig_get(path[4:] + (f"?{q}" if q else ""))
        if path in RELAYS:
            return self.relay(f"{self.upstream}/cam/{RELAYS[path]}.mjpg")
        if path.startswith("/umi/"):
            return self.umi_get(path[len("/umi/"):], urlparse(self.path).query)
        if path == "/robotrun/plans":
            return self._json(200, list_plans())
        if path == "/robotrun/status":
            return self._json(200, ROBOTRUN)
        if path.startswith("/data/"):
            return self.serve_data(path[len("/data/"):])
        return super().do_GET()

    def serve_data(self, rel):
        """Read-only access to results: episodes (videos, plots, plan JSON, browser-replay data), the shared
        browser meshes (data/web/), and each rig capture's meta.json. Nothing else under data/."""
        f = (DATA_DIR / rel).resolve()
        types = {".mp4": "video/mp4", ".png": "image/png", ".json": "application/json", ".bin": "application/octet-stream",
                 ".csv": "text/csv"}
        ep_root, web_root = (DATA_DIR / "episodes").resolve(), (DATA_DIR / "web").resolve()
        allowed = ((ep_root in f.parents and f.suffix in types)
                   or (f.parent == web_root and f.suffix in (".bin", ".json"))
                   or (f.name == "meta.json" and f.parent.parent == DATA_DIR.resolve() and f.parent.name.endswith("-rig")))
        if not allowed or not f.is_file():
            return self.send_error(404)
        data = f.read_bytes()
        rng = self.headers.get("Range")
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", types.get(f.suffix, "application/octet-stream"))
        self.send_header("Accept-Ranges", "bytes")
        if rng:                                  # Safari needs ranges to play video
            a, _, b = rng.removeprefix("bytes=").partition("-")
            a, b = int(a or 0), int(b) if b else len(data) - 1
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            data = data[a:b + 1]
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def relay(self, url):
        """Serve the overhead feed to the page from the server's single upstream connection.
        (A second connection to the rig would split the tunnel's ~1 MB/s and halve both fps.)"""
        RECORDER.ensure_reader(url)
        for _ in range(50):
            if RECORDER.connected and RECORDER.latest is not None:
                break
            time.sleep(0.1)
        else:
            return self.send_error(502, "overhead feed unreachable; is the tunnel up and monitor.py running?")
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            for jpg in RECORDER.frames():
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                 str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass  # page closed

    def rig_get(self, sub):
        if sub == "/fetch/status":
            return self._json(200, FETCH)
        if sub == "/api/mode":
            return self._json(200, {"rig": RIG["host"]})
        try:
            up = urllib.request.urlopen(rig_url(sub), timeout=5)
        except Exception as e:
            return self._json(502, {"error": f"rig recorder unreachable ({e}). Is viewer/rig/umi_capture.py running and the tunnel up?"})
        with up:
            self.send_response(200)
            self.send_header("Content-Type", up.headers.get("Content-Type", "application/octet-stream"))
            self.end_headers()
            try:
                while chunk := up.read1(64 * 1024):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                pass

    def rig_post(self, sub, query):
        if sub == "/fetch":
            name = parse_qs(query).get("name", [""])[0]
            if not SAFE_NAME.match(name):
                return self._json(400, {"error": "bad name"})
            if FETCH.get(name, {}).get("state") == "running":
                return self._json(409, {"error": "already fetching"})
            only = parse_qs(query).get("only", [None])[0]
            if only not in (None, "umi", "overhead"):
                return self._json(400, {"error": "only must be umi or overhead"})
            threading.Thread(target=fetch_recording, args=(name, only), daemon=True).start()
            return self._json(200, {"fetching": name})
        auto_side = None
        if sub == "/record/stop":           # studio.html: process the new recording automatically
            auto_side = parse_qs(query).get("auto", [None])[0]
            query = ""
        try:
            req = urllib.request.Request(rig_url(sub + (f"?{query}" if query else "")), method="POST", data=b"")
            with urllib.request.urlopen(req, timeout=30) as up:
                res = __import__("json").loads(up.read())
            if auto_side in ("left", "right", "both") and res.get("name", "").startswith("capture-"):
                try:
                    res["auto"] = UJ.generate(res["name"], auto_side, fetch_recording, lambda: FETCH, fetch_live=fetch_live)
                except RuntimeError as e:          # another job is running
                    res["auto"] = {"error": str(e)}
            return self._json(200, res)
        except urllib.error.HTTPError as e:
            return self._json(e.code, __import__("json").loads(e.read() or b"{}"))
        except Exception as e:
            return self._json(502, {"error": f"rig recorder unreachable ({e})"})

    def _json(self, code, obj):
        body = __import__("json").dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    # ---- "Generate UMI trajectory" / "Run real" (viewer/umi_jobs.py)
    def umi_get(self, sub, query):
        q = parse_qs(query)
        if sub == "status":
            return self._json(200, UJ.status())
        if sub == "captures":
            return self._json(200, UJ.captures())
        if sub == "episode":
            ts = q.get("ts", [""])[0]
            if not SAFE_NAME.match(ts):
                return self._json(400, {"error": "bad episode"})
            return self._json(200, UJ.episode_info(ts))
        return self._json(404, {"error": "unknown"})

    def umi_post(self, sub, query):
        q = parse_qs(query)
        side = q.get("side", ["right"])[0]
        try:
            if sub == "generate":
                name = q.get("capture", [""])[0]
                if not (SAFE_NAME.match(name) and name.startswith("capture-")) or side not in ("left", "right", "both"):
                    return self._json(400, {"error": "bad capture or side"})
                if not RIG["host"] and not (DATA_DIR / f"{name}-rig").exists():
                    return self._json(400, {"error": "not on the Mac and no rig (--rig) to copy it from"})
                return self._json(200, UJ.generate(name, side, fetch_recording, lambda: FETCH, fetch_live=fetch_live))
            if sub == "run":
                ts = q.get("episode", [""])[0]
                if not SAFE_NAME.match(ts) or side not in ("left", "right"):
                    return self._json(400, {"error": "bad episode or side"})
                return self._json(200, UJ.run(ts, side, q.get("live", ["0"])[0] == "1"))
            if sub == "confirm":
                return self._json(200, UJ.confirm(q.get("text", [""])[0]))
            if sub == "stop":
                return self._json(200, UJ.stop(RIG["host"]))
        except RuntimeError as e:
            return self._json(409, {"error": str(e)})
        return self._json(404, {"error": "unknown"})

    def robotrun_post(self, sub, query):
        q = parse_qs(query)
        if sub == "start":
            if ROBOTRUN["state"] in ("filming", "processing"):
                return self._json(409, {"error": f"already {ROBOTRUN['state']}"})
            ep, side = q.get("episode", [""])[0], q.get("side", [""])[0]
            if not any(p["episode"] == ep and side in p["sides"] for p in list_plans()):
                return self._json(400, {"error": "no such plan"})
            name = f"robotrun-{time.strftime('%Y-%m-%dT%H-%M-%S')}-{side}"
            try:
                req = urllib.request.Request(rig_url(f"/record/start?name={name}"), method="POST", data=b"")
                urllib.request.urlopen(req, timeout=15).read()
            except Exception as e:
                return self._json(502, {"error": f"could not start the rig recording ({e})"})
            ROBOTRUN.clear()
            ROBOTRUN.update(state="filming", name=name, episode=ep, side=side, t_start=time.time(),
                            command=f"python3 real/run.py {ep} --side {side} --live")
            return self._json(200, ROBOTRUN)
        if sub == "stop":
            if ROBOTRUN["state"] != "filming":
                return self._json(409, {"error": "not filming"})
            try:
                req = urllib.request.Request(rig_url("/record/stop"), method="POST", data=b"")
                urllib.request.urlopen(req, timeout=30).read()
            except Exception as e:
                ROBOTRUN.update(state="error", error=f"could not stop the rig recording ({e})")
                return self._json(502, ROBOTRUN)
            ROBOTRUN.update(state="processing", step="stopping", t_stop=time.time())
            threading.Thread(target=robotrun_process, daemon=True,
                             args=(ROBOTRUN["name"], ROBOTRUN["episode"], ROBOTRUN["side"],
                                   ROBOTRUN["t_start"], ROBOTRUN["t_stop"])).start()
            return self._json(200, ROBOTRUN)
        return self._json(404, {"error": "unknown"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path.startswith("/umi/"):
            return self.umi_post(url.path[len("/umi/"):], url.query)
        if url.path.startswith("/robotrun/"):
            return self.robotrun_post(url.path[len("/robotrun/"):], url.query)
        if url.path.startswith("/rig/"):
            return self.rig_post(url.path[4:], url.query)
        if url.path == "/record/start":
            base = parse_qs(url.query).get("name", [""])[0]
            if not SAFE_NAME.match(base):
                return self.send_error(400, "bad name")
            try:
                return self._json(200, RECORDER.start(f"{self.upstream}/cam/overhead.mjpg", base))
            except Exception as e:
                return self._json(409, {"error": str(e)})
        if url.path == "/record/stop":
            return self._json(200, RECORDER.stop())
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


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def open_tunnel(host, port):
    """Start `ssh -N -L port:127.0.0.1:port host` unless something already listens on port."""
    if port_open(port):
        print(f"Overhead: port {port} already open, reusing it")
        return
    proc = subprocess.Popen(
        ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
         "-o", "ServerAliveInterval=15", "-L", f"{port}:127.0.0.1:{port}", host],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    atexit.register(proc.terminate)
    for _ in range(40):
        if port_open(port):
            print(f"Overhead: tunnel to {host} up on port {port}")
            return
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    print(f"Overhead: could not open tunnel to {host} (ssh exit {proc.poll()}); continuing without it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tunnel", metavar="SSH_HOST", nargs="?", const=L.HOST or None,
                    help="open an SSH tunnel to the rig PC running monitor.py (host defaults to SOUMI_RIG_HOST)")
    ap.add_argument("--monitor-port", type=int, default=L.MONITOR_PORT, help="monitor.py port on the rig PC")
    ap.add_argument("--rig", metavar="SSH_HOST", nargs="?", const=L.HOST or None,
                    help=f"record on the rig PC with viewer/rig/umi_capture.py (port {L.CAPTURE_PORT}): opens that tunnel too; "
                         "use viewer/studio.html (host defaults to SOUMI_RIG_HOST)")
    args = ap.parse_args()
    for flag in ("tunnel", "rig"):
        if f"--{flag}" in sys.argv and not getattr(args, flag):
            ap.error(f"--{flag} needs a host: pass one or set SOUMI_RIG_HOST in .env")
    if args.tunnel:
        open_tunnel(args.tunnel, args.monitor_port)
    if args.rig:
        RIG["host"] = args.rig
        open_tunnel(args.rig, RIG["port"])
    Handler.upstream = f"http://127.0.0.1:{args.monitor_port}"
    RECORDER.ensure_reader(f"{Handler.upstream}/cam/overhead.mjpg")
    handler = partial(Handler, directory=str(VIEWER_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server.daemon_threads = True
    print(f"Viewer: http://localhost:{args.port}  (saving to {DATA_DIR})")
    if args.rig:
        print(f"UMI Studio: http://localhost:{args.port}/studio.html  (cameras and arms on {args.rig})")
    server.serve_forever()


if __name__ == "__main__":
    main()
