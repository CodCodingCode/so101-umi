"""Jobs behind studio.html's "Make the sim" and "Run on robot" (one at a time).

generate(capture, side): make sure the rig recording is on the Mac (copy it if needed, and check every
    frame arrived), umi/convert.py -> umi/to_robot.py --side, then transcode plan_<side>.mp4 to H.264 so the
    browser can play it (plan_<side>.web.mp4).
run(episode, side, live): upload real/replay.py + the plan and run it on the rig (what real/run.py does). Without
    live it is a preflight only. With live, replay.py asks for the side to be typed; the page relays it (confirm()).
stop(): touch C:\\robot\\real\\STOP on the rig (the player and every rig runner watch it).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
UV_CV = ["uv", "run", "-q", "--with", "opencv-python<5", "--with", "numpy", "--with", "matplotlib"]
UV_SIM = ["uv", "run", "-q", "--python", "3.12", "--with", "mujoco==3.15.0", "--with", "gymnasium==1.4.0",
          "--with", "numpy", "--with", "pillow", "--with", "opencv-python<5", "--with", "matplotlib", "--with", "scipy"]
NOISE = ("SecurityError", "PSSecurityException", "UnauthorizedAccess", "~~~", "cannot be loaded because running",
         "about_Execution_Policies", "go.microsoft.com", "Microsoft.PowerShell_profile", "CategoryInfo", "FullyQualifiedErrorId")

LOCK = threading.Lock()
JOB = {"id": 0, "kind": None, "state": "idle", "log": [], "result": None, "args": None, "proc": None, "awaiting": None}


def status():
    with LOCK:
        j = {k: v for k, v in JOB.items() if k != "proc"}
        j["log"] = j["log"][-400:]
        return j


def _log(line):
    line = line.rstrip()
    if line and not any(n in line for n in NOISE) and not line.lstrip().startswith(("+ ", "At line:")):
        with LOCK:
            JOB["log"].append(line)


def _start(kind, args, target):
    with LOCK:
        if JOB["state"] == "running":
            raise RuntimeError(f"busy: {JOB['kind']} is still running")
        JOB.update(id=JOB["id"] + 1, kind=kind, state="running", log=[], result=None, args=args, proc=None, awaiting=None,
                   started=time.time())   # the page shows elapsed time from this, so a reload doesn't reset it
    threading.Thread(target=_wrap, args=(target,), daemon=True).start()
    return status()


def _wrap(target):
    try:
        res = target()
        with LOCK:
            JOB.update(state="done", result=res)
    except Exception as e:
        _log(f"ERROR: {e}")
        with LOCK:
            JOB.update(state="error", result={"error": str(e)})
    finally:
        with LOCK:
            JOB["proc"] = None


def _sh(cmd, stdin_text=None):
    _log("$ " + " ".join(c if len(c) < 80 else c[:77] + "..." for c in cmd))
    p = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with LOCK:
        JOB["proc"] = p
    if stdin_text is not None:
        p.stdin.write(stdin_text)
        p.stdin.close()
    for line in p.stdout:
        _log(line)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"exit {rc}: {' '.join(c for c in cmd if not c.startswith('--with') and '==' not in c)[:140]}")


# ---------------------------------------------------------------- captures on the Mac
def capture_complete(d: Path):
    """(ok, why): both cameras present and every frame listed in frames.csv is on disk -- or, for recordings made
    with live tracking on the rig, both cameras' detections.jsonl (no frames needed)."""
    if all((d / cam / "detections.jsonl").exists() and (d / cam / "frames.csv").exists() for cam in ("umi", "overhead")):
        return True, "live tracking from the rig"
    for cam in ("umi", "overhead"):
        csv = d / cam / "frames.csv"
        if not csv.exists():
            return False, f"{cam} missing"
        n = len(csv.read_text().splitlines()) - 1
        have = sum(1 for _ in (d / cam / "frames").glob("*.jpg")) if (d / cam / "frames").exists() else 0
        if have < n:
            return False, f"{cam} has {have}/{n} frames"
    return True, "complete"


def captures():
    """Rig recordings already on the Mac, newest first, with completeness and existing plans."""
    out = []
    for d in sorted(DATA.glob("capture-*-rig"), reverse=True):
        name = d.name.removesuffix("-rig")
        ts = name.removeprefix("capture-")
        ok, why = capture_complete(d)
        ep = DATA / "episodes" / ts
        plans = {s: (ep / f"plan_{s}.json").exists() for s in ("left", "right")}
        out.append({"name": name, "ts": ts, "complete": ok, "why": why, "plans": plans})
    return out


def _ensure_capture(name, fetch):
    d = DATA / f"{name}-rig"
    ok, why = capture_complete(d) if d.exists() else (False, "not on the Mac")
    if ok:
        _log(f"{d.name}: on the Mac, all frames present")
        return
    _log(f"{name}: {why}; copying it from the rig (slow link, a few minutes)")
    if d.exists():
        aside = d.with_name(d.name + f".incomplete-{int(time.time())}")
        d.rename(aside)
        _log(f"  moved the incomplete copy aside to {aside.name}")
    fetch(name)
    ok, why = capture_complete(d) if d.exists() else (False, "copy produced nothing")
    if not ok:
        raise RuntimeError(f"copy from the rig is incomplete ({why}); try again")
    _log("  copy complete")


# ---------------------------------------------------------------- generate
def generate(name, side, fetch, fetch_state, fetch_live=None, video=False):
    """fetch_live(name) -> bool: get just the rig's live tracking (KBs, seconds); falls back to copying frames.
    video=False: optimized plan + in-browser replay data only (no rendering / encoding)."""
    ts = name.removeprefix("capture-")

    def job():
        def fetch_logged(n):
            th = threading.Thread(target=fetch, args=(n,), daemon=True)
            th.start()
            last = -1
            while th.is_alive():
                st = fetch_state().get(n, {})
                if st.get("total"):
                    pct = int(100 * st["bytes"] / st["total"])
                    if pct // 10 != last // 10:
                        _log(f"  copying {pct}% ({st['bytes'] / 1e6:.0f} / {st['total'] / 1e6:.0f} MB)")
                        last = pct
                time.sleep(1.0)
            if fetch_state().get(n, {}).get("state") == "error":
                raise RuntimeError(f"copy failed: {fetch_state()[n]['error']}")
        t_job = time.time()
        d = DATA / f"{name}-rig"
        if fetch_live is not None and not capture_complete(d)[0]:
            _log("fetching the rig's live tracking (no frames to copy)")
            if fetch_live(name):
                _log(f"  live tracking fetched in {time.time() - t_job:.1f} s")
            else:
                _log("  no live tracking for this recording (made before live tracking): copying frames instead")
        _ensure_capture(name, fetch_logged)
        _sh(UV_CV + ["umi/convert.py", ts])
        sides = ("left", "right") if side == "both" else (side,)
        videos = {}
        for s in sides:
            if video:
                _sh(UV_SIM + ["umi/to_robot_fast.py", ts, "--side", s, "--optimize"])
                src = DATA / "episodes" / ts / f"plan_{s}.mp4"
                web = src.with_name(f"plan_{s}.web.mp4")
                _sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-crf", "20", "-preset", "veryfast", "-movflags", "+faststart", str(web)])
                videos[s] = f"/data/episodes/{ts}/{web.name}"
            else:
                _sh(UV_SIM + ["umi/to_robot_fast.py", ts, "--side", s, "--optimize", "--no-video", "--export-web"])
        _log(f"done in {time.time() - t_job:.1f} s")
        plans = {s: json.loads((DATA / "episodes" / ts / f"plan_{s}.json").read_text()) for s in sides}
        return {"episode": ts, "videos": videos,
                "plans": {s: {"checks": p["checks"], "recommended_time_scale": p["recommended_time_scale"],
                              "seconds": p["t"][-1]} for s, p in plans.items()}}

    return _start("generate", {"capture": name, "side": side}, job)


def episode_info(ts):
    ep = DATA / "episodes" / ts
    out = {"episode": ts, "videos": {}, "plans": {}}
    for s in ("left", "right"):
        if (ep / f"plan_{s}.json").exists():
            p = json.loads((ep / f"plan_{s}.json").read_text())
            out["plans"][s] = {"checks": p["checks"], "recommended_time_scale": p["recommended_time_scale"],
                               "seconds": p["t"][-1]}
            if (ep / f"plan_{s}.web.mp4").exists():
                out["videos"][s] = f"/data/episodes/{ts}/plan_{s}.web.mp4"
    out["real_runs"] = []                      # saved comparisons: UMI demo vs the real robot (save_comparison)
    for d in sorted((ep / "real_runs").glob("*"), reverse=True):
        if (d / "meta.json").exists() and (d / "comparison.mp4").exists():
            m = json.loads((d / "meta.json").read_text())
            out["real_runs"].append({"name": d.name, "side": m.get("side"), "saved": m.get("saved"),
                                     "stats": m.get("stats", {}), "folder": str(d.relative_to(REPO)),
                                     "video": f"/data/episodes/{ts}/real_runs/{d.name}/comparison.mp4",
                                     "plot": f"/data/episodes/{ts}/real_runs/{d.name}/comparison.png",
                                     "wrist_video": f"/data/episodes/{ts}/real_runs/{d.name}/wrist.mp4"
                                     if (d / "wrist.mp4").exists() else None})
    return out


# ---------------------------------------------------------------- run on the real arm (real/run.py's path)
def run(ts, side, live, extra=()):
    """Upload real/replay.py + the plan (umi/rig_link.py, as real/run.py does) and run it on the rig.
    Dry: preflight only. Live: replay.py asks a person to type the side; the page relays that answer (confirm())."""
    import os
    import sys
    plan = DATA / "episodes" / ts / f"plan_{side}.json"
    if not plan.exists():
        raise RuntimeError(f"no {plan.relative_to(REPO)}: generate the trajectory first")
    sys.path.insert(0, str(REPO / "umi"))
    import rig_link as L

    chk = json.loads(plan.read_text()).get("checks", {})
    if chk.get("realtime") and not extra:      # plan was shaped for 1x (speed + acceleration capped): play it in real time
        extra = ("--time-scale", "1", "--max-deg-s", f"{float(chk['realtime_vmax_deg_s']):g}")

    def job():
        rplan, rscript = f"{L.UMI}\\plan_{ts}_{side}.json", f"{L.UMI}\\replay.py"
        _log(f"uploading real/replay.py and {plan.relative_to(REPO)} to {L.HOST}:{L.UMI}")
        L.upload(REPO / "real" / "replay.py", rscript)
        L.upload(plan, rplan)
        args = f"'{rplan}' " + " ".join(extra) + (" --live" if live else "")
        _log("LIVE: the arm starts as soon as the preflight passes" if live else "DRY RUN: preflight only, nothing moves")
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5", L.HOST,
               f"& '{L.PY}' -u '{rscript}' {args}; exit $LASTEXITCODE"]
        p = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with LOCK:
            JOB["proc"] = p
        buf = b""
        while True:
            chunk = os.read(p.stdout.fileno(), 4096)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                _log(ln.decode("utf-8", "replace"))
            tail = buf.decode("utf-8", "replace")
            if tail.rstrip().endswith("to start:"):             # replay.py's input() prompt (no newline)
                _log(tail)
                buf = b""
                with LOCK:
                    JOB["awaiting"] = side
        if buf:
            _log(buf.decode("utf-8", "replace"))
        with LOCK:
            JOB["awaiting"] = None
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"replay exited {rc}")
        return {"episode": ts, "side": side, "live": live}

    return _start("run-live" if live else "run-dry", {"episode": ts, "side": side, "live": live}, job)


def confirm(text):
    """Send the typed answer to replay.py's 'type <side> to start' prompt."""
    with LOCK:
        p, want = JOB.get("proc"), JOB.get("awaiting")
        JOB["awaiting"] = None
    if not p or not want:
        raise RuntimeError("nothing is waiting for a confirmation")
    _log(f"> {text}")
    p.stdin.write((text.strip() + "\n").encode())
    p.stdin.flush()
    return {"sent": text.strip(), "expected": want}


def stop(host="dzkchen"):
    """Touch the rig's STOP file (separate ssh, works while a run is streaming)."""
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                        "New-Item -ItemType File -Force -Path 'C:\\robot\\real\\STOP' | Out-Null; Write-Output STOP_OK"],
                       capture_output=True, text=True, timeout=30)
    ok = "STOP_OK" in r.stdout
    _log("STOP file created on the rig: the arm folds back to rest" if ok else f"STOP FAILED (ssh exit {r.returncode}): "
         f"{(r.stdout + r.stderr).strip()[-200:]}. Use the power switch.")
    return {"ok": ok}


# ---------------------------------------------------------------- comparison episode: UMI demo vs real robot run
def save_comparison(episode, side, recording, rig_run_dir, robot_log: Path, overlay_base: Path, stats):
    """Collect one real run into data/episodes/<episode>/real_runs/<recording>/ (self-contained):

        comparison.mp4    robot run (left) | UMI demo (right), both from the overhead, paths drawn (H.264)
        comparison.png    top view + height + distance robot <-> demo
        wrist.mp4         what the arm sees: UMI demo wrist | robot wrist | 50/50 overlay (if the robot wrist was filmed)
        stats.json        robot vs demo / plan distances (real/overlay.py)
        umi_demo.csv      the handheld demo (trajectory.csv: gripper pose in the table frame, jaw)
        plan.json         the joint plan the robot replayed
        robot_log.json    what the real arm did, every tick (rig clock)
        meta.json         where everything came from
    Returns the folder, relative to the repo."""
    ep = DATA / "episodes" / episode
    out = ep / "real_runs" / recording
    out.mkdir(parents=True, exist_ok=True)
    src_mp4 = overlay_base.with_suffix(".mp4")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_mp4), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-preset", "veryfast", "-movflags", "+faststart", str(out / "comparison.mp4")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        shutil.copy2(src_mp4, out / "comparison.mp4")
    shutil.copy2(overlay_base.with_suffix(".png"), out / "comparison.png")
    wsrc = overlay_base.parent / (overlay_base.name + "_wrist.mp4")
    if wsrc.exists():                            # what the arm sees: demo wrist | robot wrist | overlay
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wsrc), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "20", "-preset", "veryfast", "-movflags", "+faststart", str(out / "wrist.mp4")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            shutil.copy2(wsrc, out / "wrist.mp4")
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    shutil.copy2(ep / "trajectory.csv", out / "umi_demo.csv")
    shutil.copy2(ep / f"plan_{side}.json", out / "plan.json")
    shutil.copy2(robot_log, out / "robot_log.json")
    (out / "meta.json").write_text(json.dumps({
        "episode": episode, "side": side,
        "umi_demo": {"recording": f"data/capture-{episode}-rig", "trajectory": f"data/episodes/{episode}/trajectory.csv"},
        "robot_run": {"overhead_recording": f"data/{recording}-rig", "rig_log_dir": rig_run_dir},
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "stats": stats,
    }, indent=1))
    return str(out.relative_to(REPO))
