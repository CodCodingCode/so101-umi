"""Run a SO-UMI plan on the real SO-101 rig from the Mac (rig settings: .env, see .env.example).

    python3 real/run.py <capture-timestamp> --side right            # DRY: checks on the rig, nothing moves
    python3 real/run.py <capture-timestamp> --side right --live     # a PERSON at the rig: moves the arm

Uploads real/replay.py and data/episodes/<ts>/plan_<side>.json to <SOUMI_RIG_REAL_DIR>\\umi\\ on the rig PC and runs the replay there.
Stop monitor.py (and rig/umi_capture.py if it holds the arms) first: the COM ports are exclusive.
Extra arguments after `--` go to replay.py (e.g. `-- --max-deg-s 30 --no-gripper`).
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "umi"))
import rig_link as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", help="capture timestamp under data/episodes/")
    ap.add_argument("--side", choices=("left", "right"), required=True)
    ap.add_argument("--live", action="store_true", help="move the arm (starts right away; add `-- --confirm` for the typed prompt)")
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

    plan = REPO / "data" / "episodes" / a.episode / f"plan_{a.side}.json"
    if not plan.exists():
        raise SystemExit(f"{plan} not found: run umi/to_robot.py {a.episode} --side {a.side} first")
    rplan = f"{L.UMI}\\plan_{a.episode}_{a.side}.json"
    rscript = f"{L.UMI}\\replay.py"
    print(f"uploading replay.py and {plan.relative_to(REPO)} to {L.host()}:{L.UMI}", flush=True)
    L.upload(Path(__file__).resolve().parent / "replay.py", rscript)
    L.upload(plan, rplan)

    extra = " ".join(rest)
    args = f"'{rplan}' {extra}".strip()
    if a.live:
        sys.exit(L.run_py_live(rscript, args + " --live"))
    out = L.run_py(rscript, args, timeout=300)
    print(out)


if __name__ == "__main__":
    main()
