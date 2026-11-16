"""Print one arm's joint angles as JSON. READ ONLY: never writes a servo register (no torque, no goals).
Runs on the rig PC with monitor.py stopped (it holds the COM ports).

    python read_joints.py left
"""
import json
import os
import sys
from pathlib import Path

# The rig folder with the arm drivers (so101_real.py, safety.py, rt_control.py, rig.json). real/run.py uploads this
# script to <that folder>\umi\, so by default it is the parent of this script's folder; SOUMI_RIG_REAL_DIR overrides.
HERE = Path(os.environ.get("SOUMI_RIG_REAL_DIR") or Path(__file__).resolve().parents[1])
sys.path[:0] = [str(HERE)]
import so101_real as R  # noqa: E402

side = sys.argv[1]
rig = json.loads((HERE / "rig.json").read_text())
try:
    arm = R.Arm(side, rig["arms"][side], rig.get("joint_offsets_deg", {}).get(side))
except Exception as e:
    print(json.dumps({"error": f"cannot open the {side} arm on {rig['arms'][side]}: {e!r}. "
                      "Is monitor.py (or another runner) still holding the port?"}))
    sys.exit(2)
try:
    if arm.fingerprint != R.FINGERPRINTS[side]:
        print(json.dumps({"error": f"{rig['arms'][side]} is not the {side} arm (fingerprint {arm.fingerprint})"}))
        sys.exit(3)
    raw = arm.read_raw()
    print(json.dumps({"side": side, "units": [round(float(v), 3) for v in arm.to_units(raw)],
                      "raw": [int(v) for v in raw]}))
finally:
    arm.close()
