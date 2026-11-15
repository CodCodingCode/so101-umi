"""SSH helpers for the Windows rig PC (see robots/sim/DZKCHEN_REMOTE.md for why it works this way).

The remote shell is PowerShell 5 and every login prints profile noise, so output is read after a marker line,
and files move as base64 (scp/sftp don't work there).
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path

HOST = "dzkchen"
PY = r"C:\Users\dzkch\.venvs\so101-pi\Scripts\python.exe"
REAL = r"C:\robot\real"
UMI = REAL + r"\umi"                 # where the SO-UMI rig scripts and replay files live
MARK = "@@UMI@@"
SSH = ["ssh", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", HOST]


def run(ps: str, timeout=120) -> str:
    """Run a PowerShell command; return what it printed after the noise."""
    r = subprocess.run(SSH + [f"Write-Output '{MARK}'; {ps}"], capture_output=True, text=True, timeout=timeout)
    # The login profile writes an error to stderr every time; real output is on stdout after the marker.
    if MARK not in r.stdout:
        raise RuntimeError(f"ssh {HOST} failed (exit {r.returncode}):\n{(r.stdout + r.stderr).strip()[-2000:]}")
    return r.stdout.split(MARK, 1)[1].strip("\r\n")


def run_py(script: str, args: str = "", timeout=120) -> str:
    return run(f"& '{PY}' -u '{script}' {args}; Write-Output \"PYTHON_EXIT $LASTEXITCODE\"", timeout=timeout)


def run_py_live(script: str, args: str = "") -> int:
    """Run interactively (output streams, Ctrl+C reaches the script). Returns the exit code."""
    return subprocess.call(["ssh", "-t", "-o", "ServerAliveInterval=15", HOST,
                            f"& '{PY}' -u '{script}' {args}; exit $LASTEXITCODE"])


def upload(local: Path, remote: str, chunk=6000):
    b64 = base64.b64encode(Path(local).read_bytes()).decode()
    tmp = remote + ".b64"
    run(f"New-Item -ItemType Directory -Force -Path (Split-Path '{remote}') | Out-Null")
    for i in range(0, max(len(b64), 1), chunk):
        part = b64[i:i + chunk]
        verb = "Set-Content" if i == 0 else "Add-Content"
        run(f"{verb} -NoNewline -Encoding ascii -Path '{tmp}' -Value '{part}'")
    size = run(f"[IO.File]::WriteAllBytes('{remote}', [Convert]::FromBase64String((Get-Content -Raw '{tmp}').Trim())); "
               f"Remove-Item '{tmp}'; (Get-Item '{remote}').Length").splitlines()[-1].strip()
    want = Path(local).stat().st_size
    if int(size) != want:
        raise RuntimeError(f"upload of {local} -> {remote}: {size} bytes on the rig, {want} locally")


def sync_rig_scripts():
    """Copy umi/rig/*.py to the rig if they differ."""
    here = Path(__file__).resolve().parent / "rig"
    for f in sorted(here.glob("*.py")):
        remote = f"{UMI}\\{f.name}"
        try:
            n = int(run(f"if (Test-Path '{remote}') {{ (Get-Item '{remote}').Length }} else {{ -1 }}").splitlines()[-1])
        except ValueError:
            n = -1
        if n != f.stat().st_size:
            upload(f, remote)


def download(remote: str, local: Path):
    """Copy one file from the rig (base64 over plain ssh)."""
    out = run(f"[Convert]::ToBase64String([IO.File]::ReadAllBytes('{remote}'))", timeout=300)
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    Path(local).write_bytes(base64.b64decode("".join(out.split())))


def latest_replay_run(episode: str, side: str):
    """Newest C:\\robot\\real\\runs\\umi_replay_* whose meta.json is a LIVE run of this episode/side, or None."""
    ps = (f"Get-ChildItem '{REAL}\\runs' -Directory -Filter 'umi_replay_*' | Sort-Object Name -Descending | "
          f"Select-Object -First 30 | ForEach-Object {{ $m = Get-Content -Raw (Join-Path $_.FullName 'meta.json') | "
          f"ConvertFrom-Json; if ($m.live -and $m.episode -eq '{episode}' -and $m.side -eq '{side}' -and "
          f"(Test-Path (Join-Path $_.FullName 'log.json'))) {{ $_.FullName }} }} | Select-Object -First 1")
    out = run(ps, timeout=60).strip()
    return out or None
