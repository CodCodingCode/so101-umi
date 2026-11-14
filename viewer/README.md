# Camera Viewer

A dependency-free browser page for viewing cameras connected to this machine
(built-in webcam, USB cameras, etc.).

## Run

```bash
python3 viewer/server.py            # optional: --port 8000
```

Then open http://localhost:8000 and allow camera access when prompted.

Captures and snapshots are saved to `data/` at the repo root
(e.g. `data/capture-2026-11-14T10-00-00-000Z.mp4`).

## Features

- Pick any connected camera from the dropdown (list updates when devices are plugged/unplugged)
- Choose resolution (640×480 up to 4K, falls back to what the camera supports)
- Live resolution + FPS readout
- Mirror toggle
- **Capture**: click to start recording video, click again to stop (MP4 in Chrome/Safari, WebM fallback)
- Snapshot to PNG
