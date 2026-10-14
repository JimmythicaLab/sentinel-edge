"""Export per-frame tracks for the webdemo overlay.

Reuses the project's object detector backend (src/detectors.TorchDetector)
to run detection + tracking on assets/sample_factory.mp4 and writes a
compact JSON file that webdemo/js/app.js consumes at runtime to render
real bounding boxes on top of the video.

The resulting webdemo/assets/tracks/cam42.json is backed by the same
detector that scripts/demo.py uses — the overlay in the browser is not
hand-placed CSS.

Usage:
    DETECTOR_WEIGHTS=/path/to/weights.pt python3 scripts/export_webdemo_tracks.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detectors import build_detector  # noqa: E402


SOURCE_VIDEO = Path("assets/sample_factory.mp4")
OUTPUT_JSON = Path("webdemo/assets/tracks/cam42.json")
OUTPUT_JS   = Path("webdemo/assets/tracks/cam42.js")
CLASS_ALLOWLIST = {"person"}


async def run() -> None:
    weights = os.environ.get("DETECTOR_WEIGHTS")
    if not weights:
        raise SystemExit(
            "Set DETECTOR_WEIGHTS to your object-detector weights file "
            "(e.g. DETECTOR_WEIGHTS=detector.pt)."
        )

    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    detector = build_detector("torch", weights)

    frames: list[dict] = []
    async for i, _img, dets in detector.stream(str(SOURCE_VIDEO)):
        t = i / fps
        boxes = []
        for d in dets:
            if d.cls not in CLASS_ALLOWLIST:
                continue
            x1, y1, x2, y2 = d.xyxy
            boxes.append(
                {
                    "id": d.track_id if d.track_id is not None else -1,
                    "cls": d.cls,
                    "conf": round(d.conf, 2),
                    "x": round(x1 / width, 4),
                    "y": round(y1 / height, 4),
                    "w": round((x2 - x1) / width, 4),
                    "h": round((y2 - y1) / height, 4),
                }
            )
        frames.append({"t": round(t, 3), "boxes": boxes})

    payload = {
        "video": str(SOURCE_VIDEO),
        "fps": round(fps, 2),
        "frames": frames,
        "zone": {
            "name": "restricted_zone",
            "polygon": [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]],
            "rule": "restricted_access",
        },
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, separators=(",", ":")))
    # Also write a JSONP-style sibling so webdemo loads under file://
    # without needing a local HTTP server.
    OUTPUT_JS.write_text(
        "window.CAM42_TRACKS = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    hits = sum(1 for f in frames if f["boxes"])
    print(
        f"wrote {OUTPUT_JSON} · {len(frames)} frames @ {fps:.1f}fps · "
        f"{hits} frames with detections ({width}x{height})"
    )


if __name__ == "__main__":
    asyncio.run(run())
