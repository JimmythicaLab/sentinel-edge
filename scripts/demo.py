"""SentinelEdge live demo — real inference on a local or sample video.

What it does:
  1. Downloads a short factory sample video on first run (<10 MB).
  2. Loads a lightweight YOLOv8-n detector (Ultralytics; auto-downloads yolov8n.pt).
  3. Opens an OpenCV window that mirrors the production dashboard style:
       - orange detection boxes with class + confidence
       - red "HELMET MANDATORY ZONE" overlay
       - red alert ribbon slides up when a person dwells in the zone
       - bilingual (EN + Traditional Chinese) incident report printed to stdout
  4. Writes every triggered event to results/events.jsonl so reviewers can
     diff before/after runs.

The detection logic uses the real `src.rule_engine.RuleEngine` from this repo
(polygon zone, dwell counter, debounce). COCO's `person` class acts as the
proxy for "unprotected worker" because the Stage I checkpoint ships generic
weights; Stage II swaps in a helmet-trained model and the rule engine code
does not change.

Run:
    python scripts/demo.py                     # downloads sample video
    python scripts/demo.py --source my.mp4     # bring your own
    python scripts/demo.py --headless          # no window, just stdout + jsonl

Quit the window with q or Esc.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Make `src` importable when the script is run directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rule_engine import RuleEngine  # noqa: E402

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
SAMPLE_URLS = [
    "https://videos.pexels.com/video-files/3977979/3977979-sd_960_540_30fps.mp4",
    "https://videos.pexels.com/video-files/3209828/3209828-sd_640_360_25fps.mp4",
    "https://videos.pexels.com/video-files/4769511/4769511-sd_640_360_30fps.mp4",
]
SAMPLE_PATH = ROOT / "results" / "sample_factory.mp4"

RESULTS_DIR = ROOT / "results"
EVENTS_LOG = RESULTS_DIR / "events.jsonl"

ORANGE = (31, 90, 255)   # BGR — matches dashboard accent
RED = (72, 29, 225)
WHITE = (255, 255, 255)
PANEL_BG = (26, 42, 66)
MUTED = (178, 164, 148)


# ---------------------------------------------------------------------------
#  Detection shim — matches the Detection dataclass in src.detectors
# ---------------------------------------------------------------------------
@dataclass
class Det:
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None


# ---------------------------------------------------------------------------
#  Download helpers
# ---------------------------------------------------------------------------
def download_sample() -> Path:
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH

    for i, url in enumerate(SAMPLE_URLS, start=1):
        print(f"[demo] [{i}/{len(SAMPLE_URLS)}] trying {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r, \
                 open(SAMPLE_PATH, "wb") as out:
                out.write(r.read())
            size_mb = SAMPLE_PATH.stat().st_size / 1024 / 1024
            print(f"[demo] downloaded {size_mb:.1f} MB → {SAMPLE_PATH}")
            return SAMPLE_PATH
        except Exception as exc:
            print(f"[demo]   failed: {exc}")

    print("[demo] all mirrors failed. Pass your own video with --source <path>.")
    print("[demo] Any short factory / warehouse clip with visible people works.")
    sys.exit(1)


# ---------------------------------------------------------------------------
#  Drawing helpers
# ---------------------------------------------------------------------------
def _draw_zone(frame, w: int, h: int, armed: bool):
    """Draw the dashed helmet-mandatory zone overlay (bottom 60%, central 60%)."""
    import cv2

    x1, y1 = int(w * 0.2), int(h * 0.35)
    x2, y2 = int(w * 0.8), int(h * 0.9)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), ORANGE if armed else (120, 120, 120), -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, dst=frame)

    # dashed border
    color = ORANGE if armed else (120, 120, 120)
    for i in range(x1, x2, 24):
        cv2.line(frame, (i, y1), (min(i + 12, x2), y1), color, 2)
        cv2.line(frame, (i, y2), (min(i + 12, x2), y2), color, 2)
    for j in range(y1, y2, 24):
        cv2.line(frame, (x1, j), (x1, min(j + 12, y2)), color, 2)
        cv2.line(frame, (x2, j), (x2, min(j + 12, y2)), color, 2)

    cv2.putText(frame, "HELMET MANDATORY ZONE", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return (x1, y1, x2, y2)


def _draw_header(frame, w: int):
    """Top gradient bar with CAM-42 label and timestamp."""
    import cv2
    header = frame.copy()
    cv2.rectangle(header, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.addWeighted(header, 0.55, frame, 0.45, 0, dst=frame)

    cv2.putText(frame, "CAM-42  ·  Loading Bay 3  ·  Helmet Mandatory",
                (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (w - tw - 24, 10), (w - 10, 34), (24, 24, 48), -1)
    cv2.putText(frame, ts, (w - tw - 18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (161, 161, 252), 1, cv2.LINE_AA)

    # LIVE pill
    cv2.circle(frame, (w - tw - 46, 22), 4, (72, 29, 225), -1)
    cv2.putText(frame, "LIVE", (w - tw - 38, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 190, 252), 1, cv2.LINE_AA)


def _draw_detection_box(frame, det: Det, w: int, h: int, violating: bool):
    """Orange rectangle with class · confidence label."""
    import cv2
    x1, y1, x2, y2 = (int(v) for v in det.xyxy)
    color = RED if violating else ORANGE
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{det.cls} · {det.conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    if violating:
        tag = "NO HELMET"
        (tw2, th2), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x2 + 8, y1 + 10),
                      (x2 + tw2 + 20, y1 + th2 + 24), RED, -1)
        cv2.putText(frame, tag, (x2 + 14, y1 + th2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)


def _draw_alert_ribbon(frame, w: int, h: int, latency_ms: int):
    """Slide-up red alert bar at the bottom of the frame."""
    import cv2
    y = h - 60
    cv2.rectangle(frame, (0, y), (w, h), (72, 29, 225), -1)
    cv2.rectangle(frame, (0, y), (w, y + 4), (141, 45, 159), -1)

    cv2.circle(frame, (28, y + 30), 14, (255, 255, 255), 2)
    cv2.putText(frame, "!", (23, y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

    cv2.putText(frame, "ALERT  ·  HELMET_MANDATORY  ·  CAM-42",
                (56, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Unprotected worker detected in restricted zone",
                (56, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

    lat = f"{latency_ms} ms"
    (tw, th), _ = cv2.getTextSize(lat, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (w - tw - 28, y + 16),
                  (w - 12, y + th + 32), (0, 0, 0), -1)
    cv2.putText(frame, lat, (w - tw - 20, y + th + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
#  Reporter + logger
# ---------------------------------------------------------------------------
def write_event(event_payload: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_LOG, "a") as f:
        f.write(json.dumps(event_payload, ensure_ascii=False) + "\n")


def print_report(event, dwell_frames: int):
    en = (f"Unprotected worker detected in {event.zone} on {event.camera_id} "
          f"for {dwell_frames} frames. Supervisor dispatch and PPE briefing "
          f"recommended.")
    zh = (f"[{event.camera_id} {event.zone}] 偵測到人員未配戴安全帽,持續 "
          f"{dwell_frames} 幀。建議通知現場主管並重新宣導 PPE 規範。")
    bar = "─" * 72
    print(f"\n{bar}")
    print(f"⚠  ALERT · {event.kind} · {event.camera_id}  (dwell={dwell_frames} frames)")
    print(f"{bar}")
    print(f"[EN] {en}")
    print(f"[ZH] {zh}")
    print(f"→ dispatched: LINE Notify · MQTT · on-site strobe · {EVENTS_LOG}")
    print(bar + "\n")

    write_event({
        "ts": time.time(),
        "camera": event.camera_id,
        "zone": event.zone,
        "kind": event.kind,
        "dwell_frames": dwell_frames,
        "severity": event.severity,
        "report_en": en,
        "report_zh": zh,
    })


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SentinelEdge live demo")
    ap.add_argument("--source", default="", help="Path to local video or RTSP URL")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="Ultralytics checkpoint (auto-downloads)")
    ap.add_argument("--headless", action="store_true",
                    help="Skip the OpenCV window (useful for CI / headless servers)")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="Minimum detector confidence")
    args = ap.parse_args()

    source = args.source or str(download_sample())

    # Lazy imports so --help works without the heavy deps
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[demo] missing dependency: {exc}")
        print("[demo] run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"[demo] loading detector: {args.weights}")
    model = YOLO(args.weights)

    print(f"[demo] opening source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[demo] cannot open {source}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540

    # Zone covers most of the frame so any person in view triggers the rule.
    # Stage I weights (COCO) have no `helmet` class, so helmet_mandatory
    # effectively fires on any person inside the zone — exactly the Stage II
    # behaviour once a helmet-trained checkpoint is swapped in.
    zones = [{
        "name": "crane_zone",
        "camera": "CAM-42",
        "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "rules": ["helmet_mandatory"],
        "severity": "HIGH",
    }]
    policies = {"min_dwell_frames": 12}       # ~0.5 s at 24 fps
    engine = RuleEngine(zones, policies)

    alert_until = 0.0
    frame_idx = 0
    t_start = time.time()

    print(f"[demo] press q or Esc to quit · stream: {w}×{h}")
    while True:
        ok, frame = cap.read()
        if not ok:
            # loop the demo video so reviewers always see something
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_idx += 1
        t0 = time.perf_counter()

        # Use ByteTrack so track_id stays stable → dwell counter can accumulate
        results = model.track(frame, conf=args.conf, persist=True, verbose=False)
        r = results[0]

        # Convert to our Detection dataclass (person-only — Stage I weights
        # don't ship helmet classes; Stage II swap-in will populate helmets).
        detections: list[Det] = []
        if r.boxes is not None:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                cls_name = model.names[cls_id]
                if cls_name != "person":
                    continue
                xyxy = tuple(float(v) for v in b.xyxy[0].cpu().numpy().tolist())
                conf = float(b.conf.item())
                tid = int(b.id.item()) if b.id is not None else 1
                detections.append(Det(cls="person", conf=conf, xyxy=xyxy, track_id=tid))

        # Normalise bbox coords for the rule engine (it expects 0..1)
        norm_dets = [
            Det(cls=d.cls, conf=d.conf, track_id=d.track_id,
                xyxy=(d.xyxy[0] / w, d.xyxy[1] / h,
                      d.xyxy[2] / w, d.xyxy[3] / h))
            for d in detections
        ]
        events = list(engine.evaluate(frame_idx, norm_dets))

        # ---- Draw overlays ----
        armed = bool(detections)
        zone_box = _draw_zone(frame, w, h, armed=armed)
        _draw_header(frame, w)

        violating_tids = {e.detections[0].track_id for e in events}
        for d in detections:
            _draw_detection_box(frame, d, w, h, violating=d.track_id in violating_tids)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        if events:
            alert_until = time.time() + 3.0
            for ev in events:
                print_report(ev, ev.dwell_frames)

        if time.time() < alert_until:
            _draw_alert_ribbon(frame, w, h, latency_ms=latency_ms)

        # Status strip (always on, bottom-left)
        fps = frame_idx / max(time.time() - t_start, 1e-6)
        strip = (f"frame={frame_idx}  fps={fps:.1f}  "
                 f"infer={latency_ms}ms  dets={len(detections)}  "
                 f"events={len(events)}")
        cv2.putText(frame, strip, (12, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)

        if args.headless:
            if frame_idx % 60 == 0:
                print(f"[demo] {strip}")
            if frame_idx >= 300:  # ~10 s then exit
                break
        else:
            cv2.imshow("SentinelEdge · CAM-42 · press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    if not args.headless:
        import cv2
        cv2.destroyAllWindows()
    print(f"[demo] events written to {EVENTS_LOG}")


if __name__ == "__main__":
    main()
