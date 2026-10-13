"""Benchmark the Torch backend on the host machine.

Measures single-stream FPS and per-frame latency on CPU and (optional) GPU.
Produces results/bench_torch.json for the deck's benchmark table.

Usage:
    python scripts/bench_torch.py                 # all backends, default source
    python scripts/bench_torch.py --device cpu    # CPU only
    python scripts/bench_torch.py --source path/to/video.mp4

The numbers here are the Stage I reference measurement — Hailo INT8 numbers
are produced from scripts/bench_hailo.py once a UGen300 is available.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _resolve_source(source: str) -> str:
    """Fall back to a tiny built-in COCO sample if no source is given."""
    if source and Path(source).exists():
        return source
    # Ultralytics ships a demo bus image we can re-run many times.
    return "https://ultralytics.com/images/bus.jpg"


def bench_once(weights: str, source: str, device: str, warmup: int, iters: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    # Warm-up: compiles kernels, pulls weights to device, caches disk I/O
    for _ in range(warmup):
        model.predict(source=source, device=device, verbose=False)

    lat_ms: list[float] = []
    t0 = time.perf_counter()
    for _ in range(iters):
        t_s = time.perf_counter()
        model.predict(source=source, device=device, verbose=False)
        lat_ms.append((time.perf_counter() - t_s) * 1000)
    wall = time.perf_counter() - t0

    lat_ms.sort()
    return {
        "weights": weights,
        "device": device,
        "iters": iters,
        "wall_s": round(wall, 3),
        "fps": round(iters / wall, 2),
        "latency_p50_ms": round(lat_ms[len(lat_ms) // 2], 2),
        "latency_p95_ms": round(lat_ms[int(len(lat_ms) * 0.95)], 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--source", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="results/bench_torch.json")
    args = ap.parse_args()

    source = _resolve_source(args.source)
    devices = [args.device] if args.device != "auto" else _pick_devices()

    results = []
    for dev in devices:
        print(f"[bench] weights={args.weights} source={source} device={dev}")
        r = bench_once(args.weights, source, dev, args.warmup, args.iters)
        print(f"  → fps={r['fps']} p50={r['latency_p50_ms']}ms p95={r['latency_p95_ms']}ms")
        results.append(r)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "source": source}, f, indent=2)
    print(f"[bench] wrote {args.out}")


def _pick_devices() -> list[str]:
    """Pick the best available Torch devices (CPU is always included)."""
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    return devices


if __name__ == "__main__":
    main()
