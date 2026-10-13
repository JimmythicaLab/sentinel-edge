"""End-to-end inference loop.

Stage I: PyTorch backend for development & demo recording on any laptop.
Stage II: swap `--backend hailo` to run compiled HEF models on the UGen300.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.detectors import build_detector
from src.outputs import OutputDispatcher
from src.reporter import BilingualReporter
from src.rule_engine import RuleEngine

console = Console()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run(source: str, config_path: Path, backend: str) -> None:
    cfg = load_config(config_path)
    detector = build_detector(backend=backend, weights=cfg["model"]["weights"])
    rules = RuleEngine(cfg["zones"], cfg["policies"])
    reporter = BilingualReporter(cfg["reporter"])
    outputs = OutputDispatcher(cfg["outputs"])

    stats = {"frames": 0, "fps": 0.0, "events": 0}
    t0 = time.time()

    async for frame_idx, frame, detections in detector.stream(source):
        stats["frames"] += 1
        if frame_idx % 30 == 0:
            stats["fps"] = stats["frames"] / max(time.time() - t0, 1e-6)
            console.log(
                f"[dim]frame={frame_idx} fps={stats['fps']:.1f} "
                f"det={len(detections)}[/dim]"
            )

        events = rules.evaluate(frame_idx, detections)
        for event in events:
            stats["events"] += 1
            console.rule(f"[red]EVENT[/red] #{stats['events']} — {event.kind}")
            console.print(event.summary())
            report = await reporter.describe(event, frame)
            console.print(report.pretty())
            await outputs.push(event, report, snapshot=frame)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SentinelEdge inference loop")
    p.add_argument("--source", required=True, help="RTSP URL or local file")
    p.add_argument("--config", type=Path, default=Path("configs/site_example.yaml"))
    p.add_argument("--backend", choices=["torch", "hailo"], default="torch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console.print(
        f"[bold cyan]SentinelEdge[/bold cyan] starting — backend=[green]{args.backend}[/green]"
    )
    asyncio.run(run(args.source, args.config, args.backend))


if __name__ == "__main__":
    main()
