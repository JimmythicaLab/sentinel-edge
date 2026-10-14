"""Detector backends.

Stage I ships the Torch backend (works on any laptop); Stage II adds the Hailo
backend that runs INT8-compiled HEFs on the UGen300.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None


class Detector:
    async def stream(
        self, source: str
    ) -> AsyncIterator[tuple[int, "object", list[Detection]]]:
        raise NotImplementedError


class TorchDetector(Detector):
    """PyTorch object-detector backend. Convenient for dev; not latency-optimised."""

    def __init__(self, weights: str) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)

    async def stream(self, source):  # type: ignore[override]
        results = self.model.track(
            source=source, stream=True, persist=True, verbose=False,
        )
        for i, r in enumerate(results):
            dets: list[Detection] = []
            if r.boxes is None:
                yield i, r.orig_img, dets
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item())
                dets.append(
                    Detection(
                        cls=self.model.names[cls_id],
                        conf=float(b.conf.item()),
                        xyxy=tuple(b.xyxy.cpu().numpy().flatten().tolist()),
                        track_id=int(b.id.item()) if b.id is not None else None,
                    )
                )
            yield i, r.orig_img, dets


class HailoDetector(Detector):
    """Hailo-10H / UGen300 USB AI Accelerator backend. Stage II.

    Loads a compiled HEF via HailoRT, binds VStreams, runs multi-stream
    scheduling. Interface is identical to TorchDetector so the pipeline is
    backend-agnostic — swapping backends does not touch rule_engine or
    reporter code.

    Target hardware: UGen300-USB-8G (Hailo-10H, 40 TOPS @ INT4 /
    20 TOPS @ INT8, LPDDR4 8GB, USB 3.1 Gen2 Type-C).
    """

    def __init__(self, weights: str) -> None:
        # TODO Stage II: initialise VDevice, ConfigureParams, scheduler
        raise NotImplementedError(
            "Hailo backend lands in Stage II after HailoRT SDK access."
        )

    async def stream(self, source):  # type: ignore[override]
        raise NotImplementedError


def build_detector(backend: str, weights: str) -> Detector:
    if backend == "torch":
        return TorchDetector(weights)
    if backend == "hailo":
        return HailoDetector(weights)
    raise ValueError(f"unknown backend: {backend}")
