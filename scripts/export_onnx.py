"""Export the lightweight detector and pose model to ONNX for the Hailo pipeline.

Produces artifacts/{detector,pose}.onnx. The Hailo Dataflow Compiler
(installed on the Stage II workstation, not required locally) consumes
these as input via:

    hailo parser onnx artifacts/detector.onnx -y
    hailo optimize detector.har --use-random-calib-set
    hailo compiler detector_optimized.har --hw-arch hailo10h

See `scripts/compile_hailo.sh` for the full Stage II compile flow.

Usage:
    python scripts/export_onnx.py              # both models, default resolution
    python scripts/export_onnx.py --only pose  # pose only
"""
from __future__ import annotations

import argparse
from pathlib import Path


MODELS: dict[str, dict] = {
    "detector": {
        "weights": "yolov8n.pt",
        "imgsz": 640,
    },
    "pose": {
        "weights": "yolov8n-pose.pt",
        "imgsz": 640,
    },
}


def export(name: str, out_dir: Path, imgsz: int | None = None, opset: int = 13) -> Path:
    from ultralytics import YOLO

    spec = MODELS[name]
    weights = spec["weights"]
    size = imgsz or spec["imgsz"]

    print(f"[export] {name}: weights={weights} imgsz={size} opset={opset}")
    model = YOLO(weights)
    export_path = model.export(
        format="onnx",
        imgsz=size,
        opset=opset,
        simplify=True,
        dynamic=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{name}.onnx"
    Path(export_path).rename(final)
    print(f"[export] wrote {final}")
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(MODELS), default=None)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    out_dir = Path(args.out)
    targets = [args.only] if args.only else list(MODELS)
    for name in targets:
        export(name, out_dir, imgsz=args.imgsz, opset=args.opset)


if __name__ == "__main__":
    main()
