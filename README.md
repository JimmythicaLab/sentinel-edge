# SentinelEdge

Edge industrial safety analytics on ASUS UGen300 / Hailo-10H.

![Status](https://img.shields.io/badge/stage-ASUS%20UGen%20AI%20League%202026-blue)
![Track](https://img.shields.io/badge/track-Battlefield%20Lightning-yellow)
![Theme](https://img.shields.io/badge/theme-Workplace%20AI-orange)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

SentinelEdge is a reference application for the ASUS UGen300 USB AI Accelerator (UGen300-USB-8G, Hailo-10H NPU, 40 TOPS @ INT4 / 20 TOPS @ INT8, 2.5 W, LPDDR4 8GB). It reads RTSP camera feeds, runs PPE / zone-rule detection on the NPU, and uses a small on-device LLM to produce bilingual (Traditional Chinese + English) incident reports. Everything runs on-prem; no video leaves the LAN.

Submission for the **2026 ASUS UGen AI League — Battlefield Lightning × Workplace AI**.

## Scope

- **Covered:** PPE compliance detection, zone rules with polygon geometry, bilingual incident reporting over MQTT / LINE / Teams.
- **Out of scope (for now):** full video retention UI, user management, cloud sync. Deliberate omissions — the attack surface and operational cost stay low.

## Architecture

```
         ┌── Host mini-PC (Intel N100 / RK3588 class) ──┐
         │                                                │
RTSP ──▶ │  FFmpeg decode → frame batcher                │
         │             │                                  │
         │             ▼                                  │
         │   ┌──────────────────────┐                     │
         │   │   UGen300 USB (Type-C)│  40/20 TOPS @ 2.5W │
         │   │   Hailo-10H NPU      │                     │
         │   │   • Object detector  │                     │
         │   │   • Pose estimator   │                     │
         │   └──────────┬───────────┘                     │
         │              ▼                                  │
         │   Rule / zone engine (Python)                  │
         │              │ event                            │
         │              ▼                                  │
         │   Qwen2.5-0.5B INT8 via llama.cpp (host CPU)   │
         │              │                                  │
         │              ▼                                  │
         │   MQTT · LINE · Teams · webhook · siren · log  │
         └────────────────────────────────────────────────┘
```

Alert path: frame → event in ~80 ms (Stage II target). Written report in 1–2 s via LLM. The alert is dispatched independently of the report — human notification does not wait on text generation.

## Repository layout

```
.
├── deck/
│   ├── deck_zh.html        # 12-page submission deck (source)
│   └── deck_zh.pdf         # Rendered PDF for upload
├── src/
│   ├── inference.py        # End-to-end runtime loop
│   ├── detectors.py        # Torch / Hailo detector interface
│   ├── rule_engine.py      # Polygon zones, dwell counters, PPE rules
│   ├── reporter.py         # Local bilingual LLM reporter (template fallback)
│   └── outputs.py          # MQTT / LINE / Teams sinks
├── scripts/
│   ├── bench_torch.py      # Torch backend latency / FPS benchmark
│   ├── export_onnx.py      # YOLOv8 → ONNX (feeds the Hailo compile)
│   └── compile_hailo.sh    # Reference Hailo Dataflow Compiler flow
├── tests/
│   ├── test_rule_engine.py # Polygon + dwell + helmet-overlap tests
│   └── test_reporter.py    # Template-fallback smoke tests
├── configs/
│   └── site_example.yaml   # Per-site config (zones, rules, outputs)
├── benchmark.md            # Public baselines + Stage II target table
└── requirements.txt
```

## Quickstart

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Stage I dev loop

The Torch backend works on any laptop and is enough for the Stage I demo.

```bash
# RTSP source
python -m src.inference \
    --source rtsp://<your-cam> \
    --config configs/site_example.yaml \
    --backend torch

# Or a local video file for offline demo recording
python -m src.inference \
    --source demo_video.mp4 \
    --config configs/site_example.yaml \
    --backend torch
```

For the LLM report path, drop a `qwen2.5-0.5b.gguf` under a path referenced in `configs/site_example.yaml` (`reporter.model_path`). If no model file is present the reporter falls back to a deterministic template, which is enough for CI and the Stage I demo recording.

### Benchmark the Torch backend

```bash
python scripts/bench_torch.py --iters 30 --device auto
# → results/bench_torch.json
```

### Export ONNX for the Hailo compile (Stage II prep)

```bash
python scripts/export_onnx.py
# → artifacts/detector.onnx, artifacts/pose.onnx
```

### Reference Hailo compile flow (Stage II workstation only)

See [`scripts/compile_hailo.sh`](scripts/compile_hailo.sh). Runs:

```
hailo parser onnx → hailo optimize (INT8 calibration) → hailo compiler → .hef
```

Not expected to run locally — included to document the intended Stage II pipeline for evaluators who want to see the full compile path.

### Run the test suite

```bash
pytest tests/ -v
```

## Benchmarks

Public FP32 baselines plus Stage II INT8-on-Hailo-10H targets tracked in [`benchmark.md`](benchmark.md). Nothing in this repo is currently claimed as measured on real UGen300 hardware; those numbers are published in Stage II alongside HEF artifacts.

## Roadmap

- **Oct 14, 2026** — Stage I submission bundle (this repo + deck + demo video).
- **Nov 6 – Dec 18, 2026** — Stage II on real hardware: Hailo HEF compile, INT8 accuracy on CHV / SH17 / Pictor-v3, multi-stream throughput, LLM report path, MQTT / LINE sinks.
- **Dec 19, 2026** — Finals demo at ASUS HQ, Taipei.

## Team

- **Lead / AI** — ML background covering training, quantization, ONNX / OpenVINO / Hailo toolchains.

Solo entry is allowed under competition rules (team cap 5). Stage II hiring from the League cohort is part of the plan.

## Licensing

Code: Apache 2.0. Model weights follow upstream licences:
- Ultralytics YOLOv8 — AGPL-3.0 (commercial path available for productization)
- Qwen2.5-0.5B — Apache 2.0
- llama.cpp — MIT
