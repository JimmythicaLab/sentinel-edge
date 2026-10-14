# SentinelEdge — Benchmark Reference & Stage II Plan

> **Stage I honesty note:** every number in this file that is not a *public literature baseline* is a **target to be measured in Stage II** on real UGen300 hardware, not a claim of measured performance. The Stage I prototype uses an open-source lightweight object detector; the production model choice is finalised in Stage II after INT8 drop measurement.

## 1. Public FP32 baselines (reference figures from third-party sources)

| Dataset   | Images  | Classes                | Reference model      | Approx. mAP@0.5  | Source                     |
|-----------|--------:|------------------------|----------------------|-----------------:|----------------------------|
| CHV       | ~1,300  | helmet / vest / person | lightweight detector | ~0.85 ± 0.05     | Roboflow Universe baseline |
| SH17      | ~8,000  | 17 PPE classes         | lightweight detector | ~0.75 ± 0.05     | Author's released weights  |
| Pictor-v3 | ~1,500  | construction PPE       | lightweight detector | ~0.70 ± 0.05     | Pictor benchmark           |
| COCO-Pose | 200,000 | 17 keypoints           | pose estimator       | ~0.55 ± 0.05 OKS | Ultralytics docs           |

> Baselines above are approximate figures from third-party sources. They have **not been reproduced on our pipeline yet**. Stage II work includes reproducing the FP32 baselines before measuring the INT8 drop.

## 2. Stage I measured (Torch backend, local hardware)

Run `scripts/bench_torch.py` to populate `results/bench_torch.json`. Numbers depend on the host machine; include the report in your submission if you want evaluators to see concrete FPS on reference hardware.

| Metric                                       | Where measured           | Value                                     |
|----------------------------------------------|--------------------------|-------------------------------------------|
| Object detector single-image latency, p50    | `scripts/bench_torch.py` | *(fill after running on your machine)*    |
| Object detector single-image latency, p95    | `scripts/bench_torch.py` | *(fill)*                                  |
| Throughput, single-stream                    | `scripts/bench_torch.py` | *(fill)*                                  |

## 3. Stage II target (INT8 on Hailo-10H / UGen300 USB AI Accelerator)

Reference figures from the official ASUS UGen300 datasheet (Hailo-10H) for common vision models at batch=1 INT8:

| Reference model          | Reported FPS | Source                 |
|--------------------------|-------------:|------------------------|
| ResNet-V1-50 (224²)      | 316          | ASUS UGen300 datasheet |
| SSD-MobileNet-V1 (300²)  | 501          | ASUS UGen300 datasheet |
| MobileNet-V2 (224²)      | 2944         | ASUS UGen300 datasheet |
| EfficientNet-M (240²)    | 508          | ASUS UGen300 datasheet |

These are the public reference numbers we calibrate our own object-detector targets against. Our targets below are intentionally conservative.

| Metric                                                | Target        | Why it matters                                 |
|-------------------------------------------------------|---------------|------------------------------------------------|
| Object detector @ 640² — single-stream latency        | ≤ 7 ms        | Enables ≥140 FPS single-stream on the NPU      |
| Object detector @ 640² — mAP@0.5 drop vs FP32         | ≤ 2 points    | Within industry acceptable quantization budget |
| Pose estimator @ 640² — single-stream latency         | ≤ 11 ms       | ≥90 FPS target                                 |
| Multi-model residency (detector + pose on chip)       | both resident | Prerequisite for one-dongle multi-stream       |
| Chip power envelope under full load                   | ≤ 2.5 W       | Fits product thermal budget                    |

Multi-stream scaling is deliberately not listed as a numeric target at Stage I; the Stage II deliverable publishes a scaling curve rather than a single number.

## 4. Measurement protocol (Stage II)

1. Compile each model to HEF via `scripts/compile_hailo.sh` using Hailo Dataflow Compiler v4.x, with calibration on 500 held-out images per model.
2. Baseline FP32 vs INT8 mAP on each dataset's standard split.
3. `hailortcli benchmark --measure-latency` for per-model latency distribution (p50 / p95 / p99).
4. End-to-end latency measured on the full pipeline (RTSP → det → rules → report) and reported per event.
5. Multi-stream scaling: run with 1 / 2 / 4 / 8 concurrent RTSP streams and record the per-stream FPS curve.
6. All raw results published under `results/` as JSON + Markdown summary.

## 5. Current status (Stage I)

- [x] Dataset survey and candidate model selection (open-source lightweight detector for Stage I prototyping).
- [x] Pipeline skeleton runs on Torch backend (`src/inference.py`).
- [x] Rule engine with polygon zones, dwell counters, helmet / vest / lone-worker / restricted-access rules (`src/rule_engine.py`).
- [x] Reporter template-fallback path (`src/reporter.py`); LLM path tested without live weights.
- [x] Benchmark harness (`scripts/bench_torch.py`), ONNX export (`scripts/export_onnx.py`), Hailo reference compile flow (`scripts/compile_hailo.sh`).
- [x] Unit tests (`pytest tests/`), 8 tests passing.
- [ ] Pose estimation integration — **Stage II**.
- [ ] Hailo HEF compile — pending UGen300 unit arrival (**Stage II**).
- [ ] INT8 accuracy table filled with measured values (**Stage II**).
- [ ] Multi-stream scaling curve (**Stage II**).
