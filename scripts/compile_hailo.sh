#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#  compile_hailo.sh — reference Stage II compile flow for SentinelEdge.
#
#  Prerequisites (installed on Stage II workstation only):
#    - Hailo Dataflow Compiler (DFC) v4.x
#    - Hailo Model Zoo (optional; shortcut for pre-shipped configs)
#    - A calibration set — 500 held-out images per model, as .npy / folder
#
#  This script is *reference flow*, not expected to run on a laptop without
#  the Hailo SDK. Including it in the repo documents the intended pipeline
#  for Stage II reviewers.
# -----------------------------------------------------------------------------
set -euo pipefail

ARTIFACTS="${ARTIFACTS:-artifacts}"
HW_ARCH="${HW_ARCH:-hailo10h}"
CALIB_DIR="${CALIB_DIR:-calibration/images}"

mkdir -p "${ARTIFACTS}"

# ---------- Detector: YOLOv8-n ----------
hailo parser onnx \
    "${ARTIFACTS}/detector.onnx" \
    --hw-arch "${HW_ARCH}" \
    --output-dir "${ARTIFACTS}"

hailo optimize \
    "${ARTIFACTS}/detector.har" \
    --calib-set-path "${CALIB_DIR}" \
    --use-random-calib-set-fallback \
    --hw-arch "${HW_ARCH}" \
    --output-dir "${ARTIFACTS}"

hailo compiler \
    "${ARTIFACTS}/detector_optimized.har" \
    --hw-arch "${HW_ARCH}" \
    --output-dir "${ARTIFACTS}"

# ---------- Pose: YOLOv8-Pose-n ----------
hailo parser onnx \
    "${ARTIFACTS}/pose.onnx" \
    --hw-arch "${HW_ARCH}" \
    --output-dir "${ARTIFACTS}"

hailo optimize \
    "${ARTIFACTS}/pose.har" \
    --calib-set-path "${CALIB_DIR}" \
    --use-random-calib-set-fallback \
    --hw-arch "${HW_ARCH}" \
    --output-dir "${ARTIFACTS}"

hailo compiler \
    "${ARTIFACTS}/pose_optimized.har" \
    --hw-arch "${HW_ARCH}" \
    --output-dir "${ARTIFACTS}"

echo "[compile_hailo] artifacts/detector.hef and artifacts/pose.hef ready."
echo "[compile_hailo] next: hailortcli run --measure-latency --power-mode performance"
