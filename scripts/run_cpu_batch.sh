#!/usr/bin/env bash
# Run the Wave 6 CPU recipe bank (#30-#35) with the QR/Barcode fine-tuned
# checkpoint and emit results + a CPU-specific report. Mirrors the GPU
# batch flow but writes to results_cpu_qr/ so GPU and CPU runs never
# clobber each other's JSON.
#
# Usage:
#   bash scripts/run_cpu_batch.sh              # uses best_qr.pt + coco calibration
#   OMNI_WEIGHTS_OVERRIDE=... bash scripts/run_cpu_batch.sh   # custom checkpoint
set -u

export OMNI_COCO_YAML="${OMNI_COCO_YAML:-$PWD/qr_barcode.yaml}"
# QR val has 133 images — too small for INT8 calibration statistics.
# Use COCO val for calibration, QR for eval (same pattern Wave 5 used).
export OMNI_CALIB_YAML="${OMNI_CALIB_YAML:-$PWD/coco.yaml}"
export OMNI_WEIGHTS_OVERRIDE="${OMNI_WEIGHTS_OVERRIDE:-$PWD/best_qr.pt}"

RESULTS_DIR="results_cpu_qr"
REPORT="report_cpu_qr.md"
# openvino_int8_qat is a parked recipe slot (#36 planning) — exclude.
PARKED="openvino_int8_qat"

mkdir -p "$RESULTS_DIR"

CPU_RECIPES=(
    30_ort_cpu_fp32
    31_ort_cpu_bf16
    32_ort_cpu_int8_dynamic
    33_ort_cpu_int8_static
    34_openvino_fp32
    35_openvino_int8_nncf
)

for r in "${CPU_RECIPES[@]}"; do
    out="$RESULTS_DIR/${r}.json"
    if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        continue
    fi
    echo "=== [$(date +%H:%M:%S)] running $r ==="
    python scripts/run_cpu.py --recipe "recipes/${r}.yaml" --out "$out"
    ec=$?
    echo "--- exit=$ec ---"
done

echo "=== generating $REPORT ==="
python scripts/recommend.py \
    --results-dir "$RESULTS_DIR" \
    --out "$REPORT" \
    --exclude "$PARKED"

echo "all CPU recipes done; see $REPORT"
