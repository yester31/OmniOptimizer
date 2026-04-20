#!/usr/bin/env bash
# Train QAT / sparsity recipes with the QR/Barcode fine-tuned checkpoint.
# Produces trained_weights/*.pt, skips existing.
set -u

export OMNI_COCO_YAML="$PWD/qr_barcode.yaml"
export OMNI_WEIGHTS_OVERRIDE="$PWD/best_qr.pt"

mkdir -p trained_weights

TRAINING_RECIPES=(
    07_trt_int8_sparsity
    11_modelopt_int8_sparsity
    17_modelopt_int8_qat
)

for r in "${TRAINING_RECIPES[@]}"; do
    out="trained_weights/${r}.pt"
    if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        continue
    fi
    echo "=== [$(date +%H:%M:%S)] training $r ==="
    python scripts/train.py --recipe "recipes/${r}.yaml"
    ec=$?
    echo "--- exit=$ec ---"
done

echo "all training done."
