#!/usr/bin/env bash
# Run all active recipes against the QR/Barcode fine-tuned checkpoint.
# Produces results_qr/*.json, leaves results/ (COCO baseline) untouched.
set -u

export OMNI_COCO_YAML="$PWD/qr_barcode.yaml"
export OMNI_CALIB_YAML="$PWD/coco_val_only.yaml"    # calib on COCO (QR val has only 133 imgs)
export OMNI_WEIGHTS_OVERRIDE="$PWD/best_qr.pt"

mkdir -p results_qr

declare -A RECIPES=(
    [00_trt_fp32]="run_trt"
    [00_trt_fp32_tf32]="run_trt"
    [01_pytorch_fp32]="run_pytorch"
    [02_torchcompile_fp16]="run_pytorch"
    [03_ort_cuda_fp16]="run_ort"
    [04_ort_trt_fp16]="run_ort"
    [05_trt_fp16]="run_trt"
    [06_trt_int8_ptq]="run_trt"
    [08_modelopt_int8_ptq]="run_trt"
    [09_modelopt_int8_entropy]="run_trt"
    [10_modelopt_int8_percentile]="run_trt"
    [12_modelopt_int8_mixed]="run_trt"
    [13_ort_int8_minmax]="run_trt"
    [14_ort_int8_entropy]="run_trt"
    [15_ort_int8_percentile]="run_trt"
    [16_ort_int8_distribution]="run_trt"
    [20_brevitas_int8_percentile]="run_trt"
    [21_brevitas_int8_mse]="run_trt"
)

ORDER=(
    00_trt_fp32 00_trt_fp32_tf32
    01_pytorch_fp32 02_torchcompile_fp16
    03_ort_cuda_fp16 04_ort_trt_fp16
    05_trt_fp16 06_trt_int8_ptq
    08_modelopt_int8_ptq 09_modelopt_int8_entropy
    10_modelopt_int8_percentile 12_modelopt_int8_mixed
    13_ort_int8_minmax 14_ort_int8_entropy
    15_ort_int8_percentile 16_ort_int8_distribution
    20_brevitas_int8_percentile 21_brevitas_int8_mse
)

for r in "${ORDER[@]}"; do
    runner="${RECIPES[$r]}"
    out="results_qr/${r}.json"
    if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        continue
    fi
    echo "=== [$(date +%H:%M:%S)] running $r via $runner ==="
    python "scripts/${runner}.py" --recipe "recipes/${r}.yaml" --out "$out" \
        2>&1 | tail -3
    echo "--- exit=$? ---"
done

echo "all done."
