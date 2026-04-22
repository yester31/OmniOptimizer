#!/usr/bin/env bash
# Re-run all active recipes against the default yolo26n.pt / COCO baseline
# with the deterministic calib/eval split enabled. Overwrites results/.
set -u

export OMNI_COCO_YAML="$PWD/coco_val_only.yaml"
unset OMNI_CALIB_YAML         # same dataset -> split auto-applied
unset OMNI_WEIGHTS_OVERRIDE    # default yolo26n.pt from each recipe

mkdir -p results

ORDER=(
    00_trt_fp32 00_trt_fp32_tf32
    01_pytorch_fp32
    04_ort_trt_fp16
    05_trt_fp16 06_trt_int8_ptq
    08_modelopt_int8_ptq 09_modelopt_int8_entropy
    10_modelopt_int8_percentile 12_modelopt_int8_mixed
    13_ort_int8_minmax 14_ort_int8_entropy
    15_ort_int8_percentile 16_ort_int8_distribution
)

declare -A RUNNER=(
    [00_trt_fp32]=run_trt [00_trt_fp32_tf32]=run_trt
    [01_pytorch_fp32]=run_pytorch
    [04_ort_trt_fp16]=run_ort
    [05_trt_fp16]=run_trt [06_trt_int8_ptq]=run_trt
    [08_modelopt_int8_ptq]=run_trt [09_modelopt_int8_entropy]=run_trt
    [10_modelopt_int8_percentile]=run_trt [12_modelopt_int8_mixed]=run_trt
    [13_ort_int8_minmax]=run_trt [14_ort_int8_entropy]=run_trt
    [15_ort_int8_percentile]=run_trt [16_ort_int8_distribution]=run_trt
)

for r in "${ORDER[@]}"; do
    runner="${RUNNER[$r]}"
    out="results/${r}.json"
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
