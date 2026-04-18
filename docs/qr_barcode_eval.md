# QR/Barcode Fine-tuned Checkpoint Evaluation

OmniOptimizer 전체 레시피 뱅크를 **QR/Barcode 전용 fine-tuned 체크포인트**로 재평가한
결과입니다. 기본 YOLO26n (COCO) 평가와 별도로 보관됩니다.

## Setup

| 항목 | 값 |
|---|---|
| 체크포인트 | `yolo26n_qrcode_barcode_bg/weights/best.pt` (→ `./best_qr.pt`) |
| 데이터셋 | `E:/datasets/QR_BARCODE/sample_dataset_5000_refined_sampled_labelme_yolo_dataset2_bg` |
| 클래스 | 2 (barcode, qrcode) |
| 학습/검증/테스트 이미지 | 1068 / 133 / (test 미사용) |
| GPU | NVIDIA GeForce RTX 3060 Laptop (8.6) |
| TensorRT | 10.16.0.72 |
| 평가 일시 | 2026-04-18 |
| 평가 대상 | 활성 레시피 18개 (#07/#11/#19/#22 파킹) |
| 실행 방법 | `bash scripts/run_qr_batch.sh` — `OMNI_WEIGHTS_OVERRIDE`/`OMNI_COCO_YAML` 로 override |

## Baseline reference

TensorRT FP32 (#00) 기준:
- mAP@50 = **0.9893**
- mAP@50-95 = **0.9328**
- p50 latency = 2.69 ms, bs1 fps = 372

아래 표의 mAP drop 은 이 baseline 대비 %p 값입니다.

## Full results

지연은 ms, throughput 은 img/s. mAP 는 ultralytics `model.val` 결과.

| # | Recipe | p50 ms | p95 ms | bs1 fps | bs8 fps | mAP@50 | ΔmAP@50 | mAP@50-95 | ΔmAP@50-95 | peak MB | cold ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | trt_fp32 | 2.69 | 2.80 | 372.1 | 527.0 | 0.9893 | +0.00 | 0.9328 | +0.00 | 38.1 | 268 |
| 00-tf32 | trt_fp32_tf32 | 2.74 | 2.91 | 365.3 | 412.0 | 0.9893 | +0.00 | 0.9328 | +0.00 | 38.1 | 275 |
| 01 | pytorch_fp32 | 14.52 | 18.51 | 68.9 | 218.1 | 0.9876 | -0.17 | 0.9354 | +0.26 | 280.4 | 1772 |
| 02 | torchcompile_fp16 | 17.81 | 22.90 | 56.1 | 313.8 | 0.9876 | -0.17 | 0.9360 | +0.32 | 160.2 | 1811 |
| 03 | ort_cuda_fp16 | 270.64 | 402.09 | 3.7 | — | 0.9893 | +0.00 | 0.9334 | +0.06 | — | 469 |
| 04 | ort_trt_fp16 | 4.56 | 5.42 | 219.3 | 404.4 | 0.9893 | +0.00 | 0.9334 | +0.06 | — | 136533 |
| 05 | trt_fp16 | 2.25 | 2.35 | 445.3 | 868.8 | 0.9892 | -0.01 | 0.9338 | +0.10 | 38.1 | 169 |
| 06 | trt_int8_ptq | **1.33** | 1.42 | **753.0** | 741.8 | 0.9847 | -0.46 | 0.8889 | -4.39 | 38.1 | 154 |
| 08 | modelopt_int8_ptq | 1.70 | 1.84 | 586.8 | 841.8 | 0.9857 | -0.36 | 0.8987 | -3.41 | 38.1 | 242 |
| **09** | **modelopt_int8_entropy** | **1.34** | 1.64 | 746.4 | **842.8** | **0.9885** | **-0.08** | **0.9222** | **-1.06** | 38.1 | 219 |
| 10 | modelopt_int8_percentile | 1.38 | 1.72 | 725.7 | 841.2 | 0.9857 | -0.36 | 0.8987 | -3.41 | 38.1 | 232 |
| 12 | modelopt_int8_mixed | 1.76 | 2.16 | 568.9 | 837.2 | 0.9885 | -0.08 | 0.9222 | -1.06 | 38.1 | 387 |
| 13 | ort_int8_minmax | 1.89 | 2.01 | 529.5 | 699.0 | 0.9857 | -0.36 | 0.8869 | -4.59 | 38.1 | 270 |
| 14 | ort_int8_entropy | 1.45 | 1.78 | 691.6 | 700.4 | 0.9849 | -0.44 | 0.8903 | -4.25 | 38.1 | 407 |
| 15 | ort_int8_percentile | 2.53 | 2.67 | 395.7 | 697.1 | 0.9867 | -0.26 | 0.8953 | -3.75 | 38.1 | 289 |
| 16 | ort_int8_distribution | 1.87 | 2.47 | 535.4 | 700.2 | 0.9851 | -0.42 | 0.8913 | -4.15 | 38.1 | 295 |
| 20 | brevitas_int8_percentile | 2.50 | 2.60 | 400.6 | 685.8 | 0.9888 | -0.05 | 0.9278 | -0.50 | 90.5 | 248 |
| 21 | brevitas_int8_mse | 2.53 | 2.71 | 394.9 | 680.6 | 0.9885 | -0.08 | 0.9219 | -1.09 | 90.9 | 310 |

## Key observations

1. **추천 승자 — `#09 modelopt_int8_entropy`**. bs1 746 fps (FP32 대비 **2.0×**), mAP@50-95
   손실은 **-1.06%p** 뿐. 정확도/속도 파레토 프론트에 있음. `#12 modelopt_int8_mixed` 도
   동일 mAP 지만 p50 1.76ms 로 더 느림.
2. **정확도 최우선이면 `#20 brevitas_int8_percentile`**. mAP@50-95 손실이 **-0.50%p** 로
   INT8 중 최소. 처리량은 400 fps 로 TRT 자체 INT8 의 절반이지만 FP32 보다 여전히 빠름.
3. **`#06 trt_int8_ptq` 의 양면성** — 최고 bs1 fps (753) 이지만 mAP@50-95 손실 **-4.39%p**.
   클래스 수가 적은 QR 데이터셋에서 entropy calibrator 없는 기본 PTQ 는 손실이 큼.
4. **`#03 ort_cuda_fp16` 이상치** — p50 270ms, bs1 3.7 fps. onnxruntime-gpu CUDA EP 가
   fine-tuned 체크포인트의 특정 그래프를 제대로 최적화하지 못한 것으로 보이며,
   `meets_constraints=False`. 기본 YOLO26n(COCO) 에서는 관측되지 않던 회귀. 조사 필요.
5. **`#04 ort_trt_fp16` cold_start 136s** — onnxruntime 의 TRT EP 가 첫 호출 시
   engine 을 처음부터 빌드해 측정 시간이 큼. 정상적이며 캐시 후 2 회차부터는 짧아짐.
6. **FP16 vs FP32 mAP 가 사실상 동일** (`#05 trt_fp16` mAP@50-95 = 0.9338 vs 0.9328).
   QR/Barcode 는 low-dynamic-range 라 FP16 손실이 거의 없음.
7. **Brevitas 두 variant 간 차이 확인됨** — percentile mAP@50-95 0.9278, mse 0.9219.
   COCO 재평가 결과(0.5392 vs 0.5425) 와 역전 방향이라 calibrator 선호는 데이터셋에
   의존적임을 시사.

## Recommendation

| 제약 | 추천 |
|---|---|
| 최대 fps, mAP 허용치 ≤ 1%p drop | **#09 modelopt_int8_entropy** (bs1 746, ΔmAP@50-95 -1.06%p) |
| 최소 mAP drop INT8 | **#20 brevitas_int8_percentile** (bs1 400, ΔmAP@50-95 -0.50%p) |
| FP16 sweet-spot | **#05 trt_fp16** (bs1 445, ΔmAP@50-95 +0.10%p, bs8 868 최고) |
| 정확도 무손실 | **#00 trt_fp32** (bs1 372, baseline) |

## Follow-ups

- `#03 ort_cuda_fp16` 270ms 이상치 원인 조사 (fine-tuned 그래프의 어떤 op 이 CUDA EP 에서 느려지는지).
- Brevitas 의 COCO-vs-QR 역전 현상 — per-tensor 스케일이 narrow-class 데이터셋에서 어떻게 작동하는지 추가 실험.
- 파킹된 sparsity 레시피 (#07/#11) 는 QR 데이터셋으로 학습 파이프라인 붙인 후 재평가 예정.

## Reproduction

```bash
# 1. 체크포인트를 프로젝트 루트에 복사
cp "C:/Users/yeste/OneDrive/Desktop/QR_Barcode/QR_Barcode_detection/yolo26n_qrcode_barcode_bg/weights/best.pt" ./best_qr.pt

# 2. 데이터 yaml 은 qr_barcode.yaml 사용 (루트에 있음)

# 3. 배치 실행
bash scripts/run_qr_batch.sh

# 4. 결과
ls results_qr/
```

환경변수:
- `OMNI_WEIGHTS_OVERRIDE=./best_qr.pt` — 레시피의 `model.weights` 를 덮어씀
- `OMNI_COCO_YAML=./qr_barcode.yaml` — 검증/캘리브레이션 데이터셋 yaml (이름만 coco 일 뿐 데이터셋은 일반 ultralytics yaml 지원)
