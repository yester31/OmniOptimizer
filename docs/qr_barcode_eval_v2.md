# QR/Barcode Evaluation v2 — Overlap-free Calibration

`docs/qr_barcode_eval.md` 의 **v2** 갱신판. v1 대비 변경점은 **INT8 캘리브레이션
데이터를 COCO val2017 로 전환**한 것 하나. QR val 은 133장 뿐이고 그중 512장 샘플링은
불가능했어서 실제로는 133장 전체가 calib = eval 로 이중 사용되어 mAP 가
**overlap-biased** 되어 있었음.

이 문서는 그 편향을 수치로 기록합니다.

## 설정 변경

| 항목 | v1 | v2 |
|---|---|---|
| Calibration 데이터셋 | `qr_barcode.yaml` (QR val 133장) | **`coco_val_only.yaml` (COCO val2017 512장, seed=42)** |
| Evaluation 데이터셋 | `qr_barcode.yaml` (QR val 133장) | 동일 |
| Calib/eval 데이터셋 동일? | **예** (overlap = 100%) | **아니오** (disjoint, overlap = 0) |
| `OMNI_CALIB_YAML` 환경변수 | 미지원 | 도입 |
| Split 적용? | — | 데이터셋 다르므로 split 불필요 |

Infra 변경사항은 `scripts/_split.py` (신규), `scripts/eval_coco.py`,
`scripts/run_{trt,ort,pytorch}.py`, `scripts/run_qr_batch.sh`.

## Full results (v2)

baseline: `00 trt_fp32` mAP@50=0.9893, mAP@50-95=0.9328.

| # | Recipe | p50 ms | bs1 fps | bs8 fps | mAP@50 | ΔmAP@50 (%p) | mAP@50-95 | ΔmAP@50-95 (%p) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 00 | trt_fp32 | 2.80 | 356.5 | 527.1 | 0.9893 | +0.00 | 0.9328 | +0.00 |
| 00-tf32 | trt_fp32_tf32 | 2.89 | 346.6 | 467.6 | 0.9893 | +0.00 | 0.9328 | +0.00 |
| 01 | pytorch_fp32 | 21.50 | 46.5 | 213.4 | 0.9876 | -0.17 | 0.9354 | +0.26 |
| 02 | torchcompile_fp16 | 25.62 | 39.0 | 312.1 | 0.9876 | -0.17 | 0.9360 | +0.32 |
| 03 | ort_cuda_fp16 | 314.61 | 3.2 | — | 0.9893 | +0.00 | 0.9334 | +0.06 |
| 04 | ort_trt_fp16 | 4.74 | 211.0 | 399.3 | 0.9893 | +0.00 | 0.9334 | +0.06 |
| 05 | trt_fp16 | 2.30 | 435.1 | 864.4 | 0.9892 | -0.01 | 0.9338 | +0.10 |
| 06 | trt_int8_ptq | 1.98 | 504.0 | 726.7 | 0.9847 | -0.46 | 0.8889 | -4.39 |
| 08 | modelopt_int8_ptq | 2.33 | 430.1 | 832.9 | 0.9852 | -0.41 | 0.9006 | -3.22 |
| **09** | **modelopt_int8_entropy** | **1.31** | **763.9** | **1078.5** | **0.9869** | **-0.24** | **0.9152** | **-1.76** |
| 10 | modelopt_int8_percentile | 1.32 | 755.1 | 993.7 | 0.9852 | -0.41 | 0.9006 | -3.22 |
| 12 | modelopt_int8_mixed | 1.32 | 760.0 | 840.8 | 0.9869 | -0.24 | 0.9152 | -1.76 |
| 13 | ort_int8_minmax | 1.67 | 598.7 | 698.6 | 0.9843 | -0.50 | 0.8873 | -4.55 |
| 14 | ort_int8_entropy | 1.53 | 653.5 | 697.9 | 0.9874 | -0.19 | 0.9068 | -2.60 |
| 15 | ort_int8_percentile | 1.48 | 674.5 | 694.4 | 0.9878 | -0.15 | 0.9067 | -2.61 |
| 16 | ort_int8_distribution | 2.48 | 402.5 | 696.7 | 0.9874 | -0.19 | 0.9068 | -2.60 |
| 20 | brevitas_int8_percentile | 2.49 | 401.2 | 679.1 | 0.9877 | -0.16 | 0.9228 | -1.00 |
| 21 | brevitas_int8_mse | 2.52 | 396.9 | 675.1 | 0.9879 | -0.14 | 0.9212 | -1.16 |

## v1 vs v2 delta — overlap bias 크기

mAP 는 overlap 이 제거되면서 대부분 **하락** (FP32/FP16 은 calib 없으므로 변화 없음,
단 INT8 만 변동). mAP@50-95 에서 편향이 가장 뚜렷하게 드러남.

| # | Recipe | v1 mAP@50-95 | v2 mAP@50-95 | Δ (%p) |
|---|---|---:|---:|---:|
| 06 | trt_int8_ptq | 0.8889 | 0.8889 | 0.00 (seed calib 무관) |
| 08 | modelopt_int8_ptq | 0.8987 | 0.9006 | +0.19 |
| 09 | modelopt_int8_entropy | 0.9222 | 0.9152 | **-0.70** |
| 10 | modelopt_int8_percentile | 0.8987 | 0.9006 | +0.19 |
| 12 | modelopt_int8_mixed | 0.9222 | 0.9152 | **-0.70** |
| 13 | ort_int8_minmax | 0.8869 | 0.8873 | +0.04 |
| 14 | ort_int8_entropy | 0.8903 | 0.9068 | **+1.65** |
| 15 | ort_int8_percentile | 0.8953 | 0.9067 | **+1.14** |
| 16 | ort_int8_distribution | 0.8913 | 0.9068 | **+1.55** |
| 20 | brevitas_int8_percentile | 0.9278 | 0.9228 | **-0.50** |
| 21 | brevitas_int8_mse | 0.9219 | 0.9212 | -0.07 |

**해석:**
- **modelopt entropy/mixed 계열**은 v1 에서 overlap 덕분에 ~0.7%p **과대평가** 돼 있었음.
- **ORT entropy/percentile/distribution 계열**은 반대로 1-1.6%p **과소평가**. 이는 calib 데이터
  양의 함수 — v1 QR calib 은 133장, v2 COCO calib 은 512장. 더 많은 calib 이 histogram
  기반 알고리즘에 유리.
- **Brevitas percentile**은 -0.50%p 하락. Brevitas 는 class 수에 덜 민감함을 시사.
- **trt_int8_ptq** 와 **modelopt_ptq/percentile** 는 calib 인스턴스보다 scale 공식에 더 의존
  (min-max / absmax) 하는 기법이라 변화 미미.

## 새 Recommendation

| 제약 | v2 추천 |
|---|---|
| 최대 fps, mAP 허용치 ≤ 2%p drop | **#09 modelopt_int8_entropy** (bs1 763.9, ΔmAP@50-95 -1.76%p) |
| 최소 mAP drop INT8 | **#20 brevitas_int8_percentile** (bs1 401, ΔmAP@50-95 -1.00%p) |
| 중간 타협 (속도+정확도) | **#12 modelopt_int8_mixed** (bs1 760, ΔmAP@50-95 -1.76%p, bs8 840) |
| FP16 sweet-spot | **#05 trt_fp16** (bs1 435, bs8 864, ΔmAP@50-95 +0.10%p) |

v1 에서의 winner (#09) 는 유지되지만, **mAP drop 의 실제 크기는 v1 보고치(-1.06%p)
보다 거의 두 배 큰 -1.76%p**. v1 문서의 "1%p drop 허용" 제약으로는 #09 이 원래
실격이었어야 함.

## Reproduction (v2)

```bash
cp "C:/Users/yeste/OneDrive/Desktop/QR_Barcode/QR_Barcode_detection/yolo26n_qrcode_barcode_bg/weights/best.pt" ./best_qr.pt
bash scripts/run_qr_batch.sh
```

환경변수 (스크립트 내부):
- `OMNI_WEIGHTS_OVERRIDE=./best_qr.pt` — fine-tuned 체크포인트
- `OMNI_COCO_YAML=./qr_barcode.yaml` — 평가 데이터셋
- `OMNI_CALIB_YAML=./coco_val_only.yaml` — **calib 만 COCO 로 분리** (v2 핵심)
- (opt-out) `OMNI_DISABLE_CALIB_EVAL_SPLIT=1` — 같은 데이터셋 split 비활성

## v1 의 문제점 회고

1. QR val 은 133장이고 calibration_samples=512 설정이었음에도 resolver 가 `paths[:n_samples]`
   로 cap → 실제 calib 은 133장 전체, eval 도 133장 전체 (100% overlap).
2. 이 상태에서 histogram 기반 INT8 calibrator 는 eval 이미지로 직접 학습한 셈이라
   **mAP 가 낙관적**으로 측정됨.
3. calibration_samples 가 val 크기보다 클 때 자동 대응 없음 (현재는 warn 만 출력).
4. 근본 해결: calibration source 를 명시적으로 분리 (`OMNI_CALIB_YAML`).

## Follow-ups

- `#03 ort_cuda_fp16` anomaly (314ms p50) v1/v2 모두 재현 — best.pt 의 특정 op 에서 CUDA EP 실행 경로 회귀.
- COCO baseline (`results/`) 도 같은 split 로직으로 재측정 필요 (별도 배치 실행 중).
- v1 보고서 (`docs/qr_barcode_eval.md`) 는 **deprecated** 표시하고 이 문서를 정본으로 사용.
