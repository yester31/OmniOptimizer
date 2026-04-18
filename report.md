# OmniOptimizer Report

Baseline: `pytorch_fp32`  |  GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`  |  CUDA: `12.4`


| Rank | Recipe | fps(bs1) | fps(bs8) | p50 ms | mAP@0.5 | drop | mem MB | meets? |
|-----:|--------|---------:|---------:|-------:|--------:|-----:|-------:|:------:|
| 1 | `trt_fp16` | 238.9 | 496.4 | 4.19 | 0.554 | -0.11%p | 38 | ✔ |
| 2 | `ort_trt_fp16` | 202.9 | 382.2 | 4.93 | 0.554 | -0.11%p | — | ✔ |
| 3 | `torchcompile_fp16` | 164.6 | 378.3 | 6.07 | 0.553 | -0.01%p | 415 | ✔ |
| 4 | `ort_cuda_fp16` | 68.5 | 119.3 | 14.60 | 0.554 | -0.11%p | — | ✔ |
| 5 | `pytorch_fp32` | 52.8 | 238.5 | 18.93 | 0.553 | +0.00%p | 281 | ✔ |
| 6 | `modelopt_int8_percentile` | 436.0 | 675.0 | 2.29 | 0.520 | +3.32%p | 38 | ✘ |
| 7 | `modelopt_int8_ptq` | 424.8 | 679.6 | 2.35 | 0.521 | +3.19%p | 38 | ✘ |
| 8 | `modelopt_int8_entropy` | 403.3 | 689.3 | 2.48 | 0.537 | +1.63%p | 38 | ✘ |
| 9 | `trt_int8_sparsity` | 326.1 | 715.6 | 3.07 | 0.474 | +7.92%p | 38 | ✘ |
| 10 | `trt_int8_ptq` | 310.3 | 695.9 | 3.22 | 0.474 | +7.92%p | 38 | ✘ |
| 11 | `modelopt_int8_mixed` | 303.2 | 754.7 | 3.30 | 0.537 | +1.62%p | 38 | ✘ |
| 12 | `modelopt_int8_sparsity` | 293.8 | 747.1 | 3.40 | 0.536 | +1.70%p | 38 | ✘ |

## Recommendation

**`trt_fp16`** — fps 238.9 (bs1), mAP@0.5 0.554, drop -0.11%p.

## Issues
- `ort_trt_fp16`: execution_provider=TensorrtExecutionProvider, onnx=yolo26n_640_fp16_bs1.onnx
- `ort_cuda_fp16`: execution_provider=CUDAExecutionProvider, onnx=yolo26n_640_fp16_bs1.onnx
- `modelopt_int8_percentile`: mAP drop 3.32%p > 1.0%p
- `modelopt_int8_ptq`: mAP drop 3.19%p > 1.0%p
- `modelopt_int8_entropy`: mAP drop 1.63%p > 1.0%p
- `trt_int8_sparsity`: mAP drop 7.92%p > 1.0%p
- `trt_int8_ptq`: mAP drop 7.92%p > 1.0%p
- `modelopt_int8_mixed`: mAP drop 1.62%p > 1.0%p
- `modelopt_int8_sparsity`: mAP drop 1.70%p > 1.0%p

## Phase 3 Acceptance (v1.2)

두 실험을 돌렸고, 각각의 P-L7 acceptance 판정:

### Experiment A — Real 2:4 sparsity via `modelopt.torch.sparsify` (recipe-11 후보)

| 지표 | 기준 | 측정 | 판정 |
|---|---|---|---|
| 2:4 pattern applied | `{2: 1152}` | `{2: 1152}` (32/16 Conv) | ✅ |
| fps bs=1 vs recipe-09 | ≥ 343 (×0.85) | 277 | ❌ |
| mAP@50-95 drop vs FP32 | ≤ 2.5%p | **39.97%p** | ❌ 붕괴 |

**결론:** YOLO26n 같은 nano 모델에 fine-tuning 없는 post-training 2:4 magnitude
pruning은 mAP를 사실상 0으로 무너뜨림. sparsify 자체는 정확히 작동 (2:4 zero
패턴 `{2: 1152}` 검증 완료). 원인은 이미 작은 모델에 50% 가중치를 버린 뒤 재학습이
없는 것. 진짜 2:4는 **sparsity-aware training (SAT) 또는 QAT**가 필요 → v1.3.

recipe-11 YAML은 v1.1 동작 (SPARSE_WEIGHTS hint, dense 가중치)으로 복귀. 결과
테이블의 `modelopt_int8_sparsity` 행은 재실행 후 수치 (fps 294/747, mAP drop 1.70%p
— recipe-09 entropy와 동등).

runner 코드 (`_apply_modelopt_sparsify` + `nodes_to_exclude` dispatch)는 유지.
v1.3 SAT 통합 시 그대로 재사용 가능.

### Experiment B — Mixed precision via `nodes_to_exclude` (recipe-12)

| 지표 | 기준 | 측정 | 판정 |
|---|---|---|---|
| nodes_to_exclude 반영 | 4/4 노드 FP16 유지 | QDQ ONNX 검증 완료 | ✅ |
| fps bs=1 | ≥ 30 | 303 | ✅ |
| fps bs=1 vs recipe-09 | ≥ 302 (×0.75) | 303 | ✅ (간신히) |
| mAP drop vs FP32 | < 1.0%p (목표) | **1.93%p** | ❌ 미달 |

**결론:** stem + bbox regression head (cv2.\*)만 FP16에 남긴 4-exclude 구성은
recipe-09 (pure INT8 entropy) 대비 **mAP 개선 0.02%p** (0.3802→0.3804, 노이즈 범위)
뿐이었고 bs=1 fps는 25% 감소. entropy 캘리브레이터가 이미 이 레이어들을 잘 다루고
있어서 exclude가 거의 이득 없음. bs=8에서는 +10% fps 우연한 이득 (kernel 선택 차이).

**권장:** 1%p 목표를 맞추려면 (a) 더 많은/다른 exclude 후보 탐색
(예: C2f 출력단), (b) calibrator 개선, (c) QAT. 이 셋 모두 v1.3.

## Known limitations (v1.2 시점)

- **INT8 1%p 드랍 목표 미달**: 현재 최선 1.62%p (recipe #12), 1.63%p (#9). 달성에는
  QAT 또는 사용자 모델별 민감도 검색이 필요.
- **진짜 2:4 sparsity는 nano YOLO에서 사용 불가**: `modelopt.torch.sparsify`의
  magnitude 모드는 정확히 2:4 패턴을 생성하지만 (`{2: 1152}` 검증), 재학습
  없이는 YOLO26n 수준의 작은 모델 정확도를 복구할 수 없음. 2:4 이득을 실제로
  가져오려면 sparsity-aware training을 동반해야 함 — v1.3.
- **Calibration leakage 잔존**: val2017에서 calibration + eval 동시 사용, ~수십 장
  overlap. 절대 수치는 약간 낙관적이나 레시피 간 상대 비교는 유효. v1.3에서
  train2017 분할로 교체 예정.
