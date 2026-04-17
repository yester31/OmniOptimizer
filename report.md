# OmniOptimizer Report

Baseline: `pytorch_fp32`  |  GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`  |  CUDA driver: `576.80`
Environment: WSL2 Ubuntu 22.04, torch 2.6.0+cu124, TensorRT 10.16 (cu12), onnxruntime 1.22, nvidia-modelopt 0.43

## Full ranking (all 11 recipes, bs=1 fps descending)

| Rank | Recipe | fps(bs1) | fps(bs8) | p50 ms | mAP@0.5 | drop | meets? |
|-----:|--------|---------:|---------:|-------:|--------:|-----:|:------:|
| 1 | `modelopt_int8_percentile` | **436.0** | 675.0 | **2.29** | 0.520 | -3.32%p | ✘ |
| 2 | `modelopt_int8_ptq` (max) | 424.8 | 679.6 | 2.35 | 0.521 | -3.19%p | ✘ |
| 3 | `modelopt_int8_entropy` | 403.3 | 689.3 | 2.48 | **0.537** | **-1.63%p** | ✘ (near) |
| 4 | `trt_int8_sparsity` | 326.1 | 715.6 | 3.07 | 0.474 | +7.91%p | ✘ |
| 5 | `trt_int8_ptq` | 310.3 | 695.9 | 3.22 | 0.474 | +7.92%p | ✘ |
| 6 | `modelopt_int8_sparsity` | 271.1 | **825.5** | 3.69 | 0.537 | -1.62%p | ✘ (near) |
| 7 | `trt_fp16` | 238.9 | 496.4 | 4.19 | 0.554 | -0.11%p | ✔ |
| 8 | `ort_trt_fp16` | 202.9 | 382.2 | 4.93 | 0.554 | -0.11%p | ✔ |
| 9 | `torchcompile_fp16` | 164.6 | 378.3 | 6.07 | 0.553 | -0.01%p | ✔ |
| 10 | `ort_cuda_fp16` | 68.5 | 119.3 | 14.60 | 0.554 | -0.11%p | ✔ |
| 11 | `pytorch_fp32` | 52.8 | 238.5 | 18.93 | 0.553 | ±0.00%p | ✔ |

## Recommendations

- **1%p 제약 완전 만족 중 최고 속도**: **`trt_fp16`** — 238.9 fps bs=1, mAP 0.554, drop -0.11%p.
- **bs=8 최고 throughput**: **`modelopt_int8_sparsity`** — 825.5 fps bs=8 (타깃 근접 mAP drop -1.62%p).
- **bs=1 최고 throughput**: `modelopt_int8_percentile` — 436 fps, drop -3.32%p.
- **정확도/속도 균형**: `modelopt_int8_entropy` — 403 fps bs=1 / 689 fps bs=8, drop -1.63%p (1%p 타깃에 가장 근접한 INT8).

## Phase 2 ModelOpt 성과 (trt_builtin INT8 대비)

| 지표 | trt_int8_ptq | modelopt_int8_entropy | Δ |
|------|-------------:|---------------------:|---|
| fps bs1 | 310.3 | 403.3 | +30% |
| fps bs8 | 695.9 | 689.3 | -1% |
| p50 ms | 3.22 | 2.48 | -23% |
| mAP@0.5 | 0.474 | 0.537 | **+13.3%** |
| drop %p | +7.92 | **-1.63** | **4.9배 개선** |

modelopt의 per-channel 양자화 + KL-div 스케일 산출이 YOLO26n에서 TRT 내장 캘리브레이터를 압도함.

## Task 1 — bs=8 dynamic batch 지원

ultralytics ONNX export가 `dynamic=False`였어서 TRT 빌더가 bs>1 프로파일을 거부하던 문제 해결.

**per-bs ONNX 전략 채택**:
- bs=1 → static ONNX (`yolo26n_640_fp32_bs1.onnx`) — TRT가 공격적 fusion 가능
- bs>1 → dynamic ONNX (`yolo26n_640_fp32_dyn.onnx`) — 동일 파일로 여러 bs 커버
- ORT/TRT runner 모두 동일 전략 적용

영향: 모든 TRT/ORT 레시피의 **bs=8 fps가 새로 측정**됨. bs=1 성능 회귀 없음.

## Task 2 — mAP eval 회귀 픽스

WSL에서 `ort_cuda_fp16` / `trt_fp16`이 `accuracy eval failed: Invalid device id`로 mAP=null 이던 문제 해결.

**두 가지 원인 동시 수정**:
1. `YOLO.val(...)`에 `device=0` 명시 (ultralytics의 암묵적 디바이스 해석 실패)
2. TRT runner가 루프 마지막 `onnx_path.stem`으로 bs=1 엔진 찾아서 경로 미스매치 → `bs1_engine` 변수에 명시 저장

영향: 전 레시피 mAP 정상 측정.

## Known Limitations (남은 개선)

- **INT8 1%p 드랍 미달** (현재 -1.63%p): 샘플 수 ↑, 민감 레이어 제외, QAT 파인튜닝이 v1.2 범위.
- **진짜 2:4 Sparsity 미사용** (`modelopt_int8_sparsity` mAP가 `modelopt_int8_entropy`와 동일): `SPARSE_WEIGHTS` 플래그만으로는 커널 전환 안 됨. `modelopt.torch.sparsity.sparsify`로 가중치 사전 pruning 필요.
- **`trt_int8_sparsity`** (trt_builtin 경로): mAP drop 7.9%p, INT8 캘리브레이터 자체 한계.
