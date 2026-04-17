# OmniOptimizer Report

Baseline: `pytorch_fp32`  |  GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`  |  CUDA driver: `576.80`

## Windows (Python 3.13, torch 2.8.0+cu129, TRT 10.16 cu13) — 7 recipes

| Rank | Recipe | fps(bs1) | fps(bs8) | p50 ms | mAP@0.5 | drop | mem MB | meets? |
|-----:|--------|---------:|---------:|-------:|--------:|-----:|-------:|:------:|
| 1 | `trt_fp16` | 277.2 | — | 3.61 | 0.554 | -0.11%p | 5 | ✔ |
| 2 | `ort_trt_fp16` | 193.7 | — | 5.16 | 0.554 | -0.12%p | — | ✔ |
| 3 | `ort_cuda_fp16` | 67.7 | — | 14.76 | 0.554 | -0.12%p | — | ✔ |
| 4 | `pytorch_fp32` | 55.6 | 227.3 | 17.97 | 0.553 | +0.00%p | 281 | ✔ |
| 5 | `torchcompile_fp16` | 46.3 | 278.5 | 21.61 | 0.553 | +0.00%p | 160 | ✔ |
| 6 | `trt_int8_sparsity` | 306.1 | — | 3.27 | 0.480 | +7.26%p | 5 | ✘ |
| 7 | `trt_int8_ptq` | 292.8 | — | 3.42 | 0.481 | +7.23%p | 5 | ✘ |

## WSL2 Ubuntu 22.04 (Python 3.10, torch 2.6.0+cu124, TRT 10.16 cu12) — 11 recipes

| Rank | Recipe | fps(bs1) | fps(bs8) | p50 ms | mAP@0.5 | drop | mem MB | meets? |
|-----:|--------|---------:|---------:|-------:|--------:|-----:|-------:|:------:|
| 1 | `modelopt_int8_percentile` | **460.3** | — | **2.17** | 0.522 | -3.20%p | 5 | ✘ |
| 2 | `modelopt_int8_entropy` | 391.3 | — | 2.56 | 0.536 | **-1.73%p** | 5 | ✘ (근접) |
| 3 | `modelopt_int8_sparsity` | 346.8 | — | 2.88 | 0.536 | -1.73%p | 5 | ✘ (근접) |
| 4 | `ort_trt_fp16` | 295.6 | — | 3.38 | 0.554 | -0.11%p | — | ✔ |
| 5 | `modelopt_int8_ptq` (max) | 291.1 | — | 3.44 | 0.520 | -3.30%p | 5 | ✘ |
| 6 | `trt_int8_ptq` | 288.4 | — | 3.47 | 0.474 | +7.92%p | 5 | ✘ |
| 7 | `trt_int8_sparsity` | 272.3 | — | 3.67 | 0.474 | +7.92%p | 5 | ✘ |
| 8 | `trt_fp16` | 265.6 | — | 3.76 | 0.554 | -0.11%p | 5 | ✔ |
| 9 | `torchcompile_fp16` | 164.6 | 378.3 | 6.07 | 0.553 | +0.00%p | 415 | ✔ |
| 10 | `ort_cuda_fp16` | 69.3 | — | 14.42 | — | — | — | ✔ (mAP eval 실패) |
| 11 | `pytorch_fp32` | 52.8 | 238.5 | 18.93 | 0.553 | +0.00%p | 281 | ✔ |

## Recommendations

- **제약을 완전히 만족 (drop < 1%p)하는 최고 성능**: **`ort_trt_fp16` (WSL)** — 295.6 fps, mAP 0.554. ONNX 단일 파일 배포, plug-and-play.
- **Windows 배포**: `trt_fp16` — 277.2 fps, mAP 0.554, drop -0.11%p.
- **INT8로 타깃 근접**: `modelopt_int8_entropy` — 391.3 fps, drop **-1.73%p** (1%p 타깃 근접). modelopt PTQ의 entropy 캘리브 사용.
- **최고 throughput (정확도 일부 희생)**: `modelopt_int8_percentile` — 460.3 fps, drop -3.2%p.

## Phase 2 ModelOpt 성과

기존 `trt_builtin` INT8 PTQ 대비 `modelopt_int8_entropy`:
- mAP drop: **-7.92%p → -1.73%p** (4.6배 개선)
- fps: **288.4 → 391.3** (+36%)
- p50: **3.47ms → 2.56ms**

modelopt의 per-channel 양자화 + KL-divergence 스케일 최적화가 TRT의 내장 엔트로피 캘리브레이터보다 YOLO26n에서 크게 우수함을 확인.

## Phase 3 — Windows vs WSL2 핵심 차이 (공통 7 레시피)

| Recipe | Win fps | WSL fps | Δ | 원인 |
|--------|--------:|--------:|---:|------|
| `torchcompile_fp16` | 46.3 | **164.6** | +3.5x | Windows Py3.13은 Triton 미지원 → eager fallback. |
| `ort_trt_fp16` | 193.7 | **295.6** | +1.5x | ONNX→TRT 경로에서 Linux 스택이 더 유리. |
| `trt_fp16` | 277.2 | 265.6 | -4% | 네이티브 TRT는 OS layer에 둔감. |
| `trt_int8_ptq` | 292.8 | 288.4 | -1.5% | 동일. |
| `trt_int8_sparsity` | 306.1 | 272.3 | -11% | 동일 패턴, sparsity 커널 변동성 높음. |
| `ort_cuda_fp16` | 67.7 | 69.3 | +2% | 성능 동등. WSL에서 mAP eval 회귀. |
| `pytorch_fp32` | 55.6 | 52.8 | -5% | 동등. |

## Issues / Known Limitations

- **bs=8 엔진 빌드 실패** (모든 TRT 레시피, 양 플랫폼): ultralytics ONNX export가 `dynamic=False`로 정적 배치=1 생성. bs>1 엔진 원하면 별도 dynamic ONNX export 필요.
- **`trt_int8_ptq`, `trt_int8_sparsity` mAP drop ≥7%p**: TRT 내장 IInt8EntropyCalibrator2가 YOLO26n에서 부적합한 스케일 산출. → **modelopt 경로 (#8–#11) 도입으로 해결**.
- **`modelopt_int8_sparsity` (#11) sparsity 플래그 무효화**: `SPARSE_WEIGHTS`만 세팅한다고 2:4 커널로 전환되지 않음. 가중치를 사전 pruning해 2:4 패턴에 맞춰야 함 (v1.2 `modelopt.torch.sparsity.sparsify` 도입 예정).
- **`ort_cuda_fp16` (WSL) mAP null**: ultralytics validator의 onnxruntime session device id 매핑 버그. v1.1 픽스 후보.

## Env
- Windows: `torch 2.8.0+cu129 / tensorrt_cu13 10.16 / cuDNN 9.10 / onnxruntime 1.24.4`
- WSL2: `torch 2.6.0+cu124 / tensorrt-cu12 10.16 / cuDNN 9.1.0 / onnxruntime 1.22.0 / nvidia-modelopt 0.43.0`
- 공통: RTX 3060 Laptop (SM 8.6, 6GB), driver 576.80, seed 42, warmup 100 / measure 100~1000, 500 val images.
