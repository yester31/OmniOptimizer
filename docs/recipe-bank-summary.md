# OmniOptimizer Recipe Bank — 전체 정리

**Last updated**: 2026-04-23 (Wave 15 ship 이후)
**Benchmark env**: YOLO26n · best_qr.pt · QR val (133 images) · RTX 3060 Laptop GPU (sm_86) + i7-11375H Tiger Lake · Windows 11 · TRT 10.16 + CUDA 12.9 + ORT 1.22
**Measurement protocol**: `scripts/measure.py::measure_latency` (warmup 100-200 / measure 100-300 iters · CUDA events + perf_counter percentiles)

> **Wave 15 (2026-04-23)** — audit-driven runtime tuning. 레시피 수/ 랭킹
> 변동 없음; runner 레벨 knob 조정만: OpenVINO `CACHE_DIR` persistent
> kernel cache + ORT TRT EP `trt_builder_optimization_level=5` / timing
> cache (backward-compat 폴백 포함) + `MeasurementSpec.build_ceiling_s`
> schema. D2 opt_level=5 opt-in to #09/#12/#42 는 측정 후 ROLLBACK
> (INT8 modelopt 은 opt_level=3 에서 이미 ceiling 근처). 자세한 내용은
> [`docs/improvements/2026-04-23-wave15-results.md`](improvements/2026-04-23-wave15-results.md).

## GPU 레시피 (23 active + 5 archived)

| ID | Recipe | Backend 도구 | Dtype | Quant 방식 / Calibrator | Structural | fps bs1 | mAP@0.5 | Status | 비고 |
|---:|---|---|---|---|---|---:|---:|:--:|---|
| 01 | `pytorch_fp32` | PyTorch eager | FP32 | — | — | 46.5 | 0.988 | ✔ | Baseline |
| 04 | `ort_trt_fp16` | ORT TRT EP | FP16 | — | — | 211.0 | 0.989 | ✔ | Wrapper overhead vs native TRT |
| 00 | `trt_fp32` | TensorRT 10.16 | FP32 | — | — | 356.5 | 0.989 | ✔ | |
| 00 | `trt_fp32_tf32` | TensorRT 10.16 | FP32 + TF32 | — | — | 346.6 | 0.989 | ✔ | `BuilderFlag.TF32` on Ampere |
| 05 | `trt_fp16` | TensorRT 10.16 | FP16 | — | — | 435.1 | 0.989 | ✔ | `BuilderFlag.FP16` |
| **40** | **`trt_fp16_opt5`** | **TensorRT 10.16** | **FP16** | **—** | **—** | **645.2** | **0.989** | ✔ | **Wave 14 · `builder_optimization_level=5`** |
| 41 | `trt_bf16` | TensorRT 10.16 | BF16 | — | — | 372.7 | 0.989 | ✔ | Wave 14 · `BuilderFlag.BF16` (sm_80+) |
| 06 | `trt_int8_ptq` | TensorRT 10.16 built-in | INT8 | TRT entropy calibrator (COCO 512) | — | 504.0 | 0.985 | ✔ | `config.int8_calibrator` |
| 07 | `trt_int8_sparsity` | TensorRT 10.16 built-in | INT8 | TRT entropy | 2:4 sparse (post-training) | 649.2 | 0.973 | ✘ | mAP drop 1.5%p (sparsity-aware training 필요) |
| 08 | `modelopt_int8_ptq` | modelopt.onnx.quantize | INT8 | max | — | 430.1 | 0.985 | ✔ | TRT nondeterminism 영향 (Wave 11 B4) |
| 09 | `modelopt_int8_entropy` | modelopt.onnx.quantize | INT8 | entropy (KL-div) | — | 763.9 | 0.987 | ✔ | Wave 14 이전 Top |
| 10 | `modelopt_int8_percentile` | modelopt.onnx.quantize | INT8 | percentile 99.99% | — | 755.1 | 0.985 | ✔ | |
| 11 | `modelopt_int8_sparsity` | modelopt.onnx.quantize | INT8 | entropy | 2:4 sparse + modelopt.sparsify | 439.4 | 0.987 | ✔ | `sparsity_preprocess: "2:4"` |
| 12 | `modelopt_int8_mixed` | modelopt.onnx.quantize | INT8 + FP16 mixed | mixed cfg | — | 760.0 | 0.987 | ✔ | `high_precision_dtype=fp16` |
| 17 | `modelopt_int8_qat` | modelopt.torch.quantization | INT8 QAT | QAT fine-tune | — | 425.8 | 0.985 | ✔ | `TrainingSpec.modifier=modelopt_qat` |
| **42** | **`modelopt_int8_asymmetric`** | **modelopt.onnx.quantize** | **INT8** | **entropy + `use_zero_point=True`** | **—** | **770.5** | **0.987** | ✔ | **Wave 14 · NEW TOP** |
| 13 | `ort_int8_minmax` | onnxruntime.quantization | INT8 QDQ | MinMax (sym activation) | — | 598.7 | 0.984 | ✔ | `QuantFormat.QDQ` + `ActivationSymmetric=True` |
| 14 | `ort_int8_entropy` | onnxruntime.quantization | INT8 QDQ | Entropy (128 samples) | — | 653.5 | 0.987 | ✔ | Histogram OOM cap at 128 |
| 15 | `ort_int8_percentile` | onnxruntime.quantization | INT8 QDQ | Percentile (128) | — | 674.5 | 0.988 | ✔ | |
| 16 | `ort_int8_distribution` | onnxruntime.quantization | INT8 QDQ | Distribution (128) | — | 402.5 | 0.987 | ✔ | |
| 23 | `modelopt_fastnas_int8` | modelopt FastNAS + modelopt.onnx.quantize | INT8 | entropy | **FastNAS 15.7% FLOPs prune** | 716.3 | 0.947 | ⊘ | 엔진 5MB (−88%), mAP −4%p trade-off |
| 24 | `modelopt_fastnas_sp_int8` | FastNAS + sparsify + quantize | INT8 | entropy | FastNAS + 2:4 sparse + FT | 697.4 | 0.948 | ⊘ | 엣지/VRAM 타겟 |
| — | `02_torchcompile_fp16` | torch.compile (inductor) | FP16 | — | — | — | — | ✘ ARCHIVED | Windows MSVC blocker (Wave 11) |
| — | `03_ort_cuda_fp16` | ORT CUDA EP | FP16 | — | — | — | — | ✘ ARCHIVED | CUDA EP NMS 미지원 (Wave 11) |
| — | `20/21/22 brevitas_*` | Brevitas + TRT | INT8 QDQ | MSE / Percentile | — | — | — | ✘ ARCHIVED | modelopt 대비 redundant |

**범례**: ✔ = active (constraints pass) · ✘ = excluded from ranking · ⊘ = parked or sparsity-training-dependent

## CPU 레시피 (8 active + 1 HW-gated)

| ID | Recipe | Backend 도구 | Dtype | Quant 방식 / Calibrator | Structural | fps bs1 | mAP@0.5 | Status | 비고 |
|---:|---|---|---|---|---|---:|---:|:--:|---|
| 30 | `ort_cpu_fp32` | ORT CPU EP | FP32 | — | — | 14.4 | 0.988 | ✔ | |
| 31 | `ort_cpu_bf16` | ORT CPU EP | BF16 | — | — | — | — | ⊘ | i7-11375H 에 AMX/AVX512_BF16 없음 (HW gate) |
| 32 | `ort_cpu_int8_dynamic` | ort.quantize_dynamic | INT8 | runtime per-batch | — | 10.0 | 0.982 | ✔ | QUInt8 + DynamicQuantizeLinear |
| 33 | `ort_cpu_int8_static` | ort.quantize_static | INT8 QDQ | entropy (sym) + `nodes_to_exclude=/model.23/` | — | 9.1 | 0.983 | ✔ | **Wave 11 fix**: Detect head 제외 · mAP 0.000→0.983 |
| 34 | `openvino_fp32` | OpenVINO 2026.1 | FP32 | — | — | 18.6 | 0.988 | ✔ | |
| 35 | `openvino_int8_nncf` | OpenVINO + NNCF | INT8 | NNCF MIXED preset | — | 23.9 | 0.988 | ✔ | 빠른 CPU + 고 mAP |
| 36 | `openvino_fastnas_int8_nncf` | OpenVINO + NNCF | INT8 | NNCF MIXED | FastNAS 15.7% prune | **32.2** | 0.947 | ✔ | **CPU Rank 1** |
| 37 | `openvino_fastnas_fp32` | OpenVINO | FP32 | — | FastNAS | 23.1 | 0.947 | ✔ | |
| 38 | `ort_cpu_fastnas_fp32` | ORT CPU EP | FP32 | — | FastNAS | 14.2 | 0.947 | ✔ | |

## Dtype × Backend 매트릭스

어떤 조합이 구현되어 있는지 한눈에 — 공란은 현재 scope 외.

| Dtype \ Backend | TensorRT native | ORT TRT EP | ORT CUDA EP | ORT CPU EP | OpenVINO | PyTorch | modelopt ONNX |
|---|---|---|---|---|---|---|---|
| **FP32** | #00, #00-tf32 | — | — | #30, #38 | #34, #37 | #01 | — |
| **FP16** | #05, **#40** opt5 | #04 | archived (#03) | — (x86 비지원) | — | archived (#02) | — |
| **BF16** | #41 | — | — | #31 (HW-gated) | — | — | — |
| **INT8 PTQ** | #06 | — | — | #32 dyn, #33 static | #35, #36 | — | #08–#12, #23–#24, **#42** |
| **INT8 QAT** | — | — | — | — | — | — | #17 |
| **INT8 QDQ** | — | — | — | — | — | — | #13–#16 (via ORT) |
| **2:4 Sparsity** | #07, #11, #24 | — | — | — | — | — | — |

## Top 1 per 축 (Best-of)

| 구분 | 1위 Recipe | fps bs1 | mAP@0.5 | 근거 |
|---|---|---:|---:|---|
| **GPU 전체** | `modelopt_int8_asymmetric` (#42) | **770.5** | 0.987 | Wave 14 ship, asymmetric INT8 + `use_zero_point=True` |
| **GPU edge (VRAM 제약)** | `modelopt_fastnas_int8` (#23) | 716.3 | 0.947 | 엔진 5MB (−88% vs 38MB) |
| **GPU 정확도 우선 (FP16)** | `trt_fp16_opt5` (#40) | 645.2 | 0.989 | opt_level=5, mAP 보존 |
| **CPU 전체** | `openvino_fastnas_int8_nncf` (#36) | **32.2** | 0.947 | OpenVINO + FastNAS + NNCF |
| **CPU 정확도 우선** | `openvino_int8_nncf` (#35) | 23.9 | 0.988 | |
| **CPU x86 INT8 (OpenVINO 없이)** | `ort_cpu_int8_dynamic` (#32) | 10.0 | 0.982 | DynamicQuantizeLinear |

## Methodology 축 별 비교

### Calibrator 영향 (modelopt + TRT, same ONNX graph, 223 Q/DQ)

| Calibrator | 활성화 scale max | 활성화 scale p99 | fps | Δ fps vs max |
|---|---:|---:|---:|---:|
| `max` (#08) | 1.190 | 0.667 | 430.1 | baseline |
| `entropy` (#09) | 0.662 | 0.339 | 763.9 | +78% |
| `percentile 99.99%` (#10) | 1.190 | 0.667 | 755.1 | +76% |

**해석** (Wave 11 B4 분석): 동일 ONNX 인데 fps 차이가 크게 나타남 → TRT 빌더 autotune nondeterminism. `builder_optimization_level=5` 로 stabilize 되면 격차 줄어듦.

### Quantization tool 선택 (INT8, imgsz=640, COCO 512 calib)

| Tool | API | Symmetric default | Asymmetric 지원 | Backend 호환 |
|---|---|---|---|---|
| `modelopt.onnx.quantize` | ONNX-level Q/DQ 주입 | Yes | `use_zero_point=True` | TRT, ORT |
| `onnxruntime.quantization.quantize_static` | ONNX-level Q/DQ 주입 | Configurable | `extra_options={"ActivationSymmetric": False}` | ORT CPU/CUDA, TRT |
| `onnxruntime.quantization.quantize_dynamic` | Weight-only quantize | Sym weight | — (runtime activation) | ORT CPU |
| `nncf.quantize` | OpenVINO IR 수준 | MIXED preset | preset-driven | OpenVINO only |
| TensorRT built-in calibrator | `config.int8_calibrator` | Entropy v2 (default) | — | TRT only |

### Structural modifications (pre-quantize)

| Modifier | Tool | FLOPs 감소 | mAP impact | 엔진 크기 |
|---|---|---:|---:|---:|
| 2:4 sparse (post-training) | TRT `SPARSE_WEIGHTS` | weight-only | −1.5%p | 동일 |
| 2:4 sparse + training | `modelopt.torch.sparsity` + FT | weight-only | 보존 | 동일 |
| FastNAS prune | `modelopt.torch.prune` FX trace | 15.7% | −4%p | **−88%** (38→5MB) |

## 전체 통계

- **Active recipes**: 31 (GPU 23 + CPU 8)
- **Archive**: 5 (#02, #03, #20-#22) — 문서 보존, `recipes/_archived/`
- **HW-gated skip**: 1 (#31 bf16)
- **총 구현 시도**: 37
- **Peak fps bs1**: 770.5 (#42 asymmetric INT8)
- **Peak mAP@0.5**: 0.989 (FP16/FP32 variants)
- **Smallest engine**: 5 MB (#23/#24 FastNAS, −88% vs 38 MB baseline)

## 구조적 설계

각 코드 경로는 단일 runner + recipe YAML 1:1 매핑:
- `scripts/run_pytorch.py` — PyTorch (#01, archived #02)
- `scripts/run_ort.py` — ORT TRT EP / ORT CUDA EP (#04, archived #03)
- `scripts/run_trt.py` — TensorRT native + modelopt (#00, #05-#17, #23-#24, #40-#42)
- `scripts/run_cpu.py` — ORT CPU EP + OpenVINO + NNCF (#30-#38)

`scripts/_schemas.py::Recipe` 가 계약. `Result` JSON 이 교환 포맷. `recommend.py` 가 랭킹.

## 참고 문서

- Architecture: [`docs/architecture.md`](architecture.md)
- 최근 shipped wave: [`docs/plans/_shipped/2026-04-21-wave10-modelopt-fastnas-pruning.md`](plans/_shipped/2026-04-21-wave10-modelopt-fastnas-pruning.md)
- Wave 11 cleanup: [`docs/plans/2026-04-22-wave11-recipe-debug-cleanup.md`](plans/2026-04-22-wave11-recipe-debug-cleanup.md)
- Wave 14 TRT tuning: [`docs/plans/2026-04-22-wave14-trt-optimization.md`](plans/2026-04-22-wave14-trt-optimization.md)
- Wave 15 audit-driven tuning: [`docs/plans/2026-04-23-wave15-audit-driven-tuning.md`](plans/2026-04-23-wave15-audit-driven-tuning.md) · [`docs/improvements/2026-04-23-wave15-results.md`](improvements/2026-04-23-wave15-results.md)
- Recipe audit: [`docs/improvements/2026-04-22-recipe-bank-audit.md`](improvements/2026-04-22-recipe-bank-audit.md)
- Live reports: [`report_qr.md`](../report_qr.md), [`report_cpu_qr.md`](../report_cpu_qr.md)
