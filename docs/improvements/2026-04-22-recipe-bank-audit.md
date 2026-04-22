# Recipe Bank Audit — 2026-04-22

**Status**: 조사 완료. 개선 경로 3개 Wave 로 분기 권고.
**Baseline**: report_qr.md (2026-04-22) + report_cpu_qr.md + 현재 30 active recipes.

## 요약

30 recipe 중 **5개 GPU / 2개 CPU** 에 성능 또는 정확도 문제 확인. 기존 bank 정돈 + TRT tuning 3 recipe 추가로 현재 Top rank (fps 763) 를 +5-15% 끌어올릴 여지 있음. FP8 / NVFP4 / INT4 는 보유 HW (RTX 3060 Laptop sm_86) 제약으로 park.

## 현재 상태

### GPU (21 run, report_qr.md)

| Rank | Recipe | fps bs1 | mAP | drop | 평가 |
|---:|---|---:|---:|---:|---|
| 1 | modelopt_int8_entropy | 763.9 | 0.987 | +0.07 | 기준 / Top |
| 2 | modelopt_int8_mixed | 760.0 | 0.987 | +0.07 | OK |
| 3 | modelopt_int8_percentile | 755.1 | 0.985 | +0.24 | OK |
| 4-5 | modelopt_fastnas_* | 697-716 | 0.947 | +4.0 | OK (Wave 10 edge tier) |
| 6-10 | ort_int8_* / trt_int8_* | 504-674 | 0.984-0.988 | +0.02-1.50 | OK |
| 11-17 | modelopt_* / trt_fp16/32 | 346-439 | 0.985-0.989 | | OK |
| **18** | **ort_trt_fp16** | **211** | 0.989 | | ORT wrapper overhead vs native TRT(435) |
| **19** | **pytorch_fp32** | 46.5 | 0.988 | baseline | OK |
| **20** | **torchcompile_fp16** | **39.0** | 0.988 | | **baseline보다 느림 — 버그 의심** |
| **21** | **ort_cuda_fp16** | **3.2** | 0.989 | | **100× 느림 — 명백한 misconfig** |

### CPU (9 run, report_cpu_qr.md)

| Rank | Recipe | fps bs1 | mAP | 평가 |
|---:|---|---:|---:|---|
| 1-7 | openvino / ort_cpu_fp* / ort_cpu_int8_dynamic | 10-32 | 0.947-0.988 | OK |
| **8** | **ort_cpu_int8_static** | 6.2 | **0.000** | **Broken — Wave 6 close-out 미완** |
| 9 | ort_cpu_bf16 | — | — | HW gate (정상 skip) |

## 문제 recipe 분석 (B-series)

| ID | Recipe | 증상 | 가설 |
|---|---|---|---|
| B1 | #21 ort_cuda_fp16 | fps 3.2 (100× 느림) | CUDAExecutionProvider 초기화 실패 → CPU fallback, 또는 cuDNN 버전 mismatch |
| B2 | #20 torchcompile_fp16 | fps 39 < pytorch_fp32 46.5 | torch.compile default mode (`"default"`) 가 inductor graph 재컴파일 반복. `mode="reduce-overhead"` 미적용 가능성 |
| B3 | #18 ort_trt_fp16 | fps 211 vs native 435 (48%) | TRT cache 미활성 + graph_optimization_level 기본값, ORT wrapper overhead |
| B4 | #13 modelopt_int8_ptq | fps 430 vs #1 entropy 763 (56%) | calibrator 차이가 TRT kernel tactic 선택 분기 — inspect 필요 |
| B5 | #33 ort_cpu_int8_static | mAP=0.000 | QDQ op coverage 부재 추정. Phase 3 에서 partial fix 됐으나 QR 데이터셋에서 재발 |

각 항목 debug 비용: S-M (1-2시간 / 항목).

## 추가 recipe 후보 (조사)

### High ROI (보유 HW 에서 즉시 가능)

| ID | 제안 | 기대 | 근거 |
|---|---|---|---|
| A1 | `trt_fp16_opt5` — `builder_optimization_level=5` | top-recipe fps +5-15% | TRT 10.x `nvinfer.BuilderConfig.builder_optimization_level` (0-5, default=3). 5 는 heavy autotune, build 3-5× 오래걸리지만 런타임 tactic 선택 최적화 |
| A2 | `trt_bf16` | mAP +0.05-0.1%p vs FP16 | Ampere sm_80+ 지원. BF16 dynamic range(FP32 동일) > FP16 — overflow 민감 weight 에 유리 |
| A3 | `pytorch_fp32_channels_last` | #19 fps 46.5 → 60-70 | NVIDIA PyTorch Memory Format: `tensor.to(memory_format=torch.channels_last)` — tensor core 활용 |
| A4 | `trt_fp16_imgsz{320,512}` variant | fps 2-4× / mAP −1~2%p | 엣지 배포 실용 variant. 현재 전부 640 |
| A5 | `trt_int8_entropy_asymmetric` | mAP +0.1-0.2%p | modelopt `INT8_SMOOTHQUANT_CFG` 또는 asymmetric calibrator |
| A6 | `*_bs16` / `*_bs32` extension | throughput 지표 확장 | 현재 bs=8 상한 |

### Park (HW / scope 제약)

| ID | 제안 | 제약 |
|---|---|---|
| C1 | FP8 (E4M3 / E5M2) | Hopper/Ada 필요 (RTX 3060 Laptop = Ampere sm_86, 불가) |
| C2 | NVFP4 | Blackwell 전용 (RTX 50+) |
| C3 | INT4 W4A16 | Wave 12 archived — WoQ GEMM-only, YOLO26n Conv-dominant |
| C4 | OpenVINO on Intel iGPU | Wave 9 DML 과 동일 타깃, DML fps < CPU 로 판명 |
| C5 | Knowledge Distillation (26l→26n) | teacher 학습 루프 필요, v2 scope |
| C6 | TVM auto-scheduling | 빌드 시간 수 시간, cross-platform 가치 낮음 |

### Backend matrix 공백

- `ort_cuda_int8` — TRT 없이 CUDA INT8 직접. 빌드 빠름, fps 낮음 (trt-native 대비). low priority.
- `openvino_bf16` — Intel Sapphire Rapids+ / Arrow Lake 필요. HW 없음.
- `ort_cpu_fp16` — ARMv8.2+ / Apple Silicon 전용. x86 호환 x.
- `pytorch_int8` (torch.ao) — ultralytics 미지원 경로.

## Wave 분기 권고

| Wave | 범위 | 비용 | 우선순위 |
|---|---|---|---|
| **11** Cleanup | B1-B5 debug (5개) + `test_recommend_ranking` + `test_recommend_exclude` + `test_recipe_smoke` | M | 1순위 |
| **14** TRT tuning | A1 + A2 + A5 (3 recipe 신규) | M | 2순위 |
| **15** Multi-resolution | A4 (imgsz 320/512 variant 3개) | S-M | 3순위 |
| **16** Memory format | A3 (channels_last for pytorch path) | S | 4순위 |
| park | C1-C6 | — | HW / scope 해결 시 reopen |

> **Note (post-audit 2026-04-22)**: 최초 Wave 11 제안에는 `Dockerfile` + GH Actions CI 포함이었으나, 후속 `/plan-eng-review` Step 0 에서 scope 축소 결정 (complexity 임계치 초과). Distribution 은 별도 Wave 에서 재개 가능.

## Wave 번호 정리

사용 중: 1-10 shipped, 12-13 archived pre-exec, 9 parked. 
예약 (CLAUDE.md 언급): 11 QNN, 14 ROCm, 15 CoreML.
**결정**: 이 문서의 권고 경로에서 "Wave 11" 은 QNN 과 scope 다름 → **Wave 11 을 Cleanup 에 재할당**. QNN 은 후속 Wave (17+) 로 이관.

## 참고

- 현재 reports — `report_qr.md`, `report_cpu_qr.md` (2026-04-22 regenerate)
- TRT BuilderConfig API — TensorRT 10.x Python API docs
- modelopt INT8 configs — `modelopt.torch.quantization` 및 `modelopt.onnx.quantization.quantize`
- 보유 HW — RTX 3060 Laptop GPU (sm_86, Ampere, 6 GB VRAM) + Intel i7-11375H (Tiger Lake AVX-512 VNNI, no AMX/BF16)
