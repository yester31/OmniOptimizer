# Wave 9 Task 0 spike — ORT DirectML EP 환경 검증

**Date**: 2026-04-22
**Status**: 환경 셋업 **PASS**. 성능은 현 하드웨어(RTX 3060 Laptop + Intel Xe iGPU)에서 **CPU 이하**.
**Decision**: Wave 9 ship **park** 권고. 실측 가치는 AMD Radeon / Intel Arc / Qualcomm NPU 같은 **전용 DML 타깃 하드웨어** 확보 시 재오픈.

## TL;DR

- `.venv_dml/` 구축 OK (Python 3.13.3 + `onnxruntime-directml 1.24.4`)
- `DmlExecutionProvider` 감지 OK
- best_qr FP32 ONNX 로드 + 추론 OK
- **device_id sweep 결과: 모든 DML adapter가 CPU의 40% fps**

| run | device | p50 ms | fps_bs1 | 비교 |
|---|---|---:|---:|---|
| dml_fp32 (device 0) | RTX 3060 Laptop | 131.86 | **7.6** | CPU의 40% |
| dml_fp32 (device 1) | Intel Xe iGPU | 130.47 | **7.7** | CPU의 40% |
| dml_fp16 (device 0) | RTX 3060 Laptop | 112.19 | **8.9** | 소폭 개선, 여전히 CPU 이하 |
| cpu_fp32 (ort CPU) | Intel i7-11375H | 52.29 | **19.1** | reference |
| **reference** `trt_fp16` | RTX 3060 CUDA | 2.30 | 435.1 | CUDA native로 **57×** 빠름 |
| **reference** `ort_cpu_fp32` | Intel CPU | 69.46 | 14.4 | `scripts/run_cpu.py` 프로토콜 |

## 왜 DML 이 이 하드웨어에서 느린가

1. **DirectML은 DirectX 12 HLSL shader 기반** — NVIDIA GPU 에서는 cuDNN/TRT(CUDA-native, PTX-compiled) 가 압도적. DML 의 target은 **AMD Radeon / Intel Arc / Qualcomm NPU** 처럼 CUDA 비호환 하드웨어.
2. **YOLO26n의 end-to-end NMS** (TopK/GatherND/NonZero) 는 DML 에서 일부 op fallback → CPU-GPU 전송 비용.
3. **Intel Xe iGPU (Tiger Lake)는 GPU 컴퓨트 유닛 수가 제한적** — 저사양 iGPU 로는 CPU VNNI 대비 이득 없음.

## Wave 9 plan 재평가

| Plan 목표 | 이 랩탑 실측 | 평가 |
|---|---|---|
| Windows DirectX 12 GPU 커버 | RTX 3060: fps 7.6 | ✗ 이득 없음 |
| Intel Arc / iGPU 커버 | Intel Xe iGPU: fps 7.7 | ✗ 이 iGPU 는 CPU 이하 |
| NPU 확장 | 이 랩탑에 NPU 없음 | ? |
| 외부 컨버터 없이 ORT-native | provider 로드 OK | ✓ 유일 장점 |

**유일한 positive signal**: 환경 분리(`.venv_dml/`) + provider 감지 + ONNX 추론 path 모두 작동. Wave 7/8 의 외부 컨버터 블로커 클래스와 달리 **인프라 레벨 리스크는 없음**.

하지만 ship 비용(Task 1-5 = 8~10h: `scripts/run_dml.py` + recipes + Makefile + report + docs) 대비 **이 하드웨어에서 얻을 실측값이 전부 "DML fps < CPU fps"** → meets_constraints=False 만 양산할 것.

## 결정: Park (not archive)

Wave 12/13 처럼 pre-execution archive 가 아니라 **park**:

- **Plan 문서**는 `docs/plans/2026-04-22-wave9-directml-ep.md` 그대로 유지 (reopen-ready)
- **venv_dml + spike 스크립트**는 commit 해서 재현 가능 유지
- **Recipe / run_dml.py / report** 는 만들지 않음 (ship 비용 회피)
- **재오픈 조건**: 다음 중 하나
  - Qualcomm NPU (Snapdragon X) 탑재 노트북
  - AMD Radeon dGPU (RDNA3+)
  - Intel Arc dGPU (A380 이상)

이 지점 오면 Task 1-5 수행해서 기존 21 GPU + 9 CPU recipe 에 DML 3~5개 추가.

## 산출물

- `.venv_dml/` — `onnxruntime-directml 1.24.4` (gitignore — 재생성 비용 ~30초)
- `scripts/_spike_wave9_r1_dml.py` — provider 감지 + FP32/FP16 bench
- `scripts/_spike_wave9_r2_device_sweep.py` — device_id sweep
- `logs/wave9_r1_dml.json` — 측정 데이터

## 참고

- Wave 9 plan — `docs/plans/2026-04-22-wave9-directml-ep.md` (park)
- ORT DirectML EP docs — <https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html>
- DML 성능 특성 해설 (Microsoft blog) — <https://devblogs.microsoft.com/directx/directml/>

## 다음 Wave 후보

park 후 다음 방향:
1. **Wave 11 QNN / Snapdragon** — Qualcomm NPU delegate (전용 HW 확보 시)
2. **Wave 14 Linux ROCm** — AMD GPU 공식 support (WSL2 or bare-metal)
3. **Wave 15 CoreML** — Apple Silicon NPU
