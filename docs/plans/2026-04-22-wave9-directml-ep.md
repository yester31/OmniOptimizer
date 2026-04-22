# Wave 9: ONNX Runtime DirectML EP

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:executing-plans`. 체크박스(`- [ ]`)가 진행 트래커. Task 0 spike 통과 전에는 Task 1 이후로 진행하지 않는다.

**Goal:** Windows 네이티브 DirectML(DML) 백엔드로 **AMD GPU / Intel Arc / 모바일 iGPU / Windows NPU** 범위 확장. 현 recipe bank는 NVIDIA CUDA(TRT/ORT-CUDA) + Intel CPU(OV/ORT CPU) 뿐 — 다른 하드웨어 계층에서 OmniOptimizer가 돌아가는지 증명하는 wave.

**왜 이게 다음 wave인가**: Wave 7(PyTorch PT2E XNNPACK) / Wave 8(ncnn) 모두 **외부 변환기 체인**이 YOLO26n end-to-end NMS 경로에서 좌초. DML은 **ORT 네이티브**(공식 wheel, 외부 변환기 없음) 이므로 그 리스크 클래스 회피. Wave 10에서 확립한 `modelopt.onnx.quantize` QDQ 전략도 DML INT8 경로에 이식 가능성 있음.

**Architecture:** 변경 지점 5곳.
1. **venv 분리**: 현 `onnxruntime`(TRT/CUDA/CPU) 과 `onnxruntime-directml` 은 같은 모듈명으로 충돌 → `.venv_dml/` 별도 환경. `Makefile` 에 `dml-venv` 타깃 추가.
2. **`scripts/run_dml.py` 신규**: `scripts/run_cpu.py` 패턴 (InferenceSession + `measure_latency`) 복제. `session_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]`. Recipe `source: "ort_dml"`, `engine: "onnxruntime"`.
3. **Schema 확장**: `TechniqueSpec.source` Literal 에 `"ort_dml"` 추가. `RuntimeSpec.execution_provider` Optional str (이미 존재) 재사용.
4. **Recipe YAML × 2** (`#27`, `#28`).
5. **Report 재생성 대상**: 새 `report_dml_qr.md` 또는 기존 `report_qr.md` 통합 — Task 3 결과 보고 결정.

**Tech Stack:** `onnxruntime-directml >= 1.22`, `ultralytics == 8.4.27`, `onnx == 1.19.x`. Hardware: Windows 10/11 DirectX 12 GPU/NPU.

**중요 리스크 (착수 전 Task 0에서 확인)**:
- (R1) **패키지 충돌**: `onnxruntime` + `onnxruntime-directml` 동일 파이썬 모듈. `.venv_dml/` 분리 필수. 공유 `scripts/` import 정상 작동 확인.
- (R2) **Provider 감지**: 이 랩탑(RTX 3060 Laptop + Intel iGPU Xe)에서 DML provider가 어느 adapter를 잡는지. `get_available_providers()` + device info API로 판정. RTX/iGPU 선택 옵션 필요 시 `provider_options = [{"device_id": N}]`.
- (R3) **Dynamic shape 지원**: DML은 historically dynamic batch 불안정. bs=1 고정으로 시작, bs=8은 Task 0 Step 2에서 확인.
- (R4) **FP16 경로**: DML 에서 FP16 kernel 활성화는 ONNX 가 FP16 ops 포함해야 함. `ultralytics yolo.export(format="onnx", half=True)` 가 해당 ONNX 생성. 또는 modelopt.onnx.autocast 로 mixed precision.
- (R5) **INT8 경로**: DML INT8 quantization 지원은 제한적 — 공식 튜토리얼 부재. Task 4에서 탐색, 불가 시 `#29` reserved 처리.

---

## Scope Boundaries

### In-scope (Wave 9)
- **Backend**: ORT DirectML EP (`DmlExecutionProvider`)
- **Hardware**: Windows DirectX 12 GPU/NPU (RTX 3060 Laptop + Intel Xe iGPU 로 bench; 공식 타깃은 AMD/Intel/NPU 확장)
- **Recipe**: FP32, FP16 (2 recipes 우선)
- **Report**: 별도 `report_dml_qr.md` 또는 기존 report 통합 (Task 3 결정)

### Out-of-scope (후속 wave)
- **DML INT8**: 공식 지원 제한적. 별도 wave 에서 탐색
- **Linux DML**: `onnxruntime-directml` 는 Windows-only wheel
- **NPU-specific graph tuning**: QNN/OpenVINO NPU plugin 과 별개 — 본 wave 는 범용 DML
- **Multi-GPU**: 단일 device_id 만

### Assumptions
- DML provider 가 이 하드웨어에서 best_qr.pt 의 ONNX 를 accept. (확인: Task 0 Step 1)
- `modelopt.onnx.quantize` 로 주입한 QDQ ONNX 를 DML 이 **무시** 또는 INT8 kernel 로 실행. (확인: Task 4)
- Win32 subprocess 에서 `.venv_dml/` 의 python 호출 가능. `Makefile` 의 `dml-bench` 타깃 구현.

---

## Recipe Map (#27–#28)

| # | name | source | dtype | 비고 |
|---|---|---|---|---|
| 27 | `ort_dml_fp32` | ort_dml | fp32 | baseline — ORT CPU FP32 대비 DML gain 측정 |
| 28 | `ort_dml_fp16` | ort_dml | fp16 | ultralytics half=True ONNX → DML FP16 |

**Parked** (`#29` reserved):
- `ort_dml_int8` — modelopt.onnx.quantize QDQ ONNX → DML. INT8 kernel 실제 실행 여부 Task 4 에서 확인.

---

## Task Dependency Graph

```
Task 0 (env spike — BLOCKING gate)
   │
   ├─ Task 0 pass → Task 1 (run_dml.py) ─> Task 2 (schema + recipes)
   │                                           │
   │                                           v
   │                         Task 3 (make dml-bench + report)
   │                                           │
   │                                           v
   │                         Task 4 (optional DML INT8 exploration)
   │                                           │
   │                                           v
   │                         Task 5 (docs + CLAUDE.md)
   │
   └─ Task 0 fail → archive 또는 Linux/ROCm 탐색
```

---

## Task 0 — DML 환경 spike (BLOCKING)

**목표**: 별도 venv + `onnxruntime-directml` 패키지 + best_qr.pt ONNX 의 DML 추론이 작동하는지.

**산출물**: `.venv_dml/`, `scripts/_spike_wave9_r1_dml.py`, `docs/improvements/2026-04-22-wave9-r1-spike-results.md`.

### Step 1 — venv 분리

- [ ] `python -m venv .venv_dml` (현재 Python 3.13.3)
- [ ] `.venv_dml/Scripts/pip install onnxruntime-directml ultralytics==8.4.27 onnx numpy`
- [ ] `.venv_dml/Scripts/python -c "import onnxruntime as ort; print(ort.get_available_providers())"` → `DmlExecutionProvider` 확인
- [ ] 실패 시 대안: `pip install onnxruntime-directml==1.22.0` (현 1.22 matches 기존 venv)

### Step 2 — best_qr.pt ONNX 준비 + DML 추론

- [ ] 기존 `results/_onnx/best_qr_640_fp32_bs1.onnx` 재사용 (이미 export 완료)
- [ ] spike 스크립트 (`scripts/_spike_wave9_r1_dml.py`):
  ```python
  import numpy as np
  import onnxruntime as ort

  sess = ort.InferenceSession(
      "results/_onnx/best_qr_640_fp32_bs1.onnx",
      providers=["DmlExecutionProvider", "CPUExecutionProvider"],
  )
  print("active providers:", sess.get_providers())
  dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
  out = sess.run(None, {sess.get_inputs()[0].name: dummy})
  print("outputs:", [o.shape for o in out])
  ```
- [ ] 성공 시 warmup 100 + measure 100 (`scripts/measure.py::measure_latency` 재사용)
- [ ] RTX 3060 Laptop vs Intel iGPU 선택 실험 — `provider_options=[{"device_id": N}]` sweep
- [ ] **DML과 CPU provider fps 비교** — DML gain 확인

### Step 3 — FP16 ONNX 검증

- [ ] `ultralytics yolo.export(format="onnx", half=True, imgsz=640)` 로 FP16 ONNX 생성
- [ ] DML provider 로 로드 + warmup/measure 반복
- [ ] FP16 speedup 확인 (CUDA 에서는 ~2×, DML 에서 유사 여부)

### Step 4 — 결과 문서화

- [ ] `docs/improvements/2026-04-22-wave9-r1-spike-results.md`:
  - DML provider 감지 OK/FAIL
  - 선택된 device (RTX / iGPU)
  - fp32 / fp16 fps_bs1
  - warning / error 로그 (DML 버전 호환성)

**Gate**:
- DML provider 감지 + best_qr ONNX 추론 성공 → Task 1 진행
- FP16 ONNX 성공 → `#28` 유효
- INT8 (QDQ ONNX) 로드 확인 → `#29` unparked
- 모두 실패 → archive, 대안 wave (예: Linux ROCm)

---

## Task 1 — `scripts/run_dml.py` 구현

`scripts/run_cpu.py` 구조를 base 로. 핵심 차이:
- `session_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]`
- device_id 선택 (recipe `technique.device_id` 신규 필드 또는 `hardware.gpu` 재활용)
- `scripts/measure.py::measure_latency` warmup/measure 재사용 (CPU와 동일 프로토콜)

- [ ] `scripts/run_dml.py`:
  - `_prepare_dml_session(recipe)` — InferenceSession 생성 + provider_options
  - `run(recipe_path, out_path)` — CPU 패턴 그대로 복제 + DML provider
- [ ] CLI: `python scripts/run_dml.py --recipe recipes/27_*.yaml --out results_dml_qr/27_*.json`
- [ ] Result JSON 에 `env.execution_provider = "DmlExecutionProvider"` 기록
- [ ] `tests/test_run_dml_imports_without_directml.py` — CPU 전용 환경 (`.venv/`) 에서 이 모듈 import 안 되는지 (cross-venv 분리 검증)

---

## Task 2 — Schema + Recipe YAML

- [ ] `scripts/_schemas.py::TechniqueSpec.source` Literal 확장:
  ```python
  source: Literal[
      "trt_builtin", "modelopt", "ort_quant", "brevitas",
      "ort_cpu", "openvino", "ort_dml",  # new
  ] = "trt_builtin"
  ```
- [ ] `recipes/27_ort_dml_fp32.yaml`:
  ```yaml
  name: ort_dml_fp32
  model: {family: yolo26, variant: n, weights: best_qr.pt}
  runtime: {engine: onnxruntime, dtype: fp32, execution_provider: DmlExecutionProvider}
  technique: {name: fp32, source: ort_dml}
  measurement: {dataset: qr_barcode, num_images: 133, warmup_iters: 200,
                measure_iters: 300, batch_sizes: [1], input_size: 640, seed: 42}
  constraints: {max_map_drop_pct: 1.0}
  ```
- [ ] `recipes/28_ort_dml_fp16.yaml` — `runtime.dtype: fp16`
- [ ] `tests/test_schemas.py` round-trip 추가

---

## Task 3 — Makefile + E2E bench + report

- [ ] `Makefile`:
  ```makefile
  dml-venv:
      python -m venv .venv_dml
      .venv_dml/Scripts/pip install onnxruntime-directml ultralytics==8.4.27 onnx numpy pydantic pyyaml

  results_dml_qr/%.json:
      .venv_dml/Scripts/python scripts/run_dml.py --recipe recipes/$*.yaml --out $@

  dml-bench: results_dml_qr/27_ort_dml_fp32.json results_dml_qr/28_ort_dml_fp16.json

  report_dml_qr.md:
      python scripts/recommend.py --results-dir results_dml_qr/ --out report_dml_qr.md
  ```
- [ ] `make dml-bench && make report_dml_qr.md`
- [ ] `report_dml_qr.md` 검토 — DML fps vs CPU vs CUDA 비교. 유의미하면 `report_qr.md` 통합.

---

## Task 4 (조건부) — DML INT8 탐색

Task 0 Step 2 에서 DML provider 가 QDQ ONNX 를 accept 했다면:

- [ ] `recipes/29_ort_dml_int8.yaml` — Wave 10 의 `modelopt.onnx.quantize` QDQ ONNX 재사용 (best_qr_640_modelopt_entropy_bs1.onnx)
- [ ] `make recipe-29`
- [ ] 실측 fps + warning 수 → DML INT8 실제 이득 여부 확정

Accept 실패 시 `#29` reserved 유지 + 원인 문서화.

---

## Task 5 — 문서 갱신

- [ ] `docs/architecture.md` — "Backends" 섹션에 DML 추가
- [ ] `CLAUDE.md` — recipe 수 갱신, "Active GPU plan" 업데이트
- [ ] `README.md` — recipe 표 갱신
- [ ] 완료 커밋: `feat(wave9): ship ORT DirectML EP (#27-#28) — <결과 요약>`

---

## Definition of Done

1. `.venv_dml/` 구축 + `DmlExecutionProvider` import OK
2. `results_dml_qr/27_*.json`, `28_*.json` 2개 생성
3. `report_dml_qr.md` 생성 + `meets_constraints=True` at least 1
4. 이 plan DONE 스탬프 + spike 결과 문서
5. `CLAUDE.md` / `docs/architecture.md` 갱신

---

## Rollback

| 실패 시나리오 | 조치 |
|---|---|
| Task 0 Step 1 실패 (`onnxruntime-directml` 설치/import) | Python 3.13 호환 확인. 안되면 `.venv_dml/` 를 Python 3.11 로 재생성. 여전히 실패면 plan ARCHIVED, 별도 wave (Linux ROCm / Android QNN). |
| Step 2 실패 (DML provider 감지 X) | driver/DirectX 12 지원 여부 확인 (`dxdiag`). RTX 3060 Laptop 은 DX12 Ultimate 지원 — 발생 불가 예상 |
| Step 3 실패 (FP16 ONNX crash) | `#28` 만 parked, `#27` 만 ship |
| Task 4 실패 (DML INT8 무효) | `#29` reserved 로 유지, 별도 wave |
| Task 3 결과 CPU/CUDA 대비 이득 미미 | ship 하되 report 에 "DML는 AMD/Intel GPU + NPU 커버용 — 본 하드웨어에서는 marginal" 명시 |

---

## 후속 Wave 후보 (Wave 9 완료 후)

- **Wave 11 — QNN / Android NPU**: Snapdragon NPU bench (Qualcomm QNN delegate)
- **Wave 12 rev — INT4 NVFP4 on Blackwell**: FP8/NVFP4 HW 확보 시
- **Wave 14 — AMD ROCm (Linux)**: 별도 환경에서 MIGraphX / ORT-ROCm
- **Wave 15 — macOS CoreML**: Apple Silicon NPU 확장

---

## 참고 링크

- [ONNX Runtime DirectML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [onnxruntime-directml PyPI](https://pypi.org/project/onnxruntime-directml/)
- Wave 7/8 archive (외부 변환기 좌초) — `docs/improvements/2026-04-21-wave{7,8}-*-spike-results.md`
- Wave 10 QDQ 전략 (재사용 가능) — `docs/improvements/2026-04-22-wave10-pruning-extended-eval.md`
