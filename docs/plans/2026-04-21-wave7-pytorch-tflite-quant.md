# Wave 7: PyTorch PT2E 양자화 + TensorFlow Lite 양자화

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (권장) 또는 superpowers:executing-plans. Steps are TDD-style (`- [ ]`) — 체크박스가 진행 트래커.

**Goal:** Wave 6이 ORT CPU EP + OpenVINO로 CPU를 추가했다면, Wave 7은 **PyTorch 네이티브(pt2e) + TensorFlow Lite** 두 가지 "vendor-neutral" 양자화 경로를 붙인다. 같은 YOLO26n + 같은 호스트에서 4종 CPU backend(ORT/OV/PyTorch/TFLite)를 나란히 비교해, 배포 파이프라인 선택 자유도를 올린다.

**Architecture:** 변경 지점 5곳. (1) `scripts/_schemas.py::TechniqueSpec.source` Literal에 `pt2e`, `tflite` 추가. (2) `scripts/run_cpu.py`에 PyTorch PT2E(#40/#41)와 TFLite(#42–#44) 디스패처 브랜치 + 2개의 ORT-호환 어댑터(`PytorchPT2EAsORT`, `TFLiteAsORT`). (3) `recipes/40–44_*.yaml` 신규. (4) `pyproject.toml`에 `tflite-runtime`(가능한 OS에서만), `onnx2tf` extras. (5) `Makefile`·`docs`·README 타겟·테이블 갱신. `run_trt.py`/`run_ort.py`/기존 Wave 6 코드는 **건드리지 않는다** — 새 경로가 `_prepare_cpu_session`의 branch로 깔끔히 붙는다.

**Tech Stack:** `torch>=2.8` (`torch.export` + `torch.ao.quantization.quantize_pt2e` + `X86InductorQuantizer`, 이미 설치), `onnx2tf>=1.22` (신규, ONNX→TFLite 변환기), `tflite-runtime` 또는 `tensorflow>=2.15`의 Interpreter (신규; Windows는 제약 있음 — Risk R1 참고), `numpy`, `ultralytics`, `pydantic`. Wave 6 OVRunnerAsORT 패턴을 재사용.

---

## Scope Boundaries

### In-scope (Wave 7)
- **하드웨어**: x86_64 Intel CPU만 (AVX2 최소, AVX-512 VNNI 권장). Apple Silicon / ARM은 Wave 8 candidate.
- **모델**: YOLO26n 그대로. 다른 모델은 Wave 8+.
- **정밀도**:
  - PyTorch: FP32 (baseline) + INT8 PT2E with X86InductorQuantizer.
  - TFLite: FP32 + INT8 dynamic + INT8 full (representative dataset PTQ).
- **OS**: Linux (primary) + Windows (pt2e는 OK, TFLite는 WSL 권장). macOS best-effort.

### Out-of-scope (future waves)
- **PyTorch ExecutorTorch backend** (모바일 배포 — Wave 8)
- **TFLite XNNPACK GPU delegate** / Android NNAPI / Core ML delegate (Wave 8)
- **INT4 weight-only on either stack** (LLM-centric, Vision 연구 미성숙)
- **QAT 경로** (fake-quant 학습 필요 → Wave 5 modifier 재사용 시점에 별도 wave)
- **TFLite micro** (MCU 타겟, 범위 밖)

### Assumptions (착수 전 확인 필요)
- `best_qr.pt` + `qr_barcode.yaml`이 Wave 6와 동일하게 사용 가능.
- `torch==2.8.0+cu129` cu124 runtime 없이도 CPU inference OK (확인 완료).
- Windows 호스트에서 TFLite runtime wheel 설치 여부는 Task 0 spike(R1)에서 확인 — 실패 시 WSL 또는 Docker로 fallback.

---

## Recipe Map (#40–#44)

| # | name | source | dtype | 툴체인 | 타겟 하드웨어 |
|---|---|---|---|---|---|
| 40 | `pytorch_pt2e_fp32` | `pt2e` | fp32 | torch.export + torch.compile(inductor) | AVX2+ (all) |
| 41 | `pytorch_pt2e_int8_x86` | `pt2e` | int8 | `X86InductorQuantizer` + PT2E PTQ | AVX-512 VNNI 권장 |
| 42 | `tflite_fp32` | `tflite` | fp32 | onnx2tf → .tflite + XNNPACK | AVX2+ |
| 43 | `tflite_int8_dynamic` | `tflite` | int8 | TFLite Dynamic Range Quantization | AVX-512 VNNI 권장 |
| 44 | `tflite_int8_full` | `tflite` | int8 | TFLite full INT8 PTQ (rep dataset) | AVX-512 VNNI 권장 |

**번호 선택**: Wave 6가 #30–#35. 36–39는 parked buffer (OpenVINO QAT 등). 40부터 Wave 7 블록 — 번호 점프가 **backend family 전환의 시각 신호**다 (Wave 6 번호 정책 계승).

**Parked 후보**:
- `#45 tflite_int8_full_xnnpack_fp16` — XNNPACK FP16 delegate (ARM/Apple 전용, Wave 8)
- `#46 pt2e_int8_inductor_capture` — `torch.compile(fullgraph=True)` + capture quantized 경로의 perf tradeoff 실험

---

## Measurement Hygiene Additions

Wave 6 `measurement` 인프라(warmup 200/measure 300, `stddev_ms`, `iter_cooldown_ms`, `thread_count`)를 그대로 승계. 추가:

| 항목 | Wave 6 | Wave 7 보강 |
|---|---|---|
| CPU env 필드 | ✔ (cpu_model, cpu_flags 등) | 승계 |
| `Result.env.torch` | ✔ | 승계 — pt2e 버전 해석에 사용 |
| `Result.env.tensorflow` | — | **신규 Optional[str]** (tflite path에서 채움) |
| PT2E quantizer config 기록 | — | `Result.notes`에 `X86InductorQuantizer(symmetric=True)` 문자열로 |
| TFLite delegate 기록 | — | `Result.notes`에 `XNNPACK-delegate` 명시 |

→ `_schemas.py::EnvInfo.tensorflow: Optional[str] = None` 1줄 추가. 역호환 유지.

---

## Task Dependency Graph

```
Task 0 (R1/R2 spike)   ┐
                        │
Task 1 (schema)         ├── Task 2 (pt2e handler — #40 fp32, #41 int8)
                        │       │
                        │       └── Task 4 (recipes #40/#41)
                        │
                        └── Task 3 (tflite handler — #42 fp32, #43 dyn, #44 full)
                                │
                                └── Task 4 (recipes #42-44)
                                        │
                                        ├── Task 5 (pyproject cpu extras 확장)
                                        │
                                        └── Task 6 (Makefile + batch 확장)
                                                │
                                                └── Task 7 (docs)
                                                        │
                                                        └── Task 8 (smoke E2E)
```

**병렬 가능**: Task 2 ∥ Task 3 (서로 다른 backend, 같은 `run_cpu.py` 안에서 함수 단위 분리).
**필수 순서**: Task 0 → 나머지 (R1/R2 미해소 상태에서 구현 시작하면 삽질 위험).

---

## Task 0: Spike — TFLite Windows runtime + onnx2tf compat (R1/R2)

**Why**: Wave 6 R1 spike(`_spike_wave6_r1.py`)가 OpenVINO + YOLO26n attention block 호환을 먼저 확인한 덕에 Task 5 착수 시 decision-risk가 거의 없었다. 같은 논리로 Wave 7의 두 개 미확인 질문을 먼저 푼다.

**Files**: `scripts/_spike_wave7_r1.py` (NEW)

- [ ] **Step 1: `tflite-runtime` 설치 시도**
  - Windows: `pip install tflite-runtime` — 최근 버전 wheel 제공 여부 확인
  - 실패 시: `pip install tensorflow` (무거움) 또는 WSL로 Fallback
  - **결과 기록**: `docs/improvements/2026-04-21-wave7-tflite-install.md`에 설치 매트릭스 (OS × Python 버전 × wheel 가용성)
- [ ] **Step 2: ONNX → TFLite 변환 smoke**
  - `onnx2tf -i best_qr.onnx -o tmp_tflite/` — YOLO26n attention block이 TF graph로 번역되는지
  - Stage 0: export (Wave 6 캐시 재사용)
  - Stage 1: onnx2tf FP32 변환 → `.tflite` 파일 생성 확인
  - Stage 2: Interpreter로 1회 forward, output shape `(1, 300, 6)` 보존 확인
  - Stage 3: INT8 PTQ with representative dataset (`best_qr.onnx` + 32 random samples)
- [ ] **Step 3: PyTorch PT2E smoke**
  - Stage 0: `model = ultralytics.YOLO("best_qr.pt").model`
  - Stage 1: `torch.export.export(model, (example,))`이 YOLO26n의 dynamic forward를 처리하는지 — **Brevitas fx-trace 실패 교훈**이 여기서도 재현될 수 있음
  - Stage 2: `X86InductorQuantizer` + `prepare_pt2e` + calibrate (1 sample) + `convert_pt2e`
  - Stage 3: `torch.compile(mode="reduce-overhead")` → 1 forward
- [ ] **Step 4: Verdict**
  - 네 가지 경로(TFLite FP32/INT8, PT2E FP32/INT8) 전부 PASS → Wave 7 CLEARED, 전체 task 진행
  - 일부 FAIL → 해당 recipe만 parked 처리 후 나머지 진행
  - torch.export가 YOLO26n에서 아예 실패 → **pt2e 경로 전체 park + Wave 7을 TFLite 단독으로 재정의**
- [ ] **Step 5: Commit** — `docs(plans): Wave 7 R1 spike + install matrix` (코드는 spike script만, 프로덕션 변경 없음)

---

## Task 1: Schema 확장

**Why**: `source` Literal이 `pt2e`/`tflite`를 모르면 recipe 로드에서 pydantic이 차단. `EnvInfo.tensorflow` 필드 없으면 TFLite 결과가 버전 정보를 잃는다.

**Files**: `scripts/_schemas.py`, `tests/test_schema_wave7.py` (NEW)

- [ ] **Step 1: Write the failing test** — `Recipe.model_validate`가 `source: pt2e` / `source: tflite` 통과, `EnvInfo`가 `tensorflow` 필드 수락
- [ ] **Step 2: Run test → fail**
- [ ] **Step 3: Extend `_schemas.py`**
  - `TechniqueSpec.source`에 `"pt2e", "tflite"` 추가
  - `EnvInfo.tensorflow: Optional[str] = None`
- [ ] **Step 4: Run test → pass + 28개 기존 recipe load 통과**
- [ ] **Step 5: Commit** — `feat(schemas): Wave 7 pt2e/tflite sources + EnvInfo.tensorflow`

---

## Task 2: PyTorch PT2E path (#40, #41)

**Why**: Torch 네이티브 양자화는 ONNX를 우회. `torch.export` 그래프에 `X86InductorQuantizer`를 적용하고 `torch.compile`로 실행 — 의존성 최소(이미 torch 설치됨), 디버깅 편함, ExecutorTorch 모바일 확장의 발판.

**Files**: `scripts/run_cpu.py`, `tests/test_run_cpu_pt2e.py` (NEW)

- [ ] **Step 1: Write failing tests**
  - `source=pt2e + dtype=fp32` 디스패치
  - `source=pt2e + dtype=int8` 디스패치
  - `PytorchPT2EAsORT` 어댑터가 `.run`, `.get_inputs`, `.get_outputs` 노출
  - Module import invariant — tensorflow / tflite 등 top-level import 금지 (Wave 6 TRT-free 원칙 계승)
- [ ] **Step 2: Run tests → fail**
- [ ] **Step 3: `_prepare_pytorch_pt2e_fp32`**
  ```python
  import torch
  from ultralytics import YOLO
  model = YOLO(weights).model.eval().to("cpu")
  example = torch.randn(1, 3, imgsz, imgsz)
  exported = torch.export.export(model, (example,))
  compiled = torch.compile(exported.module(), mode="reduce-overhead")
  # 1회 warm-up trigger (compile lazy)
  with torch.no_grad():
      _ = compiled(example)
  return PytorchPT2EAsORT(compiled, "images", ["output0"])
  ```
- [ ] **Step 4: `_prepare_pytorch_pt2e_int8_x86`**
  ```python
  from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
  from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
      X86InductorQuantizer, get_default_x86_inductor_quantization_config,
  )
  quantizer = X86InductorQuantizer()
  quantizer.set_global(get_default_x86_inductor_quantization_config())
  prepared = prepare_pt2e(exported.module(), quantizer)
  # Calibrate with _iter_calib_samples (reuse Wave 6 helper)
  for x in _iter_calib_samples(val_yaml, n_samples=128, imgsz, seed):
      prepared(torch.from_numpy(x))
  quantized = convert_pt2e(prepared)
  compiled = torch.compile(quantized, mode="reduce-overhead")
  return PytorchPT2EAsORT(compiled, "images", ["output0"])
  ```
- [ ] **Step 5: `PytorchPT2EAsORT` adapter** — `OVRunnerAsORT` 패턴 재사용, forward 결과는 `torch.Tensor` → `.cpu().numpy()` 변환
- [ ] **Step 6: Dispatcher branches** in `_prepare_cpu_session`
- [ ] **Step 7: Run tests → pass**
- [ ] **Step 8: Commit** — `feat(run_cpu): Wave 7 Task 2 — PyTorch PT2E fp32 + x86 int8`

---

## Task 3: TensorFlow Lite path (#42, #43, #44)

**Why**: TFLite는 Google이 정한 *de-facto* mobile/edge 표준. XNNPACK delegate가 AVX2/VNNI 활용해 x86에서도 경쟁력 있음. OpenVINO/ORT와 나란히 두고 "어떤 CPU backend가 실제 빠른가"의 데이터 생성.

**Files**: `scripts/run_cpu.py`, `tests/test_run_cpu_tflite.py` (NEW)

- [ ] **Step 1: Write failing tests** — dispatch routing + `TFLiteAsORT` interface + module invariant
- [ ] **Step 2: Run tests → fail**
- [ ] **Step 3: `_prepare_tflite_fp32`**
  - ONNX→TFLite 변환 (onnx2tf), 캐시 to `results_cpu/_tflite/`
  - `tflite_runtime.Interpreter(model_path=..., num_threads=N)` 또는 `tensorflow.lite.Interpreter`
  - XNNPACK delegate 활성화 확인 (기본 활성)
- [ ] **Step 4: `_prepare_tflite_int8_dynamic`**
  - `converter.optimizations = [tf.lite.Optimize.DEFAULT]` → weight-only INT8 (no activation scales at build time)
- [ ] **Step 5: `_prepare_tflite_int8_full`**
  - representative_dataset generator (calibration_samples 128, reuse `_iter_calib_samples`)
  - `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`
  - `converter.inference_input_type = tf.int8`
  - `converter.inference_output_type = tf.int8`
- [ ] **Step 6: `TFLiteAsORT` adapter**
  - `interpreter.set_tensor(input_idx, x)` → `interpreter.invoke()` → `interpreter.get_tensor(output_idx)`
  - Quantized INT8 모델은 input/output scale·zero_point 처리 필요: wrapper에서 dequantize
- [ ] **Step 7: Dispatcher branches**
- [ ] **Step 8: Run tests → pass**
- [ ] **Step 9: Commit** — `feat(run_cpu): Wave 7 Task 3 — TFLite fp32 + int8 dyn + int8 full`

---

## Task 4: Recipe YAML #40–#44

**Files**: `recipes/40_*.yaml` ~ `recipes/44_*.yaml`

- [ ] **Step 1**: `40_pytorch_pt2e_fp32.yaml` — dtype fp32, min_fps_bs1 null (baseline)
- [ ] **Step 2**: `41_pytorch_pt2e_int8_x86.yaml` — dtype int8, calibrator null (PT2E 내부), calibration_samples 128
- [ ] **Step 3**: `42_tflite_fp32.yaml`
- [ ] **Step 4**: `43_tflite_int8_dynamic.yaml` — calibrator null
- [ ] **Step 5**: `44_tflite_int8_full.yaml` — calibrator "tflite_full", calibration_samples 128
- [ ] **Step 6**: 33개 기존 recipe + 5개 신규 전부 `load_recipe` 통과 확인
- [ ] **Step 7: Commit** — `feat(recipes): Wave 7 #40-#44 PyTorch PT2E + TFLite`

---

## Task 5: pyproject 의존성

**Files**: `pyproject.toml`

- [ ] `[cpu]` extra 확장:
  ```toml
  cpu = [
      "openvino>=2024.3",
      "nncf>=2.9,<3.0",
      "psutil>=5.9",
      "py-cpuinfo>=9.0",
      "onnx2tf>=1.22",
      "tflite-runtime>=2.15; sys_platform == 'linux' or sys_platform == 'darwin'",
      "tensorflow>=2.15; sys_platform == 'win32'",  # Windows는 tflite-runtime wheel 부재
  ]
  ```
- [ ] R1 spike 결과에 따라 Windows TFLite 의존성 조정 (경로가 WSL만 된다면 extras에서 제외 + docs에 WSL 안내)
- [ ] **Commit** — `chore(deps): Wave 7 tflite + onnx2tf extras`

---

## Task 6: Makefile + batch + docs

**Files**: `Makefile`, `scripts/run_cpu_batch.sh`

- [ ] **Step 1**: Makefile recipe-40 ~ recipe-44 타겟 (Wave 6 recipe-30-35 패턴 복사)
- [ ] **Step 2**: `cpu-all`, `cpu-qr` 타겟에 신규 recipe 5장 추가
- [ ] **Step 3**: `scripts/run_cpu_batch.sh`의 `CPU_RECIPES=(...)` 배열 확장
- [ ] **Step 4**: `.gitignore`에 `results_cpu/_tflite/`, `results_cpu_qr/_tflite/` 추가
- [ ] **Step 5: Commit** — `chore(infra): Wave 7 Makefile + batch targets`

---

## Task 7: Documentation

**Files**: `docs/architecture.md`, `CLAUDE.md`, `README.md`

- [ ] **Step 1**: `docs/architecture.md` Wave 7 섹션 신규 (Wave 6 섹션 스타일 매칭)
  - PyTorch PT2E 경로 설명 (`torch.export` + `X86InductorQuantizer` + `torch.compile`)
  - TFLite 경로 설명 (`onnx2tf` → `.tflite` + XNNPACK)
  - Waves 테이블에 Wave 7 행 추가
  - `cold_start_ms` 해석 테이블에 pt2e(`torch.compile` 첫 실행 수초 가능), tflite(~100ms) 추가
- [ ] **Step 2**: `CLAUDE.md` Latest plan 링크를 이 파일로 교체, Critical conventions에 "pt2e/tflite는 torch/tensorflow 최신 매칭 필요" 한 줄 추가
- [ ] **Step 3**: `README.md` 레시피 테이블 + ISA 매트릭스에 5장 row 추가
- [ ] **Step 4: Commit** — `docs: Wave 7 PyTorch PT2E + TFLite sections`

---

## Task 8: Smoke E2E (Wave 6 Task 10 패턴)

**Why**: 실기 실행 없이 실제 이슈를 예측 못 함 — Wave 6 Task 10에서 ort_cpu_int8_static mAP=0 regression을 잡은 것이 대표 사례.

**Files**: runtime only, code 변경 없음

- [ ] **Step 1**: Prerequisites — `best_qr.pt`, `qr_barcode.yaml`, `coco_val_only.yaml`, R1 spike로 확인된 runtime 설치
- [ ] **Step 2**: `#40 pytorch_pt2e_fp32` 단독 smoke → p50 / mAP / env 확인
- [ ] **Step 3**: `#42 tflite_fp32` 단독 smoke
- [ ] **Step 4**: 전체 5장 batch (또는 `make cpu-all`)
- [ ] **Step 5**: `report_cpu_qr.md` 재생성 — 이제 #30-#35 + #40-#44 전부 포함 (11장 ranking)
- [ ] **Step 6: Result comparison**
  - Wave 6 winner(#35 openvino_int8_nncf 23.9 fps bs1)와 Wave 7 best (#41 또는 #44) 비교
  - 기대: PT2E x86 INT8이 OpenVINO와 유사 수준(±15%), TFLite XNNPACK은 OV에 약간 뒤지지만 mobile 확장성 때문에 가치 있음
- [ ] **Step 7**: Follow-up 이슈 정리
  - #33 ort_cpu_int8_static mAP=0 Wave 6 regression과 별개로 Wave 7에서 생기는 이슈 기록
  - `docs/improvements/2026-04-21-wave7-findings.md` 또는 커밋 메시지에 정리
- [ ] **Step 8: Commit results** — `feat(results): Wave 7 CPU eval + report (40-44)`

---

## Known Risks & Decisions

### R1. TFLite Windows runtime 부재
**Risk**: `tflite-runtime`은 Google 공식으로 Linux/macOS/Android aarch64만 wheel 제공. Windows는 `tensorflow` 전체를 깔거나 WSL에서만 실행. Windows 호스트에서 바로 #42-#44를 못 돌릴 가능성.
**Mitigation**: Task 0 Step 1 spike에서 매트릭스 확정. 다음 셋 중 택1:
  1. **`tensorflow` 전체 설치** (사이즈 큼, 하지만 Windows CI 가능) — 기본안
  2. **WSL 필수** — `docs/architecture.md`에 명시
  3. **Docker container만** — `Dockerfile.cpu` 신규
**Decision**: Task 0 결과에 따라. 가장 간단한 길은 1번 (requirements 추가만).

### R2. onnx2tf의 YOLO26n attention block 호환성
**Risk**: Wave 3 INC SmoothQuant가 YOLO26n attention Reshape node를 stale 상태로 남겼듯, `onnx2tf`가 TF graph 변환에서 유사 이슈를 낼 수 있다.
**Mitigation**: Task 0 Step 2 spike의 Stage 1-3이 정확히 이 점을 검증. 실패 시 `onnx2tf --onnx_opset_version 17` 또는 `onnx-tf` 라이브러리 전환 시도. Worst case #44 FP32만 올리고 INT8은 parked.

### R3. `torch.export` + ultralytics dynamic forward 호환성
**Risk**: Brevitas fx-trace가 YOLO26n의 Python dynamic flow(`if ... else return`)에 막혔던 전례. `torch.export`는 TorchDynamo 기반이라 fx보다 관대하지만, ultralytics가 non-functional 구조(`self.inplace` 등)를 쓰면 export 실패 가능.
**Mitigation**: Task 0 Step 3 Stage 1이 이 점을 검증. 실패 시:
  - `torch.export.export(..., strict=False)` 시도
  - 또는 `export_onnx` 우회(ONNX → torch → pt2e 재변환) — 하지만 이건 우리 도구 목적에 반함
  - Worst case PT2E 전체 parked → Wave 7이 TFLite 단독으로 축소

### R4. PT2E 양자화된 그래프의 torch.compile 레이턴시
**Risk**: `torch.compile(mode="reduce-overhead")`의 첫 호출이 수 초 걸릴 수 있음 — `cold_start_ms`가 FP32 baseline(~200ms)과 비교할 때 크게 튐.
**Mitigation**: `measure_cold_start`가 이를 자연스럽게 기록 — report.md는 `cold_start_ms`를 ranking 축에서 제외하므로(Wave 6 결정) 문제 없음.

### R5. 의존성 충돌 (`tensorflow` vs `torch`)
**Risk**: TensorFlow와 PyTorch가 각자 `protobuf`, `numpy` 버전 고정을 다르게 요구할 수 있음. pip가 해결 가능하지만 sub-dependencies(`grpcio`, `absl-py`)가 아랫선 내린 호환 행렬 수리 필요.
**Mitigation**: Task 0 Step 1 설치 직후 `pip check`로 충돌 조기 발견. Worst case `tensorflow`를 별도 venv로 분리하고 `run_cpu.py`가 환경 변수로 분기.

---

## Extension Gates (Wave 8+ candidates)

- **ExecutorTorch** — PyTorch 모바일 inference (`torch.export` → `.pte`). Wave 7 PT2E 결과물 재사용 가능.
- **TFLite XNNPACK FP16 delegate** (ARM/Apple) — iOS/Android 배포 시 핵심.
- **TFLite NNAPI delegate** (Android accelerator) — phone-class inference.
- **Core ML EP** (Apple Silicon) — Neural Engine offload. Wave 6에서 제외된 남은 대형 타겟.
- **AMD Ryzen AI Engine** — NPU delegate. 아직 Windows Only 드라이버.
- **torch.compile `fullgraph=True` + CUDA backend** — GPU INT8 대체 경로 (modelopt/brevitas와 비교).

---

## Dependencies on Wave 6

Wave 7은 Wave 6의 다음 산출물에 **의존**:
- `scripts/_weights_io.py::_iter_calib_samples` — PT2E / TFLite INT8 calibration 공유
- `scripts/run_cpu.py::_prepare_cpu_session` dispatcher — branch 추가 지점
- `OVRunnerAsORT` 패턴 — `PytorchPT2EAsORT` / `TFLiteAsORT` 원본
- `scripts/env_lock.py::collect_env` — `cpu_flags`로 XNNPACK int8 optimal 여부 판정
- `LatencyStats.stddev_ms`, `MeasurementSpec.iter_cooldown_ms` — 동일 hygiene 적용
- `results_cpu_qr/report_cpu_qr.md` — Wave 7 실행 후 재생성하면 같은 표에 11장 ranking

## Success Criteria (Wave 7 "active" 선언 조건)

1. **5장 recipe 전부 로드 가능** (`load_recipe` 통과)
2. **최소 2장이 `meets_constraints=True`** (fp32 경로 2장 + int8 경로 1장 이상 success)
3. **#33 같은 silent 회귀 없음** — INT8 경로가 mAP=0 같은 collapse를 일으키면 해당 recipe parked, 하지만 notes에 원인 기록
4. **report에 11장(#30-#35 + #40-#44) 통합 ranking 출력** — GPU 20장과는 별도 파일(`report_cpu_qr.md`)
5. **Wave 7 spec doc** (`docs/plans/2026-04-21-wave7-pytorch-tflite-quant.md`)의 Task 체크리스트 7/8 완료 (Step 3b 같은 연구성 step은 예외)

Wave 6의 실제 값: 이 문서의 #35 openvino_int8_nncf 23.9 fps를 **benchmark**로 둔다. Wave 7 경로들이 그 근처(±20%)에 오면 "diversity of backend options", 뚜렷이 앞서면 "Wave 7 replaces Wave 6 recommendation", 뒤처지면 "Wave 7는 호환성/모바일 확장용으로만 value" — 세 해석 중 어느 쪽인지 Task 8 후 확정.
