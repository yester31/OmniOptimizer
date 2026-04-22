# Wave 7: PyTorch PT2E 양자화 + ORT XNNPACK EP — **ARCHIVED 2026-04-21**

> **⛔ ARCHIVED**: Wave 7 Task 0 spike(`scripts/_spike_wave7_r3.py`)에서 두 축 모두 BLOCKED 확인. 계획은 학습 자료로 보존.
> - **R3 PT2E**: `torch.export`가 ultralytics Detect head의 `self.anchors` 재할당 패턴에서 실패 (strict), non-strict는 dict 반환으로 pipeline 미연결.
> - **R5 XNNPACK**: 현재 `onnxruntime-gpu` wheel에 XNNPACK EP 미포함. 일반 `onnxruntime` wheel 필요 → 별도 venv 전제.
>
> 자세한 spike 결과: [`docs/improvements/2026-04-21-wave7-r3-r5-spike-results.md`](../improvements/2026-04-21-wave7-r3-r5-spike-results.md).
>
> **후속 방향**: Wave 8 ncnn으로 점프 — 모바일 배포 가치가 더 명확. PT2E는 ExecutorTorch 성숙 시점(2027+)에 재검토.

---

> **Rescope 2026-04-21**: 원안은 PyTorch PT2E + **TensorFlow Lite** 조합이었으나 TFLite Windows runtime 부재(R1) + onnx2tf의 YOLO26n attention 호환 리스크(R2)로 **TFLite 제거**. 대체로 **ORT XNNPACK EP** 채택 — 같은 XNNPACK 커널을 TFLite 무게 없이 기존 onnxruntime 설치로 쓴다. Wave 6 `ort_cpu` 인프라 99% 재사용.

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (권장) 또는 superpowers:executing-plans. Steps are TDD-style (`- [ ]`) — 체크박스가 진행 트래커.

**Goal:** Wave 6이 ORT CPU EP + OpenVINO로 CPU를 추가했다면, Wave 7은 **PyTorch 네이티브(pt2e) + ORT XNNPACK EP** 두 경로를 붙인다. 같은 YOLO26n + 같은 호스트에서 4종 CPU backend(ORT-CPU-EP / ORT-XNNPACK-EP / OpenVINO / PyTorch)를 나란히 비교해 배포 파이프라인 선택 자유도를 올린다.

**Architecture:** 변경 지점 5곳. (1) `scripts/_schemas.py::TechniqueSpec.source` Literal에 `pt2e` 추가 (XNNPACK은 source가 아니라 `runtime.execution_provider` 필드로 구분). (2) `scripts/run_cpu.py`에 PyTorch PT2E 디스패처 브랜치 + `PytorchPT2EAsORT` 어댑터. ORT XNNPACK은 기존 `_prepare_ort_cpu_*` 함수에 `execution_provider` 처리만 추가. (3) `recipes/40–43_*.yaml` 신규. (4) `pyproject.toml` 변경 최소 — **추가 의존성 0**. (5) `Makefile`·`docs`·README 타겟·테이블 갱신. `run_trt.py`/`run_ort.py`/기존 Wave 6 코드는 **건드리지 않는다**.

**Tech Stack:** `torch>=2.8` (`torch.export` + `torch.ao.quantization.quantize_pt2e` + `X86InductorQuantizer`, 이미 설치), `onnxruntime>=1.17` (이미 설치 — XNNPACK EP는 v1.14+ 내장), `numpy`, `ultralytics`, `pydantic`. Wave 6 `OVRunnerAsORT` 어댑터 패턴을 `PytorchPT2EAsORT`로 재사용.

---

## Scope Boundaries

### In-scope (Wave 7)
- **하드웨어**: x86_64 Intel CPU만 (AVX2 최소, AVX-512 VNNI 권장). Apple Silicon / ARM은 Wave 8 candidate.
- **모델**: YOLO26n 그대로. 다른 모델은 Wave 8+.
- **정밀도**:
  - PyTorch: FP32 (baseline) + INT8 PT2E with X86InductorQuantizer.
  - ORT XNNPACK: FP32 + INT8 static (QDQ with XNNPACK INT8 kernels).
- **OS**: Linux + Windows + macOS 모두 OK (TFLite Windows 장벽 사라짐).

### Out-of-scope (future waves)
- **TensorFlow Lite** — 원안에서 제거 (Windows runtime 부재, onnx2tf 호환 리스크, 추가 의존성 부담). 모바일 배포 필요 시 Wave 8에서 **ExecutorTorch** 또는 **ncnn**으로 해결.
- **PyTorch ExecutorTorch backend** (모바일 배포 — Wave 8)
- **INT4 weight-only on either stack** (LLM-centric, Vision 연구 미성숙)
- **QAT 경로** (fake-quant 학습 필요 → Wave 5 modifier 재사용 시점에 별도 wave)

### Assumptions (착수 전 확인 필요)
- `best_qr.pt` + `qr_barcode.yaml`이 Wave 6와 동일하게 사용 가능.
- `torch==2.8.0+cu129`가 CPU inference OK (Wave 6에서 확인 완료).
- `onnxruntime==1.22.0`에 `XnnpackExecutionProvider`가 실제 등록됨 — Task 0 spike에서 확인.

---

## Recipe Map (#40–#43)

| # | name | source | execution_provider | dtype | 툴체인 | 타겟 |
|---|---|---|---|---|---|---|
| 40 | `pytorch_pt2e_fp32` | `pt2e` | — | fp32 | torch.export + torch.compile | AVX2+ |
| 41 | `pytorch_pt2e_int8_x86` | `pt2e` | — | int8 | `X86InductorQuantizer` + PT2E PTQ | AVX-512 VNNI 권장 |
| 42 | `ort_cpu_xnnpack_fp32` | `ort_cpu` | `xnnpack` | fp32 | ORT XNNPACK EP (FP32 kernels) | AVX2+ |
| 43 | `ort_cpu_xnnpack_int8_static` | `ort_cpu` | `xnnpack` | int8 | ORT XNNPACK EP + static QDQ | AVX-512 VNNI 권장 |

**번호 선택**: Wave 6가 #30–#35. 36–39는 parked buffer. 40부터 Wave 7 블록.

**Parked 후보**:
- `#44 ort_cpu_xnnpack_int8_dynamic` — dynamic 경로는 XNNPACK에서 효과 제한적, ORT dynamic과 거의 동일 예상, 시간 절약 위해 생략
- `#45 executorch_int8_x86` — PyTorch ExecutorTorch backend (Wave 8 extension gate)

---

## Measurement Hygiene Additions

Wave 6 `measurement` 인프라(warmup 200/measure 300, `stddev_ms`, `iter_cooldown_ms`, `thread_count`)를 그대로 승계. 추가:

| 항목 | Wave 6 | Wave 7 보강 |
|---|---|---|
| CPU env 필드 | ✔ | 승계 |
| `Result.notes`에 provider 명시 | Wave 6 (ORT/OV) | XNNPACK 사용 시 `providers=[Xnnpack, CPU]` 문자열 기록 |
| PT2E quantizer config 기록 | — | `Result.notes`에 `X86InductorQuantizer(symmetric=True)` |

스키마 변경 없음 — Wave 6에서 이미 `cpu_flags`/`stddev_ms` 등 필요 필드 전부 확보.

---

## Task Dependency Graph

```
Task 0 (R3 spike)   ┐
                     │
Task 1 (schema)      ├── Task 2 (pt2e handler — #40 fp32, #41 int8)
                     │       │
                     │       └── Task 4 (recipes #40/#41)
                     │
                     └── Task 3 (XNNPACK EP wiring — #42/#43)
                             │
                             └── Task 4 (recipes #42/#43)
                                     │
                                     ├── Task 5 (Makefile + batch 확장)
                                     │
                                     └── Task 6 (docs)
                                             │
                                             └── Task 7 (smoke E2E)
```

**병렬 가능**: Task 2 ∥ Task 3 (서로 다른 backend, 같은 `run_cpu.py` 안에서 함수 단위 분리).
**필수 순서**: Task 0 → 나머지 (R3 미해소 상태에서 PT2E 구현 시작하면 삽질 위험).

---

## Task 0: Spike — `torch.export` + ultralytics YOLO26n 호환 (R3)

**Why**: Wave 4 Brevitas가 fx-trace로 YOLO26n을 tracing하려다 ultralytics의 Python dynamic forward(`if ... else return`)에 막혔다. `torch.export`는 TorchDynamo 기반이라 fx보다 관대하지만, 실기 확인이 먼저다. 또 XNNPACK EP가 실제 등록돼 있는지 5초 smoke로 같이 확인.

**Files**: `scripts/_spike_wave7_r3.py` (NEW)

- [ ] **Step 1: XNNPACK EP availability**
  - `onnxruntime.get_available_providers()` 출력에 `XnnpackExecutionProvider` 포함 여부
  - 없으면 `pip install onnxruntime` 재설치 or `onnxruntime-directml` 전환 검토
- [ ] **Step 2: `torch.export` smoke**
  - `model = YOLO("best_qr.pt").model.eval().cpu()`
  - `exported = torch.export.export(model, (torch.randn(1,3,640,640),))`
  - Stage 1a: strict=True 시도 — 실패 시
  - Stage 1b: strict=False fallback — 실패 시
  - Stage 1c: `torch.export.export(..., preserve_module_call_signature=(...))` 또는 수동 wrapper class
- [ ] **Step 3: PT2E quantize smoke**
  - `X86InductorQuantizer` + `prepare_pt2e` + 1 sample calibrate + `convert_pt2e`
  - 1 forward로 출력 shape `(1,300,6)` 보존 확인
- [ ] **Step 4: `torch.compile` latency**
  - `compiled = torch.compile(quantized_module)` — **CPU 경로 기본 mode 사용**.
    `mode="reduce-overhead"`는 CUDA graph 기반이라 CPU에선 무의미 또는 역효과.
    공격적 탐색이 필요하면 `mode="max-autotune-no-cudagraphs"` 옵션.
  - 1 forward → 첫 호출 시간 (cold_start 수치 참고치)
- [ ] **Step 5: XNNPACK QDQ fallback 실기 확인 (R5)**
  - `import onnxruntime as ort; ort.get_available_providers()` — `XnnpackExecutionProvider` 포함 확인
  - Wave 6 static INT8 캐시(`results_cpu/_onnx/ort_cpu_int8_static_*.onnx`) 중 아무거나 `providers=["XnnpackExecutionProvider", "CPUExecutionProvider"]`로 로드
  - 1 forward → 출력 shape 보존 확인
  - `session.get_providers()` 반환이 XNNPACK을 실제 assign했는지 / CPU로 전부 떨어졌는지 기록
  - **목적**: fallback이 100%면 #43 recipe는 `ort_cpu_int8_static`과 동일해짐 → Wave 7 implementation 전에 재양자화 (Task 3 Step 5b) 필요성 판정
- [ ] **Step 6: Verdict**
  - 4단계 모두 PASS → Wave 7 CLEARED, 전체 task 진행
  - Step 2 실패 → **PT2E 전체 parked**, Wave 7을 XNNPACK 단독(#42/#43)으로 축소
  - Step 3 실패 (export OK / quantize 깨짐) → PT2E INT8(#41) parked, FP32(#40)만 유지
  - Step 5 fallback 100% → Task 3 Step 5b 재양자화 의무화 (skip 불가)
- [ ] **Step 7: Commit** — `docs(plans): Wave 7 R3/R5 spike results`

---

## Task 1: Schema 확장

**Why**: `source` Literal이 `pt2e`를 모르면 recipe 로드에서 차단. XNNPACK은 기존 `source=ort_cpu` + `execution_provider` 필드로 표현하므로 추가 필요 없음.

**Files**: `scripts/_schemas.py`, `tests/test_schema_wave7.py` (NEW)

- [ ] **Step 1: Write failing test** — `source: pt2e` 레시피가 `Recipe.model_validate` 통과, `source: ort_cpu + execution_provider: xnnpack` 조합도 통과
- [ ] **Step 2: Run test → fail**
- [ ] **Step 3: Extend `_schemas.py`** — `TechniqueSpec.source`에 `"pt2e"` 추가
- [ ] **Step 4: Run test → pass + 28개 기존 recipe load 통과**
- [ ] **Step 5: Commit** — `feat(schemas): Wave 7 pt2e source`

---

## Task 2: PyTorch PT2E path (#40, #41)

**Why**: Torch 네이티브 양자화는 ONNX를 우회. `torch.export` 그래프에 `X86InductorQuantizer`를 적용하고 `torch.compile`로 실행 — 추가 의존성 없음, 디버깅 편함, Wave 8 ExecutorTorch 확장의 발판.

**Files**: `scripts/run_cpu.py`, `tests/test_run_cpu_pt2e.py` (NEW)

- [ ] **Step 1: Write failing tests**
  - `source=pt2e + dtype=fp32` 디스패치
  - `source=pt2e + dtype=int8` 디스패치
  - `PytorchPT2EAsORT` 어댑터가 `.run`, `.get_inputs`, `.get_outputs` 노출
  - Module import invariant — tensorflow 등 top-level import 금지
- [ ] **Step 2: Run tests → fail**
- [ ] **Step 3: `_prepare_pytorch_pt2e_fp32`**
  ```python
  import torch
  from ultralytics import YOLO
  model = YOLO(weights).model.eval().to("cpu")
  example = torch.randn(1, 3, imgsz, imgsz)
  with torch.no_grad():
      exported = torch.export.export(model, (example,))
      # CPU inference uses inductor default mode. `reduce-overhead` is
      # CUDA-graph-based and produces no speedup (often slower) on CPU.
      compiled = torch.compile(exported.module())
      _ = compiled(example)  # lazy compile trigger — first call is slow
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
  # _iter_calib_samples yields (1, 3, H, W) float32 — batch dim matches
  # the export example, so prepared.forward() accepts it directly.
  for x in _iter_calib_samples(val_yaml, n_samples=128, imgsz, seed):
      with torch.no_grad():
          prepared(torch.from_numpy(x))
  quantized = convert_pt2e(prepared)
  compiled = torch.compile(quantized)  # CPU default mode, no reduce-overhead
  return PytorchPT2EAsORT(compiled, "images", ["output0"])
  ```
  - **calibration_samples=128**: Wave 6 #33에서 Windows 페이징 제약으로 확정한 값.
    Wave 7은 같은 호스트 상속이므로 128 유지. 메모리 여유 있는 호스트는 recipe에서 256으로 override 가능.
- [ ] **Step 5: `PytorchPT2EAsORT` adapter** — Wave 6 `OVRunnerAsORT` 패턴 재사용.
  - Forward: `torch.from_numpy(x)` → `compiled(...)` → `.cpu().numpy()` 변환
  - `Result.notes`에 `pt2e_adapter: torch<->numpy conversion included in measured latency` 한 줄 기록해 다른 backend와 fairness 맥락을 report 해석 시 명시 (Wave 6 OV와 동일 overhead 포함 정책)
- [ ] **Step 6: Dispatcher branches** in `_prepare_cpu_session` — `source == "pt2e"` 분기
- [ ] **Step 7: Run tests → pass**
- [ ] **Step 8: Commit** — `feat(run_cpu): Wave 7 Task 2 — PyTorch PT2E fp32 + x86 int8`

---

## Task 3: ORT XNNPACK EP wiring (#42, #43)

**Why**: XNNPACK은 Google의 FP32/INT8 커널 라이브러리. TFLite가 쓰는 것과 **같은 커널**을 ORT EP로 노출. `providers=["XnnpackExecutionProvider", "CPUExecutionProvider"]` 한 줄 변경으로 INT8 VNNI 커널 경쟁력 테스트 가능 — 추가 변환기/런타임 없이.

**Files**: `scripts/run_cpu.py`, `tests/test_run_cpu_xnnpack.py` (NEW)

- [ ] **Step 1: Write failing tests**
  - Recipe `runtime.execution_provider: xnnpack` 있을 때 `_prepare_ort_cpu_*` 함수의 providers 리스트가 `["XnnpackExecutionProvider", "CPUExecutionProvider"]`
  - `execution_provider: null` (기본) → `["CPUExecutionProvider"]` (Wave 6 동작 유지 — 회귀 가드)
- [ ] **Step 2: Run tests → fail**
- [ ] **Step 3: `_build_ort_session_options` 확장**
  ```python
  def _resolve_providers(recipe: Recipe) -> list[str]:
      ep = recipe.runtime.execution_provider
      if ep == "xnnpack":
          return ["XnnpackExecutionProvider", "CPUExecutionProvider"]
      if ep is None or ep == "cpu":
          return ["CPUExecutionProvider"]
      raise ValueError(f"Unknown execution_provider for ort_cpu: {ep!r}")
  ```
- [ ] **Step 4: `_prepare_ort_cpu_fp32/int8_static/int8_dynamic` 모두 providers 파라미터로 분기** — InferenceSession 생성 시 `providers=_resolve_providers(recipe)`
- [ ] **Step 5: Static INT8 주의** — XNNPACK INT8은 ORT `QOperator` 포맷 선호, 현재 우리 static은 `QuantFormat.QDQ`. XNNPACK + QDQ 호환 확인:
  - ORT 1.17+에서 QDQ + XNNPACK EP는 fallback 동작 검증됨
  - 만약 XNNPACK이 INT8 Conv를 선택 안 하면 CPU EP로 떨어져 Wave 6 #33과 동일해짐 → notes에 기록
- [ ] **Step 5b: #43 전용 재양자화 (Wave 6 #33 회귀 격리)**
  - **Problem**: Wave 6 #33 (`ort_cpu_int8_static`)은 mAP=0 회귀가 확정된 ONNX. 같은 결과물을 XNNPACK EP에 투입하면 당연히 mAP=0이 나와 "XNNPACK이 유효한가" 판정 불가.
  - **Solution**: #43은 **새 quantize_static ONNX**를 생성. Wave 6 #33과 다른 파라미터 조합으로 회귀 회피 시도:
    - Option A: `activation_type=QuantType.QUInt8` (Wave 6은 QInt8) + `ActivationSymmetric=False` — per-tensor asymmetric activation
    - Option B: `nodes_to_exclude`에 detect head의 `cv2.*`, `cv3.*` 추가 — 민감 레이어 FP 유지
    - Option C: `DedicatedQDQPair=True` — Wave 6에서 False였던 플래그 flip
  - 실기 smoke로 A/B/C 중 mAP 살아남는 첫 번째 조합 채택, 다른 것은 `docs/improvements/2026-04-21-wave7-int8-quant-debug.md`에 기록
  - 결과 ONNX는 `results_cpu/_onnx/ort_cpu_xnnpack_int8_static.onnx`로 캐시 (Wave 6 #33 ONNX와 별도 파일)
- [ ] **Step 6: Run tests → pass**
- [ ] **Step 7: Commit** — `feat(run_cpu): Wave 7 Task 3 — ORT XNNPACK EP via execution_provider`

---

## Task 4: Recipe YAML #40–#43

**Files**: `recipes/40_*.yaml` ~ `recipes/43_*.yaml`

- [ ] **Step 1**: `40_pytorch_pt2e_fp32.yaml`
- [ ] **Step 2**: `41_pytorch_pt2e_int8_x86.yaml` — calibration_samples 128
- [ ] **Step 3**: `42_ort_cpu_xnnpack_fp32.yaml` — `runtime.execution_provider: xnnpack`, `source: ort_cpu`, `dtype: fp32`
- [ ] **Step 4**: `43_ort_cpu_xnnpack_int8_static.yaml` — 위에 `dtype: int8`, `calibrator: entropy`, `calibration_samples: 128` 추가
- [ ] **Step 5**: 28개 기존 recipe + 4개 신규 전부 `load_recipe` 통과 확인
- [ ] **Step 6: Commit** — `feat(recipes): Wave 7 #40-#43 PyTorch PT2E + ORT XNNPACK`

---

## Task 5: Makefile + batch + pyproject

**Files**: `Makefile`, `scripts/run_cpu_batch.sh`, `pyproject.toml`

- [ ] **Step 1**: Makefile `recipe-40` ~ `recipe-43` 타겟 (Wave 6 recipe-30-35 패턴 복사)
- [ ] **Step 2**: `cpu-all`, `cpu-qr` 타겟에 신규 recipe 4장 추가 — `.PHONY` 리스트 갱신
- [ ] **Step 3**: `scripts/run_cpu_batch.sh`의 `CPU_RECIPES=(...)` 배열 확장
- [ ] **Step 4**: `pyproject.toml` — TFLite/onnx2tf **없음**, 기존 `[cpu]` extras 그대로 유지 (py-cpuinfo/psutil/openvino/nncf). 변경 없음 확인.
- [ ] **Step 5: Commit** — `chore(infra): Wave 7 Makefile + batch recipe-40..43`

---

## Task 6: Documentation

**Files**: `docs/architecture.md`, `CLAUDE.md`, `README.md`

- [ ] **Step 1**: `docs/architecture.md` Wave 7 섹션 — Wave 6 스타일 매칭
  - PyTorch PT2E 경로 설명 (`torch.export` + `X86InductorQuantizer` + `torch.compile`)
  - XNNPACK EP 경로 설명 (`execution_provider=xnnpack` 필드 활용)
  - "TFLite 제거 결정" 1문단 — Rescope 맥락 후대 독자에게 남김
  - Waves 테이블에 Wave 7 행 추가
- [ ] **Step 2**: `CLAUDE.md` Latest plan 링크 유지 (이미 Wave 7 plan 가리킴)
- [ ] **Step 3**: `README.md` 레시피 테이블 + ISA 매트릭스에 4장 row 추가
- [ ] **Step 4: Commit** — `docs: Wave 7 PyTorch PT2E + ORT XNNPACK sections`

---

## Task 7: Smoke E2E

**Why**: 실기 실행 없이 실제 이슈 예측 못 함 — Wave 6 Task 10이 ort_cpu_int8_static mAP=0 회귀를 잡은 사례.

**Files**: runtime only

- [ ] **Step 1**: Prerequisites — `best_qr.pt`, yaml들, R3 spike CLEARED
- [ ] **Step 2**: `#40 pytorch_pt2e_fp32` 단독 smoke → p50 / mAP / env / cold_start_ms 확인
- [ ] **Step 3**: `#42 ort_cpu_xnnpack_fp32` 단독 smoke — XNNPACK 실제 활성화 여부 `Result.notes`에서 확인
- [ ] **Step 4**: 전체 4장 batch 실행
- [ ] **Step 5**: `report_cpu_qr.md` 재생성 — #30-#35 + #40-#43 총 10장 ranking
- [ ] **Step 6: Result comparison**
  - Wave 6 winner (#35 openvino_int8_nncf 23.9 fps bs1)와 Wave 7 best 비교
  - **기대 시나리오 3종**:
    - (A) 둘 다 OpenVINO 근방 (±15%) → "backend diversity" 성공
    - (B) PT2E가 앞섬 → "PyTorch native가 경쟁력", 추천 플립 검토
    - (C) 둘 다 크게 뒤짐 → Wave 7은 모바일/확장 발판 가치로만 유지
- [ ] **Step 7**: Follow-up 정리
  - Wave 6 #33 회귀 재시도 (Task 3 Step 5b에서 해결되는 조합이 생겼는지 정리)
  - R3 spike에서 발견된 torch.export 제약 문서화
- [ ] **Step 7b**: Wave 8 ncnn feasibility micro-spike (시간 여유 시, 선택)
  - `pip install ncnn` 가능 여부 (Windows wheel)
  - `onnx2ncnn best_qr.onnx` 실행 → `.param` / `.bin` 파일 생성 확인 (YOLO26n attention block 호환 검증)
  - `ncnn.Net.load_param`로 1 forward smoke — 출력 shape `(1, 300, 6)` 보존
  - 결과(3단계 PASS/FAIL)를 `docs/improvements/2026-04-21-wave8-ncnn-feasibility.md`에 30줄 이내로 기록
  - **목적**: Wave 8 착수 시점에 Task 0 전체를 재투자 않도록 핵심 R2-유형 리스크만 선행 검증
- [ ] **Step 8: Commit results** — `feat(results): Wave 7 CPU eval + report (40-43)`

---

## Known Risks & Decisions

### R1. ~~TFLite Windows runtime 부재~~ — **ELIMINATED by rescope**
원안의 핵심 리스크. TFLite 제거로 완전 해소. Wave 8에서 모바일 배포가 필요하면 ExecutorTorch/ncnn으로 우회.

### R2. ~~onnx2tf의 YOLO26n attention 호환성~~ — **ELIMINATED by rescope**
ONNX → TFLite 변환기 자체가 제거됐으므로 무관. XNNPACK은 ORT가 직접 호출하므로 변환 없음.

### R3. `torch.export` + ultralytics dynamic forward 호환성
**Risk**: Brevitas fx-trace가 YOLO26n Python dynamic flow에 막혔던 전례. `torch.export`는 TorchDynamo 기반이라 fx보다 관대하나, ultralytics가 `self.inplace` 등 non-functional 구조를 쓰면 실패 가능.
**Mitigation**: Task 0 Step 2가 이 점을 검증. 실패 시 fallback 체계:
  - `torch.export.export(..., strict=False)` 시도
  - 실패 지속 시 PT2E 경로 parked → Wave 7을 XNNPACK 단독(#42/#43)으로 축소
  - Worst case: PT2E는 Wave 8로 미루고 Wave 7을 2장 recipe로 축소 — 그래도 TFLite 없이는 이미 충분히 단순.

### R4. PT2E 양자화 그래프의 `torch.compile` 레이턴시
**Risk**: `torch.compile` 첫 호출이 CPU에서도 수 초 걸릴 수 있음 — `cold_start_ms`가 FP32 baseline(~200ms)과 비교할 때 크게 튐. 또한 mode 선택을 잘못하면(예: `reduce-overhead`는 CUDA-graph 기반) CPU에서 오히려 느려짐.
**Mitigation**: Task 2는 CPU 기본 mode(`torch.compile(module)`) 사용, `reduce-overhead`/`max-autotune-no-cudagraphs`는 옵션으로만 언급. `measure_cold_start`가 첫 호출 레이턴시를 자연스럽게 기록하고 report.md는 `cold_start_ms`를 ranking 축에서 제외(Wave 6 결정)하므로 의사결정 영향 없음.

### R5. XNNPACK EP의 QDQ INT8 fallback (Task 0 Step 5에서 선행 검증)
**Risk**: XNNPACK EP가 특정 QDQ 노드에서 INT8 지원 안 하면 silently CPU EP로 떨어져 Wave 6 #33과 같은 성능이 나옴.
**Mitigation**: Task 0 Step 5가 Wave 6 static INT8 ONNX 하나를 XNNPACK EP로 로드해 `session.get_providers()` 반환값을 기록 — Wave 7 implementation 착수 전에 "fallback 100%인가 / 부분인가 / 전부 XNNPACK assign인가" 판정. Task 7 Step 3 smoke에서도 동일 확인을 recipe #43 결과에 notes로 기록.

### R6. Wave 6 #33 mAP=0 회귀가 XNNPACK에서도 재현될 가능성 (Task 3 Step 5b로 격리)
**Risk**: 같은 `quantize_static` ONNX를 재사용하면 activation scale 문제가 provider와 무관하게 재현.
**Mitigation**: **#43은 Wave 6 #33 ONNX를 재사용하지 않는다** — Task 3 Step 5b에서 A/B/C 조합 중 mAP 살아남는 파라미터로 새 ONNX 생성. Wave 6 #33은 별도 Wave 9+ follow-up으로 이관하고, Wave 7 #43은 독립적인 INT8 경로로 취급.

---

## Extension Gates (Wave 8+ roadmap)

평가 완료된 후보군(2026-04-21 검토):
- **MNN (Alibaba)** — **EXCLUDED**. ncnn과 기능 중복, 영어 문서 얇음, 특별 차별점 없음.
- **ExecutorTorch (Meta)** — **EXCLUDED** (2026 현재 alpha). PT2E(#41)가 이미 "PyTorch native 양자화" 목적 달성하므로 추가 가치 낮음. 2027+에 성숙도 재평가 후 재검토.

### Wave 8 (planned) — ncnn 모바일 편입
- ncnn은 **YOLO 모바일 배포의 사실상 표준** (YOLOv5/v8/NAS/26n 공식 예제 다수, ~500KB 런타임)
- 신규 recipe `#50 ncnn_fp32`, `#51 ncnn_int8_ptq`, (선택) `#52 ncnn_vulkan_fp16` — 번호 점프(#44-#49 buffer)로 hardware family 전환 시각화
- 파이프라인: ONNX → `onnx2ncnn` → `.param` + `.bin` → `ncnn.Net.load_param` + Python API로 forward
- 어댑터 `NcnnAsORT` — Wave 6 `OVRunnerAsORT` 패턴 재사용
- 리스크: `onnx2ncnn`의 YOLO26n attention block 호환 (Wave 3 INC / Wave 6 R2와 유사) — Task 0 spike 필요
- Mobile build는 Wave 8 범위 밖 (Android/iOS CI는 별도 인프라, Wave 10+)

### Wave 9 (planned) — DirectML EP (Windows Edge 전용)
- Windows 10/11 + DirectX 12 환경에서 **AMD Ryzen AI NPU**, **Intel Arc GPU**, **NVIDIA GPU**를 단일 API로 접근
- 설치는 단순(`pip install onnxruntime-directml` 한 줄). 단 기본 `onnxruntime` 패키지와 **패키지 슬롯 상호배타** — 공존 시나리오 두 가지:
  - (권장) **별도 venv 분리** — Wave 6/7 환경과 독립 유지, 안정성 최고
  - (옵션) **`pip install onnxruntime onnxruntime-directml --no-deps`** 후 런타임에 provider 스위칭 — 경험적으로 가능하나 공식 지원 아님. Wave 9 Task 0 spike에서 실기 확인 후 결정
- 신규 recipe `#60 dml_fp32`, `#61 dml_fp16`, `#62 dml_int8` — provider `DmlExecutionProvider`
- 이식성 트레이드오프: Linux/macOS 완전 미지원 → Wave 9를 **Windows edge 전용 wave**로 분리 정책
- 리스크: ORT DirectML이 QDQ INT8을 DML로 실행하는지, CPU fallback 발생 정도 — spike 필요

### 더 먼 후보 (Wave 10+)
- **Apple Core ML EP** — M-series Neural Engine offload. ORT에 EP로 붙음.
- **ORT XNNPACK on ARM** — Raspberry Pi 5 / Graviton / Apple M 테스트 베드 확보 후.
- **ExecutorTorch** — 2027+ API 안정 시 `#50 executorch_int8_x86` 같은 단일 recipe로 재평가.
- **INT4 weight-only** — Vision 연구 검증 시점.

---

## Dependencies on Wave 6

Wave 7은 Wave 6의 다음 산출물에 **의존**:
- `scripts/_weights_io.py::_iter_calib_samples` — PT2E INT8 calibration 공유
- `scripts/run_cpu.py::_prepare_cpu_session` dispatcher — branch 추가 지점
- `OVRunnerAsORT` 패턴 — `PytorchPT2EAsORT` 원본
- `scripts/env_lock.py::collect_env` — `cpu_flags`로 XNNPACK INT8 가용성 판정
- `LatencyStats.stddev_ms`, `MeasurementSpec.iter_cooldown_ms` — 동일 hygiene 적용
- `scripts/run_ort.py::_make_session` — `execution_provider` 필드 사용 패턴 참조

## Success Criteria (Wave 7 "active" 선언 조건)

1. **4장 recipe 전부 로드 가능** (`load_recipe` 통과)
2. **4장 중 3장 이상이 `meets_constraints=True`** — backend diversity 강제 포함 규칙:
   - PT2E 최소 1장 success (#40 또는 #41)
   - XNNPACK 최소 1장 success (#42 또는 #43)
   - 합쳐서 ≥ 3장이면 "Wave 7 active" 선언
   - 한 backend 전체가 park되면 Wave 7은 rescoped 상태로 기록 (수동 판단)
3. **`torch.export` R3이 해소** — Task 0 Step 2에서 YOLO26n export 성공 확인
4. **XNNPACK R5 실제 동작 확인** — Task 0 Step 5에서 `session.get_providers()` 반환에 XNNPACK이 assign됨을 실측
5. **report에 10장(#30-#35 + #40-#43) 통합 ranking** — Wave 6 기준선 유지
6. **추가 의존성 0** (`pyproject.toml` 변경 없음 확인)
7. **`pytest tests/ -q` green** 유지 — 기존 105 pass + Wave 7 신규 테스트 추가분 모두 green

Wave 6 benchmark: `openvino_int8_nncf` 23.9 fps(bs1). Wave 7 결과가 그 근처(±20%)면 "backend diversity 확보", 뚜렷이 앞서면 "추천 플립", 뒤처지면 "모바일 확장 발판" — Task 7 실측 후 확정.
