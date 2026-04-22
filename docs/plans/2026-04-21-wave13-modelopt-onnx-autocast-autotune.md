# Wave 13: ModelOpt ONNX AutoCast + Autotune (#27–#28) — **ARCHIVED 2026-04-21 (draft 당일, cross-verify에서 API 허구 확정)**

> **⛔ ARCHIVED**: `/gsd-plan-phase` 교차 검증에서 plan이 **존재하지 않는 API를 전제**하고 있었음이 확인됨.
>
> - **`modelopt.onnx.autotune` 모듈 자체가 존재하지 않음** (`ModuleNotFoundError` 확정). 실제 경로는 `modelopt.onnx.quantization.autotune` — INT8 Q/DQ placement 전용 저수준 API (`QDQAutotuner` + `TrtExecBenchmark` 조립) 로 plan 의도(FP16 builder sweep)와 **목적 자체가 다름**.
> - `convert_to_f16` 첫 인자는 `onnx.ModelProto`지 경로 아님. `sensitivity_samples` / `num_samples` 파라미터는 존재하지 않음. `calibration_data`는 파일 경로(npz). Plan의 모든 호출이 즉시 `TypeError`.
> - Recipe YAML은 `model.weights: yolo26n.pt` (COCO 가중치)인데 "base ONNX = `best_qr_640_fp32_dyn.onnx`" 라고 서술 — `OMNI_WEIGHTS_OVERRIDE` 필수 컨벤션 누락.
> - `op_block_list: ["Softmax", "LayerNormalization"]` — YOLO26n에 `LayerNormalization` 없음 (BatchNorm). LLM 예제 복붙 흔적.
> - `sensitivity_samples: 32`는 Wave 5의 "QR val 133장 부족 → COCO 512장 전환" 결정에 역행. 게다가 API 자체가 샘플 카운트 파라미터를 노출하지 않음.
>
> 자세한 교차 검증 결과: [`docs/improvements/2026-04-21-wave13-api-discovery-blocker.md`](../improvements/2026-04-21-wave13-api-discovery-blocker.md).
>
> **후속 방향**: [`Wave 9 DirectML EP`](./wave9-directml.md) (미작성) 로 이동. ORT 네이티브 경로이므로 Wave 7/8/13의 "외부 API 호환성" 반복 리스크 회피. autocast/autotune은 향후 modelopt 공식 CNN 지원이 확인되는 시점에 재평가.
>
> **교훈**: 외부 라이브러리의 최신 API 서명은 **GitHub 소스 검증이 필수**. 공식 문서 페이지 / ChatGPT 답변만으로 plan을 쓰면 허구 API가 그대로 남음. Wave 13이 이 전형적 함정 케이스.

---

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:executing-plans`. 체크박스(`- [ ]`)가 진행 트래커. 본 Wave는 **low-risk filler** — Wave 10/12보다 작은 표면, 주로 ONNX graph 수준 pass 실험.

> **⚠ 이하 원본 plan 보존용. ARCHIVED 이후 실행 금지.**

**Goal:** `modelopt.onnx.autocast` (자동 mixed-precision ONNX 변환) 과 `modelopt.onnx.autotune` (TRT 빌드 파라미터 sweep) 2가지 ONNX-level pass를 GPU recipe bank에 편입한다. Wave 7/8이 반복적으로 막혔던 "외부 변환기 호환성" 이슈와 무관 — ONNX graph rewriting은 `run_trt.py` 이전 단계에서 순수 함수적으로 적용되므로 YOLO26n Detect head의 anchor mutation 패턴 영향 없음.

**Architecture:** 변경 지점 4곳. (1) `scripts/_schemas.py::TechniqueSpec`에 `onnx_pass: Optional[Literal["autocast", "autotune"]] = None` 추가. (2) `scripts/run_trt.py` ONNX export 직후, TRT build 직전에 분기 — `onnx_pass` 지정 시 `modelopt.onnx.autocast.convert_to_f16(...)` 또는 `modelopt.onnx.autotune.autotune(...)` 호출. (3) `recipes/27,28_*.yaml` 신규. (4) `Result.notes`에 autocast convert log / autotune 선택된 builder params 기록. `run_ort.py`, `run_cpu.py`, 기존 modelopt modifier는 **건드리지 않는다**.

**Tech Stack:** `nvidia-modelopt>=0.15` (onnx 서브모듈), `onnx>=1.15`, `tensorrt>=10.x`. CUDA 환경 필요 (autotune은 실제 TRT build 루프 돌림).

---

## Scope Boundaries

### In-scope (Wave 13)
- **AutoCast**: FP32 ONNX → FP16/BF16 mixed-precision ONNX. Sensitivity-aware node exclusion 자동 수행
- **Autotune**: 주어진 ONNX + TRT builder config sweep (workspace 크기, layer 선호, precision constraints)
- **타겟**: YOLO26n fp32 ONNX (Wave 1 `#00 trt_fp32` 에서 이미 생성되는 asset 재활용)
- **타겟 HW**: RTX 3060 Laptop

### Out-of-scope
- **ONNX 모델 재훈련** — pure graph rewriting만
- **Autotune latency target** sweep — 기본 "min latency" objective만 사용
- **Multi-GPU autotune** — single GPU

### Assumptions
- `modelopt.onnx.autocast`의 sensitivity analysis 단계는 한 번의 validation forward pass를 요구 — QR dataset 몇 장으로 충분 (Wave 5 인프라 재사용)
- Autotune이 기존 `run_trt.py`의 수동 builder config와 충돌하지 않는다 (autotune이 최종 builder config를 외부로 export하거나 engine 파일로 직접 빌드)

---

## Recipe Map (#27–#28)

| # | name | source | dtype | onnx_pass | 기반 ONNX |
|---|---|---|---|---|---|
| 27 | `modelopt_onnx_autocast_fp16` | `modelopt` | fp16 | `autocast` | `best_qr_640_fp32_dyn.onnx` (Wave 6 asset 재사용) |
| 28 | `modelopt_onnx_autotune_fp16` | `modelopt` | fp16 | `autotune` | 동일 |

두 recipe 모두 **TRT FP16 baseline (`#05 trt_fp16`)** 과 직접 비교 가능하도록 동일 dtype 설정. Autocast는 "자동 fp16" → 정확도 재확보, Autotune은 "자동 builder tuning" → latency 최적화.

**Parked**:
- `#27b autocast_bf16` — Ampere BF16 TRT 경로, autocast가 지원하면 활성화
- `#28b autotune_int8` — autotune이 INT8 QDQ ONNX에 적용 가능한지 Task 3 결과 후 판단

---

## Task Dependency Graph

```
Task 1 (schema 확장 + run_trt.py 분기)
   │
   ├─> Task 2 (autocast recipe #27)
   │       │
   │       v
   │  Task 4 (eval + report)
   │
   └─> Task 3 (autotune recipe #28)
           │
           v
      Task 4 (eval + report)
```

Task 2/3는 독립 (다른 modelopt API 호출). 병렬 실행 가능.

---

## Task 1 — Schema 확장 + `run_trt.py` 분기

- [ ] `scripts/_schemas.py::TechniqueSpec`에 필드 추가:
  ```python
  onnx_pass: Optional[Literal["autocast", "autotune"]] = None
  onnx_pass_config: Optional[dict] = None  # 자유 key-value, pass 별 하이퍼파라미터
  ```
- [ ] `scripts/run_trt.py`의 ONNX 준비 단계 직후(build 직전)에 hook:
  ```python
  if recipe.technique.onnx_pass == "autocast":
      from modelopt.onnx.autocast import convert_to_f16
      onnx_path_fp16 = convert_to_f16(onnx_path_fp32, ...)
      onnx_path = onnx_path_fp16
  elif recipe.technique.onnx_pass == "autotune":
      from modelopt.onnx.autotune import autotune
      best_engine = autotune(onnx_path, ...)
      # autotune이 engine 직접 반환 시 기존 TRT build 건너뛰기
  ```
- [ ] `Result.notes`에 다음 기록 필수:
  - autocast: `autocast_fp16_nodes=<N>`, `autocast_fp32_nodes=<M>`, `autocast_sensitivity_samples=<K>`
  - autotune: `autotune_trials=<N>`, `autotune_best_config=<dict summary>`
- [ ] `pytest tests/test_run_cpu_imports_without_tensorrt` 재실행 — modelopt.onnx import가 CPU 환경 imports를 깨뜨리지 않는지 확인 (`run_trt.py` 내부 분기에만 있어야 함)

---

## Task 2 — AutoCast recipe (#27)

- [ ] `recipes/27_modelopt_onnx_autocast_fp16.yaml`:
  ```yaml
  name: modelopt_onnx_autocast_fp16
  model: {family: yolo26, variant: n, weights: yolo26n.pt}
  runtime: {engine: tensorrt, dtype: fp16}
  technique:
    name: onnx_autocast_fp16
    source: modelopt
    onnx_pass: autocast
    onnx_pass_config:
      sensitivity_samples: 32          # QR val set에서 32장
      op_block_list: ["Softmax", "LayerNormalization"]  # 안전 제외
  hardware: {requires_compute_capability_min: 7.0}
  measurement: {dataset: coco_val2017, num_images: 500, warmup_iters: 100,
                measure_iters: 100, batch_sizes: [1, 8], input_size: 640,
                gpu_clock_lock: true, seed: 42}
  constraints: {max_map_drop_pct: 0.5, min_fps_bs1: 30}
  ```
- [ ] `make recipe-27` 타깃 + 실행
- [ ] 비교 대상: `#05 trt_fp16`. autocast가 더 높은 mAP를 내면 sensitivity-aware 선택이 이득, fps가 비슷하거나 떨어지면 "accuracy 우위만"으로 활용 판단

---

## Task 3 — Autotune recipe (#28)

- [ ] `recipes/28_modelopt_onnx_autotune_fp16.yaml`:
  ```yaml
  name: modelopt_onnx_autotune_fp16
  model: {family: yolo26, variant: n, weights: yolo26n.pt}
  runtime: {engine: tensorrt, dtype: fp16}
  technique:
    name: onnx_autotune_fp16
    source: modelopt
    onnx_pass: autotune
    onnx_pass_config:
      num_trials: 8
      objective: "min_latency"
      allow_fp16: true
      allow_int8: false
  hardware: {requires_compute_capability_min: 7.0}
  measurement: {dataset: coco_val2017, num_images: 500, warmup_iters: 100,
                measure_iters: 100, batch_sizes: [1, 8], input_size: 640,
                gpu_clock_lock: true, seed: 42}
  constraints: {max_map_drop_pct: 0.5, min_fps_bs1: 30}
  ```
- [ ] `make recipe-28` 타깃 + 실행
- [ ] 비교 대상: `#05 trt_fp16`. autotune이 더 빠른 fps를 내면 "builder param 수동 튜닝 불필요" 증명
- [ ] 주의: autotune은 **순수 inference time 측정**이므로 mAP drop은 거의 없어야 함 (>0.5%p 차이 시 autotune 내부가 FP16으로 과도하게 변환한 것 — sensitivity 분석 필요)

---

## Task 4 — Report + 문서 갱신

- [ ] `report_qr.md` 재생성 (#27/#28 추가)
- [ ] `docs/architecture.md` "Recipes" 섹션에 ONNX pass 흐름 1단락 추가
- [ ] `CLAUDE.md` "Current scope" 업데이트 (recipe 카운트 증가)
- [ ] 결과에 따라 다음 분기:
  - **둘 다 ranking Top-10 진입** → Wave 1~5에서 쓴 TRT 빌드 파라미터 수동 튜닝 로직을 autotune으로 대체하는 Wave 14 후보 기록
  - **autocast는 유효, autotune은 미미** → autocast만 채택, autotune은 parked
  - **둘 다 미미** → `docs/improvements/2026-04-21-wave13-onnx-pass-null-result.md` 작성 후 recipe 삭제

---

## Definition of Done

1. `pytest tests/ -q` green, `mypy scripts ...` green
2. `results/27_*.json`, `results/28_*.json` 생성 (성공 또는 graceful fail)
3. `Result.notes`에 autocast/autotune 메타데이터 기록
4. `report_qr.md` 갱신 (성공 시) 또는 null-result 문서 (실패 시)
5. `test_run_cpu_imports_without_tensorrt` 통과 유지 — modelopt.onnx import가 CPU 환경에 leak 안 됨

---

## Rollback

두 pass 모두 실패 시:
1. `recipes/27_*.yaml`, `recipes/28_*.yaml` 삭제
2. `_schemas.py::TechniqueSpec.onnx_pass` 필드 제거
3. `run_trt.py` 분기 revert
4. 이 plan ARCHIVED + `docs/improvements/2026-04-21-wave13-onnx-pass-null-result.md` 작성
