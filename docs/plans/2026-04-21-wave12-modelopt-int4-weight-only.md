# Wave 12: ModelOpt INT4 Weight-only 실험 레시피 (#26) — **ARCHIVED 2026-04-21 (draft 당일, cross-verify에서 실패 확정)**

> **⛔ ARCHIVED**: `/gsd-plan-phase` 교차 검증에서 **공식 문서로 실패 확정**. 실행 없이 분석 문서로 전환.
>
> - NVIDIA TensorRT 10 공식 문서: "WoQ is available only for INT4 block quantization with **GEMM layers**". YOLO26n은 거의 전부 Conv2d이므로 Ampere에서 INT4 weight-only로 의미 있는 benchmark 불가능.
> - `modelopt.INT4_BLOCKWISE_WEIGHT_ONLY_CFG`의 default `disable_conv_quantization`가 Conv2d를 silently skip할 가능성 높음 → `modelopt_qat.apply`의 `wrapped==0` guard가 apply 단계에서 즉시 RuntimeError 트리거 예상. 실행이 "negative result 수집"이 아니라 "공식 문서 재확인" 수준이 됨.
> - 추가로 `run_trt.py::_prepare_onnx`의 Python 예외를 catch하는 graceful degrade 경로 부재(BLOCKER) + `epochs: 0` 파이프라인 지원 부재(BLOCKER)가 실행 비용을 더 늘림.
>
> 자세한 교차 검증 결과와 공식 문서 인용: [`docs/improvements/2026-04-21-int4-ampere-not-feasible.md`](../improvements/2026-04-21-int4-ampere-not-feasible.md).
>
> **후속 방향**: FP8 (Ada SM 8.9+) 또는 NVFP4 (Blackwell SM 10.0+) 하드웨어 확보 후 재개. Ampere SM 8.6에서는 CNN × INT4 조합 parked.
>
> **Wave 3/7/8 패턴 교훈 연장**: 외부 quantization config가 LLM/MatMul 전제로 설계된 경우 CNN 경로에서 반복 좌초. 다음 wave는 **ONNX 네이티브 경로** (DirectML EP, ORT INT8 etc) 우선.

---

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:executing-plans`. 체크박스(`- [ ]`)가 진행 트래커. 본 Wave는 **negative-result 실험 성격** — 실패해도 결과를 문서화하는 것이 목표.

> **⚠ 이하 원본 plan 보존용. ARCHIVED 이후 실행 금지.**

**Goal:** `INT4_BLOCKWISE_WEIGHT_ONLY_CFG` (또는 `INT4_AWQ_CFG`)를 YOLO26n에 적용했을 때 **Ampere GPU에서 실행 가능한지**, accuracy drop이 허용 가능한 수준(mAP drop ≤ 3%p) 인지 empirical하게 확인한다. NVIDIA 공식 가이드가 "conv layer에는 blockwise 비권장"이라고 명시했으므로 기대치는 낮으나, negative result 자체가 "향후 INT4 weight-only는 이 모델에선 포기" 라는 결론을 내릴 근거가 된다.

**Architecture:** 변경 지점 3곳. (1) `scripts/_schemas.py::TrainingSpec.quant_config` preset 매핑 확장 — `int4_blockwise`와 `int4_awq` 키 허용. (2) `scripts/_modifiers/modelopt_qat.py::_QUANT_CONFIG_PRESETS` 딕셔너리에 동일 키 추가 (기존 인프라 재사용, 신규 modifier 불필요). (3) `recipes/26_modelopt_int4_blockwise_weight_only.yaml` 신규. `run_trt.py`는 TRT 10.x가 INT4 QDQ MatMul을 native 지원하는지 런타임에서 확인 — 지원 안 하면 `Result.notes`에 기록하고 `meets_constraints=False`로 degrade (crash 금지).

**Tech Stack:** `nvidia-modelopt>=0.15`, `tensorrt>=10.x` (INT4 MatMul 지원은 10.x부터), `torch>=2.8`.

---

## Scope Boundaries

### In-scope (Wave 12)
- **Config**: `INT4_BLOCKWISE_WEIGHT_ONLY_CFG` (primary). `INT4_AWQ_CFG`는 secondary — Task 3에서 primary 실패 시 사이드 실험
- **학습**: PTQ만. Fine-tune 없음 (weight-only라 scale만 잡으면 됨)
- **타겟 HW**: RTX 3060 Laptop (SM 8.6). TRT 10.x cuBLAS W4A16 MatMul 경로 필요

### Out-of-scope
- **QAT INT4** — 학습 파이프라인 비용 대비 이득 불명
- **FP8 act + INT4 weight (W4A8)** — Ampere는 FP8 native 미지원 → `W4A8_AWQ_BETA_CFG` 제외
- **Activation quantization 혼합** (W4A4 등) — 본 Wave 제외, 결과 보고 판단
- **Distillation 복구** — Wave 11 제외 결정에 따라 제외

### Assumptions
- TRT 10.x가 YOLO26n 크기(~4MB)의 Conv 기반 네트워크에서 INT4 weight-only QDQ를 파싱한다. 못 파싱하면 Task 2 Step 3에서 탐지 후 degrade.
- `INT4_BLOCKWISE_WEIGHT_ONLY_CFG`의 기본 block size (128)가 YOLO26n의 Conv weight shape (`in_channels` 종종 16/32)과 맞물린다. 안 맞으면 block size 64/32 튜닝 필요.

---

## Recipe Map (#26)

| # | name | source | dtype | config preset | 비고 |
|---|---|---|---|---|---|
| 26 | `modelopt_int4_blockwise_weight_only` | `modelopt` | int8 (runtime 표기) | `INT4_BLOCKWISE_WEIGHT_ONLY_CFG` | Weights INT4, Activations FP16 |

**주의**: `runtime.dtype`은 스키마상 `int8`로 설정 (Literal 제약). 실제 weight bit는 `technique.notes`와 calibrator 이름에서 구분. 스키마 확장은 하지 않음 (단일 실험 recipe에 대한 과투자).

**Parked** (#26 결과에 따라):
- `#26b modelopt_int4_awq` — Ampere 공식 권장이지만 LLM 전제 config. Conv에서 유효한지 #26 결과 보고 판단

---

## Task Dependency Graph

```
Task 1 (config preset 매핑) ──> Task 2 (recipe YAML) ──> Task 3 (E2E 실행 + TRT 파싱 검증)
                                                                │
                                                                v
                                                     Task 4 (문서화 / negative result 정리)
```

Task 0 spike 불필요 — 기존 modelopt_qat 인프라 재사용하므로 진입 리스크 낮음. 실패 지점은 TRT 파싱(Task 3)에서 확인되고 graceful degrade로 처리.

---

## Task 1 — Config preset 매핑 확장

- [ ] `scripts/_modifiers/modelopt_qat.py::_QUANT_CONFIG_PRESETS`에 엔트리 추가:
  ```python
  _QUANT_CONFIG_PRESETS = {
      "int8_default": "INT8_DEFAULT_CFG",
      "int4_blockwise": "INT4_BLOCKWISE_WEIGHT_ONLY_CFG",
      "int4_awq": "INT4_AWQ_CFG",
  }
  ```
- [ ] `apply()`의 guard 로직 재점검 — INT4 config는 Conv를 감쌀 때 기존 "quant" substring 체크에 걸리는지 확인 (modelopt 버전에 따라 클래스명 달라짐). 필요 시 guard를 `hasattr(m, "weight_quantizer")` 로 강화
- [ ] `tests/test_modelopt_qat_presets.py` (있으면) 에 2개 preset 추가 테스트

---

## Task 2 — Recipe YAML

- [ ] `recipes/26_modelopt_int4_blockwise_weight_only.yaml`:
  ```yaml
  name: modelopt_int4_blockwise_weight_only
  model: {family: yolo26, variant: n, weights: yolo26n.pt}
  runtime: {engine: tensorrt, dtype: int8}   # 스키마 제약; 실제 W4A16
  technique:
    name: int4_weight_only
    source: modelopt
    calibrator: minmax                       # weight-only는 activation 캘리브 불필요
    calibration_samples: 64
    calibration_dataset: coco_val2017
    training:
      base_checkpoint: best_qr.pt
      epochs: 0                              # PTQ only — train.py는 epochs=0일 때 skip
      modifier: modelopt_qat
      quant_config: int4_blockwise
      data_yaml: qr_barcode.yaml
  hardware:
    requires_compute_capability_min: 8.0
  measurement: {dataset: coco_val2017, num_images: 500, warmup_iters: 100,
                measure_iters: 100, batch_sizes: [1, 8], input_size: 640,
                gpu_clock_lock: true, seed: 42}
  constraints: {max_map_drop_pct: 3.0, min_fps_bs1: 30}  # 관대한 drop 허용
  ```
- [ ] `epochs: 0`일 때 train.py가 학습 skip 후 finalize만 호출하는지 확인. 없으면 파이프라인 짧게 확장 (1 batch forward → `mtp.quantize` calibrate → save)
- [ ] `pytest tests/test_schemas.py -q` 통과

---

## Task 3 — E2E 실행 + TRT 파싱 검증

- [ ] `make recipe-26` 추가 + 실행
- [ ] 3-way 관찰:
  - (A) **TRT build 성공 + mAP drop ≤ 3%p** → 성공, `Result.notes`에 model_size_mb, block_size 기록
  - (B) **TRT build 실패** (`INT4 MatMul not supported` 등) → `run_trt.py`가 exception을 catch해서 `meets_constraints=False` + `notes="TRT 10.x INT4 parse failed: <err>"`로 기록하는지 확인. 크래시 시 `run_trt.py` error-handling 보강
  - (C) **TRT 성공 but mAP drop > 3%p** → archive, Task 4에서 negative result 문서화
- [ ] `Result.notes`에 다음 필드 반드시 기록:
  - `weight_bits=4`, `block_size=<N>`, `int4_algo=blockwise|awq`
  - `trt_version=<x.y.z>`, `trt_int4_matmul_supported=true|false`

---

## Task 4 — 문서화 (성공/실패 무관 필수)

- [ ] `docs/improvements/2026-04-21-wave12-int4-results.md` 작성:
  - 시나리오 (A/B/C) 및 수치
  - NVIDIA 가이드("conv blockwise 비권장")와 실측 비교
  - INT4 AWQ (#26b) 진행 여부 권고
- [ ] `report_qr.md` 재생성 (성공 시 신규 행, 실패 시 Issues 섹션)
- [ ] Scenario (A) 시 `CLAUDE.md` "Critical conventions" 갱신 불필요 (추가 convention 없음)
- [ ] Scenario (B/C) 시 이 plan에 `ARCHIVED` 스탬프 추가

---

## Definition of Done

1. `pytest tests/ -q` green
2. `results/26_*.json` 생성 + `Result.notes`에 weight_bits / block_size / trt_int4_matmul_supported 기록
3. `docs/improvements/2026-04-21-wave12-int4-results.md` 작성 완료
4. Scenario (A)면 `report_qr.md` 갱신, (B/C)면 plan ARCHIVED 마킹

---

## Rollback

전부 crash + graceful degrade 실패 시:
1. `recipes/26_*.yaml` 삭제
2. `_modifiers/modelopt_qat.py::_QUANT_CONFIG_PRESETS` revert
3. 이 plan에 ARCHIVED 스탬프 + 원인 문서화
4. Wave 13으로 이동
