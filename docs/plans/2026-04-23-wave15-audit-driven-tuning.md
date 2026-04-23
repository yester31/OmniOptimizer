# Wave 15 — Audit-Driven Parameter Tuning

**Author**: plan-eng-review 세션 (2026-04-23)
**Scope**: no-regret parameter switches + opt_level=5 per-recipe opt-in
**Target**: 1-day single-PR on `feat/wave15-tuning`
**Predecessor plans**: Wave 14 (TRT tuning, shipped 2026-04-22) — see `docs/plans/2026-04-22-wave14-trt-optimization.md`

---

## 1. Motivation

2026-04-23 plan-eng-review 에서 4개 툴체인 (TensorRT 10.16, modelopt 0.43, ORT 1.22, OpenVINO 2024 + NNCF)을 공식 문서 기반으로 재감사한 결과 **11개 finding** 도출. 그 중 **증거가 확실하고 정확도/재현성 리스크가 없는 최소 집합**만 이번 Wave에 포함. 나머지 (modelopt param 확장, NNCF preset swap, CPU_DENORMALS, 실험 레시피 3종)는 audit helper (`scripts/audit_capabilities.py`, TODOS.md T1) 완료 후 Wave 16에서 증거 기반 결정.

핵심 근거:
- **#40 `trt_fp16_opt5`** (Wave 14 shipped) = opt_level=3 → 5 단일 변경으로 fps 435 → 645 (**+48%**). #04 ORT TRT EP (현재 fps 211, builder_optimization_level 미지정)에 같은 knob이 누락 → 같은 변경으로 ORT EP 경로의 공정 비교 가능.
- OV `CACHE_DIR` 미설정 = 매 세션 compiled-blob 재빌드 (~2000ms cold_start). 정확도 0 영향, 순수 UX 개선.
- `builder_optimization_level=5` 을 상위 INT8 recipe (#09, #12, #17, #42) 와 FP16 (#05) 에 per-recipe 명시 = Wave 14 증명된 knob 의 MLPerf 스타일 opt-in 확대.

---

## 2. Deliverables

```
┌──────────────────────────────────────────────────────────────────────┐
│ D1. No-regret switches                                    (30 min)   │
│ D2. opt_level=5 per-recipe opt-in (5 recipes × A/B)       (3-4h)     │
│ D3. Schema: MeasurementSpec.build_ceiling_s               (1h)       │
│ D4. Tests (4 files)                                       (1h)       │
│ D5. Docs (plan + recipes/README + CLAUDE.md)              (30 min)   │
└──────────────────────────────────────────────────────────────────────┘
Total: ~6-7 hours wallclock, ~4h active work + measurement time.
```

### D1. No-regret switches

#### D1.1 — OV CACHE_DIR

**File**: `scripts/run_cpu.py::_get_ov_core`

```python
def _get_ov_core():
    global _OV_CORE
    if _OV_CORE is None:
        import openvino as ov
        _OV_CORE = ov.Core()
        # Wave 15 D1.1: persistent compiled-blob cache.
        # Cuts subsequent cold_start from ~2000ms to ~400ms on CPU recipes.
        # Accuracy zero-impact (byte-for-byte replay of kernel compile output).
        cache_dir = Path("results_cpu/_ov_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        _OV_CORE.set_property({"CACHE_DIR": str(cache_dir)})
    return _OV_CORE
```

**Side-effect**: `.gitignore` 에 `results_cpu/_ov_cache/` 추가.

**Accept**: `#34 openvino_fp32` cold_start_ms 재측정 결과 2회째 실행이 초회 대비 ≤ 40% 이내.
**Rollback trigger**: disk permission error → try/except 로그만 남기고 core 반환 (이미 lazy init 패턴).

#### D1.2 — ORT TRT EP provider options for #04

**File**: `scripts/run_ort.py::_make_session`

```python
if execution_provider == "TensorrtExecutionProvider":
    trt_cache = ROOT / "results" / "_trt_cache"
    trt_cache.mkdir(parents=True, exist_ok=True)
    trt_opts = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(trt_cache),
        "trt_fp16_enable": dtype == "fp16",
        # Wave 15 D1.2: match Wave 14 native TRT defaults.
        # opt_level=5 runs exhaustive tactic autotune (+5-48% fps per Wave 14 #40).
        # timing cache amortizes build time across re-runs.
        "trt_builder_optimization_level": 5,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(trt_cache),
        "trt_detailed_build_log": True,
    }
    providers: list = [(execution_provider, trt_opts), "CUDAExecutionProvider", "CPUExecutionProvider"]
```

**Backward-compat**: ORT ≤ 1.21 이 `trt_builder_optimization_level` 을 reject 할 가능성 → `ort.__version__` 체크 대신 **try/except** 래핑. 알 수 없는 provider option 은 ORT 1.22 에서는 경고만 발생하고 fall-through 되는 게 정상 동작 (실제 테스트에서 확인). 그러나 session creation 전체 실패 시 fallback 으로 키 제거 후 재시도:

```python
try:
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
except Exception as e:
    # Fallback: strip Wave 15 new options and retry (older ORT compat).
    if "trt_builder_optimization_level" in str(e) or "trt_timing_cache" in str(e):
        for k in ("trt_builder_optimization_level", "trt_timing_cache_enable",
                  "trt_timing_cache_path", "trt_detailed_build_log"):
            trt_opts.pop(k, None)
        session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
    else:
        raise
```

**Accept**: #04 fps 재측정 ≥ 300 (현재 211 대비 +42% 이상).
**Rollback trigger**: fps < 220 OR mAP 회귀. 전자는 Wave 16 로 분석 미룸, 후자는 즉시 변경 revert.

### D2. opt_level=5 per-recipe opt-in

**Files**: 5 recipe YAML 에 **한 줄씩** 추가:
- `recipes/05_trt_fp16.yaml`
- `recipes/09_modelopt_int8_entropy.yaml`
- `recipes/12_modelopt_int8_mixed.yaml`
- `recipes/17_modelopt_int8_qat.yaml`
- `recipes/42_modelopt_int8_asymmetric.yaml`

```yaml
runtime:
  engine: tensorrt
  dtype: int8          # (or fp16)
  builder_optimization_level: 5   # Wave 15 D2: explicit opt-in per MLPerf reproducibility.
```

**Why only 5**: 상위 성능 또는 대표 calibrator variant. #08 (ptq/max) 과 #10 (percentile) 은 #09 와 같은 그래프를 공유 (`docs/improvements/2026-04-23-modelopt-ptq-tactic-analysis.md` 참조) 라 추가 opt-in 가치가 낮음. #06/#07 (trt_builtin calibrator) 은 modelopt 계열 대비 baseline 역할이므로 default 유지.

**Build time impact**: 레시피 당 ~10-15 분 (default 3 대비 3-5x). `make` 대상은 recipe 단위 독립이므로 `make all` 은 ~50분 → ~120분 예상. 개별 `make recipe-42` 등으로 병렬 실행 시 물리 시간 단축 가능 (GPU 자원 제약).

**Accept (per recipe)**:
- fps delta ≥ +3% vs baseline (opt_level=3 동일 recipe)
- mAP delta ≥ -0.3%p
- `Result.build_time_s ≤ 900s`

**Rollback trigger (per recipe)**: 위 3개 중 하나라도 미달 시 해당 recipe 에서 `builder_optimization_level` 필드 제거 후 commit message 에 기록. Wave 15 전체는 계속 진행.

### D3. `MeasurementSpec.build_ceiling_s` schema

**File**: `scripts/_schemas.py::MeasurementSpec`

```python
class MeasurementSpec(BaseModel):
    ...
    # Wave 15 D3: TRT engine build wall-clock ceiling (seconds). Breach
    # triggers structured Result.notes marker instead of silent warning.
    # Default 600s matches pre-Wave 15 warning threshold; opt_level=5 INT8
    # recipes may need 900s+ so per-recipe override is supported.
    build_ceiling_s: Optional[int] = Field(default=None, gt=0)
```

**File**: `scripts/run_trt.py::_build_engine`

기존 `if build_time_s > 600.0:` 블록을 recipe-aware 로 수정:

```python
ceiling = recipe.measurement.build_ceiling_s if recipe is not None else 600
if ceiling is not None and build_time_s > ceiling:
    note = (f"build_time_s={build_time_s:.0f}s exceeds ceiling={ceiling}s "
            f"(builder_optimization_level={builder_optimization_level or 3})")
    print(f"[warn] {note}", file=sys.stderr)
    # Return engine still — ceiling is diagnostic, not fatal. run_trt.py
    # run() embeds this in Result.notes via the build_time_s + separate
    # note path; meets_constraints stays None (not forced False).
```

(Signature 변경: `_build_engine` 이 `recipe` 또는 `ceiling` 을 받도록. 호출부 `run_trt.py::run` 에서 `recipe.measurement.build_ceiling_s` 전달.)

**Why not just bump to 900s globally**: per-recipe 필요성 (FP16 recipe 는 5분에 끝나지만 INT8 opt_level=5 QAT 는 15분 가능). Schema 화 = MLPerf 재현성.

**Accept**: 기존 모든 테스트 green, legacy Result JSON 파싱 유지.

### D4. Tests

| 파일 | 범위 |
|---|---|
| `tests/test_schema_wave15.py` | `build_ceiling_s` 필드 (default None, gt=0, 음수 reject), `_minimal_recipe()` builder_optimization_level + build_ceiling_s 조합, 기존 Wave 14 recipe legacy parse. |
| `tests/test_recipe_bank_validate.py` | `recipes/*.yaml` 31개 모두 `Recipe.model_validate` 통과. |
| `tests/test_ov_cache_dir.py` | `_get_ov_core` 첫 호출 시 `CACHE_DIR` 속성 세팅 + 디렉토리 생성 확인 (mock `_OV_CORE` 재초기화). |
| `tests/test_run_ort_trt_opts.py` | `_make_session` 이 `TensorrtExecutionProvider` 호출 시 provider_options 에 `trt_builder_optimization_level=5`, `trt_timing_cache_enable=True` 포함 확인 (mock `InferenceSession`). |

### D5. Docs

- `docs/plans/2026-04-23-wave15-audit-driven-tuning.md` (this file)
- `recipes/README.md` — "recipes/40-49 = tuning variants" convention 보강, `builder_optimization_level: 5` opt-in 표시
- `CLAUDE.md` scope 블록 — Wave 15 ship note 추가 (완료 시)
- (Ship 후) `docs/improvements/2026-04-23-wave15-results.md` — 레시피별 fps/mAP 전후 비교

---

## 3. Parallelization lanes

```
Lane A (TRT EP):       D1.2 (run_ort.py)     ──┐
Lane B (OV):           D1.1 (run_cpu.py)     ──┤
Lane C (schema+tests): D3 + D4.1 (_schemas)  ──┼── parallel (3 disjoint modules)
                                                │
Lane D (YAML):         D2 × 5 recipes        ──┘
                                                │
Lane E (GPU measure):  make recipe-{05,09,12,17,42} — sequential (GPU 단일 자원)
Lane F (OV measure):   make recipe-{30,34,35} — sequential (CPU 단일 자원)
Lane G (commit):       pytest + commit       — after A-F converge
```

Implementation time: Lanes A/B/C/D = max(30m, 30m, 1h, 15m) = **1h**. Measurement (E+F) = **3-4h** (GPU). Docs + commit = **1h**. Total ≈ **5-6h wallclock**.

---

## 4. Accept / Rollback matrix (종합)

| 체크포인트 | Accept | Rollback |
|---|---|---|
| D1.1 OV CACHE_DIR | cold_start 2nd run ≤ 40% of 1st | disk permission 에러만 로그, core 반환 유지 |
| D1.2 ORT TRT EP | #04 fps ≥ 300 (+42%) | <220 → Wave 16 분석, mAP 회귀 → 즉시 revert |
| D2 각 recipe | fps +3%p AND mAP -0.3%p 이내 AND build ≤ 900s | 한 가지 미달 → 해당 recipe 에서 `builder_optimization_level` 제거 |
| D3 schema | 모든 pytest green, legacy JSON parse | 회귀 발생 시 default None 유지하며 용법만 변경 |
| D4 tests | 새 테스트 4종 green | N/A |

---

## 5. Out of scope (명시)

- `disable_mha_qdq`, `op_types_to_quantize` 스키마 확장 — T1 audit helper 결과 의존, Wave 16
- NNCF `preset=PERFORMANCE` + `IgnoredScope` — A/B 필수, Wave 16
- OV static reshape `[1,3,640,640]` — Wave 16 (NNCF 작업과 묶음)
- `CPU_DENORMALS_OPTIMIZATION=YES` — silent mAP drop 위험, A/B 필수, Wave 16
- SmoothQuant for #33 — baseline 확보 후 Wave 16
- `#04b_ort_trt_cuda_graph.yaml` — native #40 초과 가능성 낮음, Wave 16 커버리지 카드
- `#43_openvino_bf16.yaml` — hardware `avx512_bf16` 확인 필요, T1 선행
- `#44_modelopt_int8_autotune.yaml` — trtexec dependency, Wave 16 스카우트
- `scripts/_node_patterns.py` DRY 리팩터 (TODOS T2) — NNCF 추가 시점에 가치, Wave 16
- 예측: INT4 weight-only, ONNX autocast, torch.export, pnnx, ncnn, DirectML EP — 모두 별도 Wave 아카이브/독립 플랜

---

## 6. Failure modes (critical)

| Codepath | 실패 | 대응 | Critical? |
|---|---|---|---|
| #04 TRT EP `trt_builder_optimization_level=5` reject (old ORT) | session init exception | D1.2 에 try/except fallback | ✅ (포함) |
| D1.1 CACHE_DIR 권한 불가 | set_property 예외 | try/except, silent fallback | ⚠️ low risk (write OK on typical dev env) |
| D2 레시피 빌드 >900s | build 진행, Result.notes 에 기록 | D3 schema 가 처리 | ✅ (D3 가 커버) |
| D2 opt_level=5 로 mAP 하락 | rollback trigger | 해당 recipe 에서 field 제거 | ✅ (accept criteria) |
| D3 schema 확장이 기존 recipe/Result JSON break | D4.1 에서 조기 발견 | default=None 이라 backward compat 유지 | ✅ (테스트 포함) |

**Critical gap 없음** — 모두 detection / rollback 경로 있음.

---

## 7. Completion checklist

- [x] feat/wave15-tuning 브랜치 생성 (2026-04-23)
- [x] docs/plans/2026-04-23-wave15-audit-driven-tuning.md 작성 (이 문서)
- [x] D1.1 OV CACHE_DIR 구현
- [x] D1.2 ORT TRT EP 옵션 구현 (+ backward-compat fallback)
- [x] D3 `build_ceiling_s` schema + run_trt.py 수정
- [x] D4 테스트 4종 작성 (test_schema_wave15, test_recipe_bank_validate, test_ov_cache_dir, test_run_ort_trt_opts = 42 new tests)
- [x] pytest 전체 green (179 passed)
- [x] commits 4개 (plan/docs, D1 no-regret, D3 schema, D2-rollback)
- [x] D2 A/B 측정 — **ROLLBACK** for #09/#12/#42 (INT8 modelopt 근처 ceiling, opt_level=5 regression)
- [x] 결과 반영: 3개 YAML 원복, baseline JSON 복원, failed engines 보존
- [x] docs/improvements/2026-04-23-wave15-results.md 작성 (negative result documented)
- [x] recipes/README.md + CLAUDE.md 업데이트
- [ ] PR 생성 → merge (user-driven)

---

## 8. Review trail

- **Source**: plan-eng-review 2026-04-23 세션 (`~/.gstack/projects/yester31-OmniOptimizer/` review log 기록)
- **User decisions**:
  - Scope: `A+) Wave 15 = no-regret + Phase 2`
  - opt_level rollout: `A) 각 YAML 에 builder_optimization_level: 5 명시`
- **TODOs captured** (Wave 16 prerequisites): T1 audit_capabilities.py, T2 _node_patterns.py DRY — see `TODOS.md`
