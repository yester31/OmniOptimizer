# Wave 14: TRT Builder Optimization + BF16 + Asymmetric INT8

**Status**: **SHIPPED 2026-04-22**. 3 축 전부 성공:
- A1 (opt_level=5): fps **645.2** (vs #05 baseline 435 = **+48%**, 목표 +5% 훨씬 초과)
- A2 (bf16): fps 372.7, mAP **0%p drop** vs FP16 (완료 기준 ≤0.05%p PASS)
- A5 (asymmetric): fps **770.5** (**NEW TOP**, +0.9% vs entropy 763.9), mAP 0%p drop

**Goal:** 현재 Top recipe (`modelopt_int8_entropy` fps 763.9) 대비 **+5-15% fps 향상** + mAP 유지. 3 신규 recipe 로 TRT builder tuning (opt_level=5) + BF16 (Ampere sm_80+) + asymmetric INT8 실험.

**Precondition**: Wave 11 (recipe debug cleanup) 완료 후 착수. **Wave 11 B4 결과가 본 Wave Task 1 가설 검증 입력** (outside voice F2 반영) — B4 가 "calibrator → tactic 분기가 원인" 으로 결론나면 opt_level=5 로 ptq/entropy 수렴 여부 확인이 Task 1 성공 기준에 포함. B4 가 "calibrator 무관" 결론이면 Task 1 은 independent delta 측정으로 수행. **결과 2026-04-22**: B4 결론이 "TRT nondeterminism" 이었고, opt_level=5 (#40) 가 실제로 fps 435→645 로 **대폭 안정화**. Wave 11 B4 가설 (opt_level 이 ceiling lift + stabilization) 이 입증됨.

**Background**: 2026-04-22 recipe bank audit 에서 high-ROI 3 candidate (A1 + A2 + A5). 전부 보유 HW (RTX 3060 Laptop sm_86) 에서 실측 가능.

## Architecture

변경 지점 5곳.

1. `scripts/_schemas.py::RuntimeSpec` — `builder_optimization_level: Optional[int] = Field(default=None, ge=0, le=5)` 신규 필드. 기본 None → TRT default (=3).
2. `scripts/run_trt.py::_build_engine` — `builder_optimization_level` recipe 값 전달 시 `config.builder_optimization_level = N` 설정. build 시간 기록 Result JSON 에 `build_time_s` 추가.
3. `scripts/_schemas.py::RuntimeSpec.dtype` Literal 에 `"bf16"` 이 이미 있는지 확인 (Wave 6 추가됨). GPU 경로에서 BF16 처리 분기 `scripts/run_trt.py::_build_engine` 에 `elif dtype == "bf16": config.set_flag(nvinfer.BuilderFlag.BF16)` 추가.
4. `scripts/run_trt.py::_prepare_modelopt_onnx` — `calibration_method` 에 `"entropy_asymmetric"` 값 시 `modelopt.onnx.quantization.quantize` 호출에 `calibration_method="entropy"` + `symmetric=False` (또는 equivalent) 전달.
5. 3 신규 recipe + Makefile targets.

## 리스크

- **opt_level=5 build time**: 소폭 GPU (YOLO26n) 라도 base build 30s → 2-3분 예상. CI 에서 돌리면 안 됨 — 수동 실행.
- **BF16 on sm_86**: Ampere BF16 지원 확인됨 but `SPARSE_WEIGHTS + BF16` 조합은 TRT 10.x 에서 SPARSE INT8 대비 미검증. BF16 은 non-sparse variant 만 작성.
- **Asymmetric INT8 in modelopt**: `modelopt.onnx.quantization.quantize` 는 기본 symmetric. asymmetric 지원이 API 레벨에 있는지 실제 `__init__.py` + source 에서 확인 필요. 없으면 ORT quantization (asymmetric 지원) 로 우회 — recipe source 를 `ort_quant` 로 바꾸고 asymmetric 옵션 설정.

## Task 0: Spike — API 확인 (선행)

CLAUDE.md §1 plan-writing convention: "External library APIs must be grounded in the repo's actual `__init__.py` / source code."

- [ ] 0.1 TRT 10.x `nvinfer.BuilderConfig.builder_optimization_level` 속성 존재 확인 (실제 Python import + dir 확인, 문서 아님)
- [ ] 0.2 TRT `BuilderFlag.BF16` 존재 확인
- [ ] 0.3 `modelopt.onnx.quantization.quantize` signature 에서 asymmetric / symmetric flag 존재 확인 — 없으면 Plan B (ORT quantization) 로 전환
- [ ] 0.4 sm_86 (compute 8.6) 에서 BF16 FLOPS 지원 확인 — 실제 API `torch.cuda.is_bf16_supported()` 호출 (outside voice F6: `get_device_properties().bf16_supported` 속성은 존재하지 않음 — CLAUDE.md §1 convention 재위반 방지)
- [ ] 0.5 **Plan B 분기 시 사전 smoke** (outside voice F1 반영): Task 0.3 이 Plan B 로 결론나면 Task 1 진입 전에 "ort_quant asymmetric (zero_point ≠ 0) ONNX 를 TRT EP 로 실행 시 fusion 유지 + fps drop < 10%" smoke 통과 확인. 미통과 시 A5 전체 archive.

**STOP 조건**: Task 0 의 4개 항목 중 하나라도 불가 → 해당 axis Plan B 또는 archive. 전부 통과해야 Task 1 진행.

## Task 1: A1 — `trt_fp16_opt5` recipe

**Baseline**: #05 `trt_fp16` (fps 435.1).

- [ ] 1.1 `recipes/40_trt_fp16_opt5.yaml` — copy from #05, add `runtime.builder_optimization_level: 5`
- [ ] 1.2 `scripts/_schemas.py::RuntimeSpec` 에 `builder_optimization_level` 필드 추가 + pydantic 검증 (`ge=0, le=5`)
- [ ] 1.3 `scripts/run_trt.py::_build_engine` 에 해당 값 TRT BuilderConfig 로 전달 + build time 측정 (`time.perf_counter()`). **Build time 상한 600s** (outside voice F7 반영) — 초과 시 `notes` 필드에 "build timeout, opt_level=5 not viable on this HW" 기록 후 meets_constraints=False
- [ ] 1.4 `scripts/_schemas.py::Result` 에 `build_time_s: Optional[float] = None` 추가 (backward compat, historical JSON None)
- [ ] 1.5 make recipe-40 실행, Result JSON 수집
- [ ] 1.6 verify — fps >= 450 (trt_fp16 435 + 5%)

## Task 2: A2 — `trt_bf16` recipe

**Baseline**: #05 `trt_fp16` (fps 435.1) + #1 `modelopt_int8_entropy` (fps 763.9, 참고).

- [ ] 2.1 `recipes/41_trt_bf16.yaml` — copy from #05, `runtime.dtype: bf16`
- [ ] 2.2 `scripts/run_trt.py::_build_engine` 에 `dtype == "bf16"` 분기 추가 (`config.set_flag(BuilderFlag.BF16)`)
- [ ] 2.3 `tests/test_run_trt_bf16_gate.py` **신규** — sm_80 미만 compute capability 에서 bf16 recipe 가 early skip or NotImplementedError (post-review meta finding: test 파일 역할 분리 — `test_schema_wave6.py` 는 bf16 Literal 유지, `test_schema_wave14.py` (Task 4.5) 는 builder_optimization_level 전용, 본 파일은 HW gate 전용)
- [ ] 2.4 make recipe-41 실행
- [ ] 2.5 verify — fps 400-450 (FP16 과 유사), mAP drop < 0.1%p vs baseline
- [ ] 2.6 `scripts/run_trt.py::_build_engine` 에 **BF16 + sparsity 조합 금지 guard** 추가 (eng review 2026-04-22 §Section 1 결정) — `dtype == "bf16" and sparsity is not None` 시 early `NotImplementedError("BF16 + SPARSE_WEIGHTS untested on sm_86")`

## Task 3: A5 — `modelopt_int8_entropy_asymmetric`

**Baseline**: #09 `modelopt_int8_entropy` (fps 763.9, mAP +0.07%p).

- [ ] 3.1 Task 0.3 결과 분기 (eng review 2026-04-22 §Section 2 결정 — Plan A/B 각각 고유 파일명):
    - **Plan A (asymmetric 지원)**: `recipes/42_modelopt_int8_asymmetric.yaml`, modelopt.onnx.quantize 호출에 asymmetric flag
    - **Plan B (지원 안 함)**: `recipes/42_ort_quant_int8_asymmetric.yaml` — source 를 `ort_quant` + `calibrator: entropy` + `activation_symmetric: False`
- [ ] 3.2 ONNX Q/DQ 노드 수 / scale zero_point 분포 확인 (asymmetric 은 zero_point ≠ 0)
- [ ] 3.3 make recipe-42 실행
- [ ] 3.4 verify — mAP drop <= 0.05%p (symmetric 대비 개선), fps 700+ (symmetric 5% 이내)

## Task 4: Recipe bank regeneration

- [ ] 4.1 Makefile `.PHONY` 에 recipe-40/41/42 추가 + `all` target 에 포함
- [ ] 4.2 `scripts/run_qr_batch.sh` ORDER array 에 3 entry 추가
- [ ] 4.3 `make all` + `make report` → `report_qr.md` 재생성
- [ ] 4.4 신규 3 recipe rank 확인 — 최소 하나가 Top 5 진입
- [ ] 4.5 `tests/test_schema_wave14.py` 신규 (eng review 2026-04-22 §Section 3 결정) — `RuntimeSpec.builder_optimization_level` Optional/None/0/5 valid + `=6` ValidationError + `Result.build_time_s` Optional parse

## Task 5: 문서

- [ ] 5.1 `docs/architecture.md` recipe table 업데이트 (30 → 33 active)
- [ ] 5.2 `recipes/README.md` numbering convention 업데이트 — range `40-49` = TRT tuning
- [ ] 5.3 `CLAUDE.md` current scope recipe count 갱신 — **Wave 11 완료 시점의 active recipe 수 + 본 Wave 에서 ship 된 신규 수** (post-review meta finding: Wave 11 에서 archive 가능한 recipe 존재 — hard-coded "30→33" 아닌 dynamic). FP16 / INT8 외에 BF16 추가.
- [ ] 5.4 본 plan 파일 에 SHIPPED stamp 추가 (Wave 6/10 스타일)

## 완료 기준 (outside voice F5 반영 — axis 별 성공 기준 분리)

- **3 신규 recipe 중 2 개 이상 ship** (Result JSON 생성, report_qr.md rank 반영). 나머지 1 개는 "원인 문서화 + archive" 로 결론나면 success.
- **Axis별 성공 판정**:
  - A1 (opt_level=5): fps >= 457.9 (trt_fp16 435 + 5%) — fps 향상 축
  - A2 (bf16): mAP drop <= 0.05%p vs #05 trt_fp16 — mAP 품질 축 (fps 증가는 부차)
  - A5 (asymmetric): mAP drop <= 0.05%p vs #09 modelopt_int8_entropy — mAP 품질 축
- **전체 Top fps 갱신** (현재 763.9 초과) 은 stretch goal, 필수 아님 (+5% 도박 리스크 인정)
- pytest 112+ passed 유지

## NOT in scope (후속 Wave)

- A4 Multi-resolution (imgsz 320/512) — Wave 15
- A3 channels_last PyTorch path — Wave 16
- A6 bs=16 / bs=32 throughput — 별도 결정 (bs=8 까지 현재)
- FP8 / NVFP4 — HW 제약

## 참고

- Audit report — `docs/improvements/2026-04-22-recipe-bank-audit.md`
- TRT BuilderConfig.builder_optimization_level — TensorRT 10.x Python API
- TRT BuilderFlag.BF16 — 동
- modelopt.onnx.quantization.quantize — `modelopt.onnx.quantization.__init__` 실제 source
- Wave 10 `modelopt.onnx.quantize` 경로 precedent — `docs/plans/_shipped/2026-04-21-wave10-modelopt-fastnas-pruning.md` Phase 8

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | **CLEAR (PLAN)** | 5 issues, 0 critical gaps |
| Outside Voice | Claude subagent | Independent 2nd opinion | 1 | **CHALLENGED** | 8 findings: 6 reflected in plan, 1 strategic kept-as-is, 1 sequencing strengthened |

- **OUTSIDE VOICE**: Claude subagent (Codex unavailable). 주요 수정: F6 (`torch.cuda.is_bf16_supported()`), F1 (Plan B TRT fusion smoke), F5 (axis 별 완료 기준), F7 (build 600s 상한), F2 (Wave 11 B4 → Wave 14 Task 1 입력 명시).
- **UNRESOLVED**: 0
- **VERDICT**: ENG CLEARED — ready to implement (Wave 11 완료 후)
