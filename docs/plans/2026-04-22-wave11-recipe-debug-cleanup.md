# Wave 11: Recipe Bank Debug + Cleanup

**Goal:** 기존 30 recipe 중 **성능/정확도 문제 5개 GPU + 1개 CPU** 를 debug 해 recipe bank 신뢰도 복구. 부수적으로 `recommend.py` regression test 추가. 신규 recipe 추가는 **없음** (Wave 14 로 분리). Distribution (Dockerfile + CI) 은 본 Wave scope 밖, NOT in scope 섹션 참조.

**Background**: 2026-04-22 recipe bank audit (`docs/improvements/2026-04-22-recipe-bank-audit.md`) 에서 식별된 B1-B5 debug 항목.

**Architecture**: 변경 지점 6곳.
1. `scripts/run_ort.py` — #21 `ort_cuda_fp16` CUDAExecutionProvider 초기화 로직 확인 / provider ordering + fallback 감지 (B1).
2. `scripts/run_pytorch.py` — torch.compile `mode="reduce-overhead"` 또는 `"max-autotune"` 적용 + cache warm-up (B2).
3. `recipes/04_ort_trt_fp16.yaml` + `scripts/run_ort.py` — TRT cache path 명시, graph_optimization_level=ALL (B3).
4. `scripts/run_trt.py::_prepare_modelopt_onnx` — calibrator 간 TRT tactic 차이 조사 + 필요 시 `tactic_sources` 명시 (B4).
5. `scripts/run_cpu.py::_prepare_ort_cpu_int8_static` — QDQ op coverage inspect, Detect head 처리 확인 (B5).
6. `tests/test_recommend_ranking.py` **신규** — constraint filter + max_map_drop_pct regression guard.

**리스크**:
- B4 원인 불명확할 수 있음 (calibrator 선택이 TRT tactic 선택에 간접 영향) — Task 4 탐색 결과에 따라 "문서화만 하고 park" 분기 가능.
- B1 / B5 는 근본 원인이 ORT 버전 + HW 조합 문제면 archive 분기 가능. 각 Task 의 마지막 step 이 "fix or archive" 결정 명시.

## Task 0: Spike — 환경 확인 (선행)

**포함 조건**: Wave 7/8/12/13 에서 spike 조기 실패로 archive. 이번 Wave 는 debug 가 핵심이라 "환경 재현성" 확인만.

- [ ] 0.1 현재 Conda env 에서 #21, #20, #18 재실행 → fps 수치 재현 확인
- [ ] 0.2 `torch.cuda.is_available()` + `onnxruntime.get_device()` 반환값 snapshot
- [ ] 0.3 `nvidia-smi` + `torch.version.cuda` + `ort.get_available_providers()` 기록
- [ ] 0.4 **ORT provider options 실존 키 검증** (CLAUDE.md §1 convention 적용 — post-review meta finding): `TensorrtExecutionProvider` provider_options dict 의 실제 키명 확인 (`trt_fp16_enable`, `trt_engine_cache_enable`, `trt_engine_cache_path` 등). 버전별 명칭 차이 가능 — Task 3 진입 전 `ort.InferenceSession(..., providers=[("TensorrtExecutionProvider", {...}), ...])` smoke 로 unknown key 경고 없는지 확인.

**STOP 조건**: 실측 fps 가 report_qr.md 와 ±20% 범위 밖이면 측정 protocol 문제 — Wave 10 Phase 7 교훈대로 `scripts.measure.measure_latency` 강제.

**Task 독립성**: Task 1-5 는 서로 독립. 각 Task 완료 후 commit 분리. 한 Task 가 archive 결정으로 끝나도 다음 Task 계속 진행 (Wave 10 phase 스타일).

## Task 1: B1 — ort_cuda_fp16 fps 3.2 debug

**가설**: CUDAExecutionProvider 초기화 실패 → CPU fallback, 또는 cuDNN DLL path / cuBLAS incompat.

- [ ] 1.1 session 생성 후 `sess.get_providers()` 반환 ordering 확인 (CPUExecutionProvider 가 primary 인지)
- [ ] 1.2 `CUDAExecutionProvider` 옵션 `device_id=0`, `arena_extend_strategy="kNextPowerOfTwo"` 명시 후 재측정
- [ ] 1.3 onnx session 로드 후 `sess._model_meta` / provider logs 출력
- [ ] 1.4 `ort.set_default_logger_severity(0)` 로 verbose 로그 수집
- [ ] 1.5 fix + verify — fps 100 이상 복구 or 불가 원인 문서화 후 `recipes/_archived/` 이동 결정

## Task 2: B2 — torchcompile_fp16 baseline 역전 debug

**가설**: `torch.compile(model)` default mode 는 inductor graph 재컴파일 반복. `reduce-overhead` / `max-autotune` 미적용.

- [ ] 2.1 `scripts/run_pytorch.py::_prepare_torchcompile` 현재 compile 호출 설정 확인
- [ ] 2.2 `mode="reduce-overhead"` 로 변경 후 warmup **200** / measure 100 재측정 (warmup 200 근거: torch.compile inductor 초기 컴파일·autotune 이 40-60 iter 에 걸쳐 변동, 기존 default 100 으로는 측정 구간 진입 시점에 안정되지 않음 — post-review meta finding)
- [ ] 2.3 `mode="max-autotune"` 비교 (build time 수 배 증가 대가로 런타임 fps 이득)
- [ ] 2.4 ultralytics YOLO 가 사용자 지정 `forward_impl` 을 호출하는 방식 확인 — compile hook 가 detect head 까지 미치는지
- [ ] 2.5 fix + verify — fps >= pytorch_fp32 (46.5) + 30% 달성

## Task 3: B3 — ort_trt_fp16 fps 211 개선

**✓ Precondition resolved 2026-04-22**: `scripts/run_ort.py::_add_tensorrt_dll_dir()` 가 `tensorrt_libs/` 를 DLL search path 에 추가 (`os.add_dll_directory`). session 생성 시 `TensorrtExecutionProvider` primary 로드 확인. 상세 `docs/improvements/2026-04-22-wave11-task0-findings.md`.

**가설**: TRT execution provider cache 미활성 + ORT graph_optimization_level default.

- [x] 3.0 DLL path 수복 (`scripts/run_ort.py` 패치) — 2026-04-22 완료
- [ ] 3.1 recipe #04 에 `trt_engine_cache_enable=True`, `trt_engine_cache_path=results/_trt_cache/` 추가 (ORT TRT provider options — 키명 Task 0.4 검증 완료)
- [ ] 3.2 `SessionOptions.graph_optimization_level = ORT_ENABLE_ALL` 명시
- [ ] 3.3 `trt_fp16_enable=True` 확인 (키명 Task 0.4 검증 완료)
- [ ] 3.4 재측정 — fps 300+ 타겟 (native TRT fp16 fps 435 의 70%)

## Task 4: B4 — modelopt INT8 ptq vs entropy fps 격차 원인 조사

**가설**: calibrator 차이 → activation scale 범위 분포 차이 → TRT kernel tactic 선택 분기.

- [ ] 4.1 #13 `modelopt_int8_ptq` 와 #09 `modelopt_int8_entropy` 의 ONNX Q/DQ 노드 수 / scale 분포 비교 (`onnx.load` + inspect)
- [ ] 4.2 `trt-engine-explorer` 또는 `nvinfer.Runtime.get_layer_info` 로 engine 내 tactic 비교
- [ ] 4.3 root cause 문서화 → `docs/improvements/2026-04-23-modelopt-ptq-tactic-analysis.md`
- [ ] 4.4 분기: 원인이 calibrator 본질이면 **문서화만**, recipe 수정 범위 밖. 해결 가능하면 `tactic_sources` or `builder_optimization_level` 로 ptq 쪽 개선.

## Task 5: B5 — ort_cpu_int8_static mAP=0 복구

**가설**: QDQ op coverage 부재 (Detect head 에 Quantize/Dequantize 못 붙음) → inference 출력 전부 0.

- [ ] 5.1 #33 recipe 로 생성된 QDQ ONNX 직접 열어 Detect head 주변 op 확인
- [ ] 5.2 ORT static quant tool (`onnxruntime.quantization.quantize_static`) 호출 시 `nodes_to_exclude` 로 Detect head 제외 시도
- [ ] 5.3 or `op_types_to_quantize` 를 Conv/MatMul 로 한정해 재실행
- [ ] 5.4 **Archive-default 분기** (outside voice F3 반영): `op_types_to_quantize=[Conv,MatMul]` 경로는 `#32 ort_cpu_int8_dynamic` 과 실질 동등. fix 하려면 `nodes_to_exclude` 로 Detect head 제외 + mAP 90% 이상 달성이 유일 조건. 그 외 모두 `recipes/_archived/` 이동이 default 결정. brevitas 중복 archive 전례 따름.

## Task 6: recommend.py regression test

**Background**: Wave 10 reopen 과정에서 `recommend.py --max-map-drop-pct 5.0` CLI flag 누락 시 FastNAS recipe 가 ✘ 처리되는 버그 발견. 자동 테스트 부재.

- [ ] 6.1 `tests/test_recommend_ranking.py` 작성 — 인공 Result JSON 3-4개로:
    - baseline mAP 0.988, test recipe mAP 0.950 (drop 3.8%p)
    - `max_map_drop_pct=1.5` → ✘ / `=5.0` → ✔
    - fps sort 정순 / constraint 통과 필터 검증
    - `Result` 모든 Optional 필드 None 인 레코드 parse 성공 (schema backward compat)
- [ ] 6.2 `tests/test_recommend_exclude.py` — `--exclude` CLI flag 동작 검증
- [ ] 6.3 `tests/test_recipe_smoke.py` 신규 — B1-B5 대상 recipe 각각 "fps > 임계값" 1개 smoke assert (eng review 2026-04-22 §Section 3 결정):
    - #21 ort_cuda_fp16: fps > 100
    - #20 torchcompile_fp16: fps > 50
    - #18 ort_trt_fp16: fps > 300
    - #33 ort_cpu_int8_static: mAP > 0.5
    - #13 modelopt_int8_ptq: fps > 400 (B4 조사 결과와 무관하게 회귀 방지 — outside voice F4 반영)
- [ ] 6.4 pytest 112+ passed (현재 96 + recommend_ranking + exclude + recipe_smoke)

## 완료 기준

- B1-B5 항목 각각 **fix + verify** or **archive + 원인 문서화** 둘 중 하나로 결론
    - B1 #21 ort_cuda_fp16: fps >= 100 or archive
    - B2 #20 torchcompile_fp16: fps >= pytorch_fp32 × 1.30 or archive
    - B3 #18 ort_trt_fp16: fps >= 300 or archive
    - B4 #13 modelopt_int8_ptq: docs/improvements 원인 분석 ship (fix 여부는 분기 결정)
    - B5 #33 ort_cpu_int8_static: mAP >= 0.9 (nodes_to_exclude 경로) or archive (default)
- `tests/test_recommend_ranking.py` + `test_recommend_exclude.py` + `test_recipe_smoke.py` 추가, pytest 112+ passed
- **CLAUDE.md `Current scope` recipe count 갱신** (archive 되는 recipe 만큼 감소 반영 — post-review meta finding. Wave 14 Task 5.3 에 이 결과가 입력됨)
- `report_qr.md` + `report_cpu_qr.md` 재생성 후 archived recipe 가 ranking 에서 제거 확인

## NOT in scope (후속 Wave)

- **Dockerfile + GH Actions CI** — 사용자 결정 (2026-04-22 plan-eng-review Step 0)으로 제외. 향후 필요 시 별도 Wave 로 재개.
- 신규 recipe 추가 — Wave 14 TRT tuning (A1 A2 A5), Wave 15 Multi-resolution (A4), Wave 16 Memory format (A3)
- OpenVINO CPU int8 NNCF 추가 calibrator — 현재 1 recipe (#35) 로 충분
- FP8 / NVFP4 / INT4 — HW / feasibility 제약
- Distillation — v2 scope

## 참고

- Audit report — `docs/improvements/2026-04-22-recipe-bank-audit.md`
- Wave 10 측정 protocol 교훈 — `CLAUDE.md` "Plan-writing conventions" §4
- TRT BuilderConfig API — TensorRT 10.x Python docs
- ORT TRT EP options — <https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html>

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | **CLEAR (PLAN)** | 5 issues, 0 critical gaps |
| Outside Voice | Claude subagent | Independent 2nd opinion | 1 | **CHALLENGED** | 8 findings: 6 reflected in plan, 1 strategic kept-as-is, 1 sequencing strengthened |

- **OUTSIDE VOICE**: Claude subagent (Codex unavailable). F6 API name error fixed (`is_bf16_supported()`). F1 Plan B fusion smoke added. F3 archive-default. F4 #13 smoke. F5 완료 기준 axis 별 분리. F7 build 600s 상한. F2 sequencing 명시. F8 strategic priority — user opted to keep original sequence.
- **UNRESOLVED**: 0
- **VERDICT**: ENG CLEARED — ready to implement (Wave 11 → Wave 14 순차)
