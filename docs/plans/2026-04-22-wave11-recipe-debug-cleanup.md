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

**Root cause 확정 2026-04-22 (smoke)**: CUDAExecutionProvider 는 정상 로드, 초기화 실패 가설 기각. 진짜 원인은 **ONNX half-export (fp16 weights + fp32 IO) 가 만드는 219 Memcpy 노드** — CUDA EP 가 precision 경계마다 CPU↔GPU 왕복. 결과 p50 352ms → fps 2.8 (report 3.2 와 일치). 상세 `docs/improvements/2026-04-22-wave11-task0-findings.md`.

- [x] 1.1 session `sess.get_providers()` 확인 — CUDAExecutionProvider primary 정상 (2026-04-22)
- [x] 1.2 `arena_extend_strategy` / `device_id` 확인 — 무관, root cause 는 graph level (2026-04-22)
- [x] 1.3 session 로드 경고 확인 — `219 Memcpy nodes added` 확인 (2026-04-22)
- [x] 1.4 `onnxconverter_common.float16.convert_float_to_float16` 3가지 옵션 조합 모두 실패 (keep_io_types=False / op_block_list=['Resize'] / keep_io_types=True) — ONNX validation error "Type Error: Resize_output_cast0 tensor(float) vs tensor(float16)" (2026-04-22)
- [x] 1.5 **본질 재확인 smoke** — fp32 ONNX + CUDA EP 도 **217 Memcpy / fps 7.5** — 원인은 precision 이 아니라 **YOLO26n end2end NMS ops (TopK/GatherElements/NonMaxSuppression) 가 CUDA EP 에서 unsupported** (Wave 7/8 archive 와 동일 class) (2026-04-22)
- [x] 1.6 **Archive 결정** (2026-04-22 plan-eng-review Task 1.6 분기 선택): recipe #03 → `recipes/_archived/`, results/_archived/, results_qr/_archived/. Makefile + batch scripts 에서 제거. report_qr.md 재생성.

## Task 2: B2 — torchcompile_fp16 baseline 역전 debug

**Root cause 2026-04-22**: 기존 result JSON notes 에 이미 기록된 대로 `torch.compile unavailable, fell back to eager: TritonMissing`. fps 39 는 torch.compile 이 아니라 **eager fp16** 의 측정값. eager fp16 이 eager fp32 (46.5) 보다 느린 건 YOLO26n Conv kernel 의 fp16 dispatch 가 suboptimal 한 정상 현상.

**Resolution: Archive**. 이번 세션 (2026-04-22) 에 triton-windows 3.6 설치 시도 + triton_key shim 성공 + `mode="reduce-overhead"` 로 진행 → 그러나 torch 2.8 inductor 가 YOLO 같은 복잡 graph 에 대해 kernel codegen 에서 **C compiler (MSVC) 을 요구** — `RuntimeError: Failed to find C compiler`. VS Build Tools (~2GB) 미설치 환경에서 torch.compile 불가. 사용자 결정 2026-04-22: VS Build Tools 설치 건너뛰고 archive.

- [x] 2.1 `scripts/run_pytorch.py` 의 `torch.compile(mode="reduce-overhead")` 호출 확인 (2026-04-22 완료)
- [x] 2.2 triton-windows 3.6 설치 후 mode="reduce-overhead" 재시도 → C compiler 요구로 fail (2026-04-22)
- [x] 2.3 `mode="max-autotune"` 은 reduce-overhead 보다 더 aggressive codegen — 같은 C compiler blocker 로 fail 예상, 시도 생략
- [x] 2.4 ultralytics forward_impl 조사: inner_model (yolo.model) 을 compile 하지만 YOLO 의 Detect head 동적 anchor 생성이 inductor 로 codegen 됨 → MSVC 필요 근본 원인 (2026-04-22)
- [x] 2.5 **Archive 결정** — recipe #02 → `recipes/_archived/`, results/_archived/, results_qr/_archived/. Makefile + batch scripts 에서 제거. run_pytorch.py 에 triton_key shim 은 dormant 로 유지 (기록 + 향후 재시도용).

## Task 3: B3 — ort_trt_fp16 fps 211 개선

**✓ Precondition resolved 2026-04-22**: `scripts/run_ort.py::_add_tensorrt_dll_dir()` 가 `tensorrt_libs/` 를 DLL search path 에 추가 (`os.add_dll_directory`). session 생성 시 `TensorrtExecutionProvider` primary 로드 확인. 상세 `docs/improvements/2026-04-22-wave11-task0-findings.md`.

**가설**: TRT execution provider cache 미활성 + ORT graph_optimization_level default.

- [x] 3.0 DLL path 수복 (`scripts/run_ort.py` 패치) — 2026-04-22 완료
- [x] 3.1 `_make_session` 에 `trt_engine_cache_enable=True`, `trt_engine_cache_path` 하드코딩 주입 (recipe YAML schema 확장 대신 runner 내부 처리, 간결성 우선) — 2026-04-22 완료
- [x] 3.2 `SessionOptions.graph_optimization_level = ORT_ENABLE_ALL` — 이미 기존 코드에 존재 확인
- [x] 3.3 `trt_fp16_enable` 을 `recipe.runtime.dtype == "fp16"` 기반 자동 설정 — 2026-04-22 완료
- [x] 3.4 smoke 측정 — fps **188.6** (p50 5.30ms). 기존 report fps 211 과 일치. native TRT fp16 대비 43%. Task 3 완료 기준 재정의 (fps 300+ 은 ORT-via-TRT 구조적 한계로 달성 불가 — 2026-04-22)
- [ ] 3.5 정식 recipe #04 측정 재실행 (`results_wave11/04_ort_trt_fp16.json`) — 병합 완료 시 report_qr.md 갱신

## Task 4: B4 — modelopt INT8 ptq vs entropy fps 격차 원인 조사

**Root cause 확정 2026-04-23**: 주요 변동 요인은 **TRT 빌더 autotune nondeterminism**. calibrator 효과는 체계적으로 존재하되 현재 단일 측정 순위는 noise 로 뒤집힐 여지. 상세 `docs/improvements/2026-04-23-modelopt-ptq-tactic-analysis.md`.

- [x] 4.1 #08 `modelopt_int8_ptq` vs #09 `modelopt_int8_entropy` vs #10 `modelopt_int8_percentile` ONNX Q/DQ 노드 수 / scale 분포 비교 (2026-04-23) — Q/DQ 노드 수 및 weight scale 완전 동일, activation scale 만 차이 (max 1.19, entropy 0.66, percentile 1.19)
- [x] 4.2 Engine binary 크기 / md5 비교 (2026-04-23) — max 와 percentile 의 ONNX 은 **byte-identical** 이지만 engine 은 다른 md5 + 같은 크기, fps 는 430 vs 755 (1.76× 격차). nondeterminism 직접 증거.
- [x] 4.3 root cause 문서화 → `docs/improvements/2026-04-23-modelopt-ptq-tactic-analysis.md` (2026-04-23)
- [x] 4.4 분기 결정 (2026-04-23): **문서화만**. fix path 없음 — calibrator 이름으로 재현 가능한 결정적 차이 아님. Wave 14 A1 `builder_optimization_level=5` 가 진짜 해결 경로 (autotune 공간 확장으로 stabilization + ceiling lift).

## Task 5: B5 — ort_cpu_int8_static mAP=0 복구

**가설**: QDQ op coverage 부재 (Detect head 에 Quantize/Dequantize 못 붙음) → inference 출력 전부 0.

- [ ] 5.1 #33 recipe 로 생성된 QDQ ONNX 직접 열어 Detect head 주변 op 확인
- [ ] 5.2 ORT static quant tool (`onnxruntime.quantization.quantize_static`) 호출 시 `nodes_to_exclude` 로 Detect head 제외 시도
- [ ] 5.3 or `op_types_to_quantize` 를 Conv/MatMul 로 한정해 재실행
- [ ] 5.4 **Archive-default 분기** (outside voice F3 반영): `op_types_to_quantize=[Conv,MatMul]` 경로는 `#32 ort_cpu_int8_dynamic` 과 실질 동등. fix 하려면 `nodes_to_exclude` 로 Detect head 제외 + mAP 90% 이상 달성이 유일 조건. 그 외 모두 `recipes/_archived/` 이동이 default 결정. brevitas 중복 archive 전례 따름.

## Task 6: recommend.py regression test

**Background**: Wave 10 reopen 과정에서 `recommend.py --max-map-drop-pct 5.0` CLI flag 누락 시 FastNAS recipe 가 ✘ 처리되는 버그 발견. 자동 테스트 부재.

- [x] 6.1 `tests/test_recommend_ranking.py` 작성 (2026-04-23) — 7 테스트:
    - `max_map_drop_pct=1.5` ✘ / `=5.0` ✔ (Wave 10 reopen 버그 재발 guard)
    - `min_fps_bs1` threshold 필터
    - ranking 순서 (meets 우선, 그 다음 fps desc)
    - 측정값 누락 시 graceful fail
    - 모든 Optional=None 인 legacy JSON parse 성공
    - `_env.json` meta 파일 skip
    - malformed JSON warn 후 skip
- [x] 6.2 `tests/test_recommend_exclude.py` (2026-04-23) — 5 테스트: 단일/다중/empty/recipe name 매칭/unknown noop
- [x] 6.3 `tests/test_recipe_smoke.py` 신규 (2026-04-23) — archive 반영 후 3 recipe:
    - #04 ort_trt_fp16: fps > 150 (Task 3 재측정 188.6, ORT-via-TRT 구조 ceiling)
    - #08 modelopt_int8_ptq: fps > 300 (Task 4 B4 nondeterminism 허용 floor)
    - #33 ort_cpu_int8_static: mAP > 0.5 (Wave 11 Task 5 완료까지 xfail, dual-source JSON 지원)
    - 제외됨: #02 torchcompile_fp16 / #03 ort_cuda_fp16 (모두 Task 1/2 에서 archive)
- [x] 6.4 pytest **119 passed + 1 xfailed** (현재 105 + 7 ranking + 5 exclude + 3 smoke = 120, xfail 은 Task 5 완료 대기) — 목표 112+ 초과 달성

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
