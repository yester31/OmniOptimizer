# TODOS

후속 작업 로그. Wave 단위 플랜에 못 들어간 작은/조건부 아이템들.

---

## Active (Wave 17+ candidates)

### T3. BF16 inference hardware path — Wave 17+ (hardware blocked on current host)

- **What**: AVX-512_BF16 지원 CPU에서 `INFERENCE_PRECISION_HINT="bf16"` OpenVINO recipe (`#43_openvino_bf16.yaml`) 또는 ORT CPU BF16 활성화.
- **Why**: 현재 `#31 ort_cpu_bf16`은 `NotImplementedError` (`run_cpu.py:645-651`) — hardware gate 실패. OV는 별도 capability 테이블이 있어 독립 결정 필요.
- **Current status**: T1 audit (2026-04-23, `results/_capabilities.json`) confirmed i7-11375H lacks `avx512_bf16` AND `amx_tile`. **Hardware-blocked on this host.** Candidate for re-evaluation on hardware upgrade (Sapphire Rapids / Tiger Lake+ / Zen 4).
- **Context**: plan-eng-review 에서 Wave 15 scope 외 분리. T1 결과 반영: Wave 17+ 로 연기.
- **Depends on**: 하드웨어 업그레이드 (AVX-512_BF16 or AMX 지원 CPU).

### T4. `modelopt.onnx.quantize` autotune (`trtexec` 의존성)

- **What**: `modelopt.onnx.quantize(autotune="default", autotune_use_trtexec=True)` 경로로 Q/DQ 배치 스킴 자동 탐색. levipereira 레포에서 YOLO 기준 +3~10% fps 보고된 신규 기능.
- **Why**: #42 위에 추가 성능 압착 가능성. 단, 빌드 타임 1~2시간, trtexec dependency 추가.
- **Cons**: NVIDIA 공식 YOLO 벤치 없음, 서드파티 관찰에만 의존.
- **Context**: plan-eng-review 에서 Wave 16 스카우트 카드로 분리. Wave 16 scope에서 빠져 Wave 17+로 이월.
- **Depends on**: modelopt API 확인 (T1 audit이 default allowlist 만 확인, autotune API 별도).

### T5. ORT TRT EP CUDA graph 레시피 (`#04b`)

- **What**: bs=1 fixed shape + `trt_cuda_graph_enable=True` 전용 레시피.
- **Why**: #04 (211 fps) 대비 잠재 +10~15% 가능성. 단, native #40 (645 fps)에 비해서는 여전히 낮을 것으로 예상 → ORT EP **커버리지 demo** 의미.
- **Cons**: recipe bank 증식, 측정 부담.
- **Context**: plan-eng-review 에서 "드롭 고려 대상"으로 명시. 실익 대비 시험 비용 검토 필요.
- **Depends on**: Wave 15 D1.2 (`trt_builder_optimization_level=5` 적용) — **완료** (PR #6).

---

## Canceled

### T2. `scripts/_node_patterns.py` 추출 (DRY) — CANCELED 2026-04-23

- **Reason**: Wave 16 NNCF IgnoredScope 작업이 T1 audit 결과로 scope 아웃됨 (`docs/plans/2026-04-23-wave16-plan.md`). 두 번째 호출 지점이 생기지 않아 YAGNI.
- **Original spec**: ONNX 노드명 prefix 패턴 확장 헬퍼를 공용 모듈로 분리. `scripts/run_cpu.py:382-398` 중복 방지 목적이었음.
- **Re-gate**: Wave 17+ 에서 유사 IgnoredScope 요구가 다시 생기면 재오픈.

---

## Completed

### T1. `scripts/audit_capabilities.py` — DONE 2026-04-23 (PR #7)

- Shipped: `scripts/audit_capabilities.py` (131 LOC) + `results/_capabilities.json` snapshot + 4 offline tests.
- Key findings: YOLO26n has **no MHA pattern** (4 MatMul, 0 feed Softmax directly). i7-11375H lacks AVX-512_BF16 / AMX. modelopt default allowlist = 31 ops incl. MatMul/Softmax. ORT ALL vs EXTENDED delta = NCHWc layout + MLAS fusion passes.
- Impact: killed Wave 16's `disable_mha_qdq` schema field (dead weight); deferred `#43_openvino_bf16` to Wave 17+; closed D2 scout as null result.

### T6. `Result.build_ceiling_breached` round-trip — DONE 2026-04-23 (PR #11)

- Shipped: `Result.build_ceiling_breached: Optional[bool]` + `run_trt.py::run()` tracker (sticky-True semantics) + `recommend.py` breach section + 7 offline tests.
- Wave 15 D3 `build_ceiling_s` signal now round-trips through Result JSON instead of dying in stderr.

### T7. ONNX cache key `nodes_to_exclude` hash — DONE 2026-04-23 (PR #8)

- Shipped: `_modelopt_onnx_tag` helper (pure, testable) with `_ex<sha8>` suffix when `nodes_to_exclude` is non-empty. sorted + `|`-joined key → order-insensitive. Backward-compat: empty exclusions yield legacy path.
- Fix: recipes #09 (no excludes) and #12 (4 Convs excluded) no longer share `best_qr_640_modelopt_entropy_bs1.onnx` — silent cache poisoning eliminated.
- 4 offline tests. Full suite 134 passed.

---

## Review / decision trail

- 2026-04-23 **plan-eng-review** 세션에서 Wave 15 scope A+ + opt-in A 확정.
- Wave 15 = 2026-04-23, D1~D5 shipped (D1/D3 positive, D2 opt_level=5 ROLLED BACK after measurement).
- Wave 16 plan (`docs/plans/2026-04-23-wave16-plan.md`) written post-T1: 4 → 1 workstream after facts landed. D1 shipped (PR #11), D2 closed as null result (PR #10).
- Wave 17+ scope will re-open after hardware upgrade (T3) or explicit decision to scout T4/T5.
