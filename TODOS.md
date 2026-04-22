# TODOS

후속 작업 로그. Wave 단위 플랜에 못 들어간 작은/조건부 아이템들.

---

## Wave 16 prerequisites

### T1. `scripts/audit_capabilities.py` 작성

- **What**: YOLO26n ONNX 및 실행 환경의 capability 팩트를 한 번에 수집해 `results/_capabilities.json`으로 고정하는 재실행 가능 헬퍼. ~80 LOC.
- **체크 항목 (4개)**:
  1. YOLO26n ONNX의 MatMul / Attention (MHA) 노드 존재 여부 (`onnx.load` + `op_type == "MatMul"` 및 근처 softmax/reshape 패턴)
  2. 호스트 CPU flags (`amx_tile`, `avx512_bf16`) — `scripts/env_lock._collect_cpu_info()` 재사용
  3. `modelopt.onnx.quantization.int8` 모듈의 실제 default `op_types_to_quantize` allowlist
  4. ORT 1.22 `GraphOptimizationLevel.ORT_ENABLE_ALL`이 EXTENDED 대비 추가하는 pass 목록 (소스 grep)
- **Why**: Wave 16에서 `disable_mha_qdq` 스키마 필드 추가 여부, `#43_openvino_bf16` 레시피 등록 여부, modelopt 파라미터 확장 방향이 전부 이 팩트에 의존. inline grep은 3개월 뒤 재현 불가.
- **Pros**: 재실행 가능, CI에서 model/wheel 버전 변경 감지, dead-code 방지, AskUserQuestion 없이 Wave 16 플래닝이 팩트 기반으로 자동 진행.
- **Cons**: Wave 16 본 작업과 직접 결합되지 않으면 unit test 1개 짜리 separate PR이 됨.
- **Context**: 2026-04-23 plan-eng-review (`/plan-eng-review` 세션)에서 원안 Phase 0을 도구화한 것. 에이전트 리서치 결과 MHA 존재를 "있음"으로 단정했던 것을 팩트로 치환하기 위함.
- **Depends on / blocked by**: 없음. Wave 15 완료 여부와 무관하게 언제든 착수 가능. 단, Wave 16 플랜 작성 전에 완료되어야 그 플랜의 scope 결정에 반영됨.

---

### T2. `scripts/_node_patterns.py` 추출 (DRY)

- **What**: ONNX 노드명 prefix 패턴 확장 헬퍼를 공용 모듈로 분리. `expand_node_patterns(patterns: list[str], onnx_model: onnx.ModelProto, output_style: Literal["names", "regex"]) -> list[str]` 같은 API.
- **Why**: 현재 `/model.23/` prefix → 실제 노드명 리스트 확장 로직이 `scripts/run_cpu.py:382-398` (ORT quantize_static용 exact names) 한 곳에 있고, Wave 16 NNCF IgnoredScope 작업 시 같은 prefix를 regex (`/model\\.23/.*`) 로 변환하는 두 번째 호출 지점이 생김.
- **Pros**: 30~40 LOC 중복 제거. NNCF 패턴 변환 버그 생겨도 한 곳 수정. 유닛 테스트가 한 곳에 집중.
- **Cons**: Wave 16 NNCF 작업이 취소/연기되면 premature abstraction. 호출 지점 1개에서는 YAGNI.
- **Context**: 2026-04-23 plan-eng-review에서 code-quality CQ3로 발견. 현재는 Wave 11 Task 5에서 ORT-only로 만들어진 코드.
- **Depends on / blocked by**: Wave 16 NNCF preset 작업 확정 여부 (T1 결과에 따라 Wave 16 scope가 결정됨). **T1 선행**.

---

## Wave 17+ Deferred (hardware / ecosystem)

### T3. BF16 inference hardware path (i7-11375H `avx512_bf16` 여부에 따라)

- **What**: AVX-512_BF16 지원 CPU에서 `INFERENCE_PRECISION_HINT="bf16"` OpenVINO recipe (`#43_openvino_bf16.yaml`) 또는 ORT CPU BF16 활성화.
- **Why**: 현재 `#31 ort_cpu_bf16`은 `NotImplementedError` (`run_cpu.py:645-651`) — hardware gate 실패. OV는 별도 capability 테이블이 있어 독립 결정 필요.
- **Context**: plan-eng-review에서 Wave 15 scope 외 분리. T1 결과 의존.
- **Depends on**: T1.

### T4. `modelopt.onnx.quantize` autotune (`trtexec` 의존성)

- **What**: `modelopt.onnx.quantize(autotune="default", autotune_use_trtexec=True)` 경로로 Q/DQ 배치 스킴 자동 탐색. levipereira 레포에서 YOLO 기준 +3~10% fps 보고된 신규 기능.
- **Why**: #42 위에 추가 성능 압착 가능성. 단, 빌드 타임 1~2시간, trtexec dependency 추가.
- **Cons**: NVIDIA 공식 YOLO 벤치 없음, 서드파티 관찰에만 의존.
- **Context**: plan-eng-review에서 Wave 16 스카우트 카드로 분리.
- **Depends on**: T1 (modelopt API 확인).

### T5. ORT TRT EP CUDA graph 레시피 (`#04b`)

- **What**: bs=1 fixed shape + `trt_cuda_graph_enable=True` 전용 레시피.
- **Why**: #04 (211 fps) 대비 잠재 +10~15% 가능성. 단, native #40 (645 fps)에 비해서는 여전히 낮을 것으로 예상 → ORT EP **커버리지 demo** 의미.
- **Cons**: recipe bank 증식, 측정 부담.
- **Context**: plan-eng-review에서 "드롭 고려 대상"으로 명시.
- **Depends on**: Wave 15 D1.2 (`trt_builder_optimization_level=5` 적용) 선행.

---

## Review / decision trail

- 2026-04-23 **plan-eng-review** 세션에서 Wave 15 scope A+ + opt-in A 확정.
- Wave 15 = 2026-04-23, D1~D5 (~4시간 예상).
- Wave 16는 T1 완료 후 audit 결과 반영하여 별도 `/gsd-plan-phase` 사이클.
