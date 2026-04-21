# Wave 6: CPU 추론 백엔드 (ORT CPU EP + OpenVINO NNCF)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** OmniOptimizer를 NVIDIA GPU 단일 타겟에서 **x86_64 Intel CPU까지 확장**한다. 기존 `technique.source` 디스패치 구조에 `ort_cpu`와 `openvino` 2종을 편입해, 동일 `Recipe` 포맷으로 CPU 추론 권고까지 한 번에 낼 수 있게 만든다. 6개 레시피(#30–#35)를 한 장씩 뽑아 FP32/BF16/INT8(dynamic/static VNNI)/OpenVINO INT8 NNCF 공간을 덮는다.

**Architecture:** 변경 지점은 5곳으로 국한된다. (1) `scripts/_schemas.py::TechniqueSpec.source` Literal 확장 + `HardwareSpec` CPU 필드 추가. (2) `scripts/run_cpu.py` **신규** — TRT 의존 없이 ORT CPU EP + OpenVINO 직접 호출. (3) `scripts/env_lock.py` CPU 잠금(`cpupower`/`powercfg`) + 스레드 수 고정. (4) `recipes/`에 신규 YAML 6장 (#30–#35). (5) `Makefile`·배치 스크립트·`pyproject.toml` 확장. `run_trt.py`와 `run_ort.py`는 건드리지 않는다 — CPU는 TRT explicit QDQ 경로와 서로 다른 실행 그래프라 **별도 runner가 올바른 분리**다.

**Tech Stack:** `onnxruntime>=1.17` (이미 설치), `openvino>=2024.3` (신규), `nncf>=2.9,<3.0` (신규, OpenVINO PTQ), `numpy`, `ultralytics`, `pydantic`, `pyyaml`. AMD/ARM/Apple는 **Wave 6 범위 밖** — Task 10 extension gate 참고.

---

## Scope Boundaries

### In-scope (Wave 6)
- **하드웨어**: x86_64 Intel CPU만 (AVX2 최소, VNNI 권장, AMX는 optional recipe)
- **모델**: 기존 YOLO26n (다른 모델은 v2 work)
- **정밀도**: FP32 / BF16 / INT8(dynamic, static VNNI) / OpenVINO INT8 NNCF
- **OS**: Linux + Windows. macOS는 best-effort (Apple Silicon은 parked)

### Out-of-scope (future waves)
- ARM NEON / Apple CoreML EP → Wave 7 candidate
- AMD Zen4+ 전용 튜닝 (공통 AVX-512 경로는 자동으로 작동해야 함)
- INT4 weight-only (LLM용, Vision에 아직 연구 단계)
- CPU pruning / SAT (Wave 5 GPU 파이프라인 재사용 가능하나 별도 이슈)

### Assumptions (착수 전 확인 필요)
- `best_qr.pt` + `qr_barcode.yaml` 데이터셋은 CPU 경로에서도 그대로 평가 대상 (OMNI_WEIGHTS_OVERRIDE 패턴 재사용)
- 측정 대상은 **단일 이미지 latency (bs=1) 우선**. 서버 처리량(bs=8) 병행 측정
- CPU는 GPU와 달리 **thermal throttling 실측 리스크** 크므로 warmup·cooldown 늘림 (measure.py 패치 필요할 수 있음 — Task 6에서 판단)

---

## Recipe Map (#30–#35)

| # | name | source | dtype | calibrator | 타겟 하드웨어 |
|---|---|---|---|---|---|
| 30 | `ort_cpu_fp32` | `ort_cpu` | fp32 | — | AVX2+ (all) |
| 31 | `ort_cpu_bf16` | `ort_cpu` | bf16 | — | SPR+ / Zen4+ (AMX 또는 AVX-512 BF16) |
| 32 | `ort_cpu_int8_dynamic` | `ort_cpu` | int8 | — (runtime) | AVX-512 VNNI 권장 |
| 33 | `ort_cpu_int8_static` | `ort_cpu` | int8 | entropy | AVX-512 VNNI 권장 |
| 34 | `openvino_fp32` | `openvino` | fp32 | — | AVX2+ (all) |
| 35 | `openvino_int8_nncf` | `openvino` | int8 | nncf | AVX-512 VNNI 권장 |

**범위 선택 이유**: #00–22가 GPU, #22까지는 `brevitas_int8_entropy`(parked). #23–29는 미래 GPU recipe 여유 공간으로 비워둠. #30부터 CPU 블록 시작 — 번호 점프가 곧 **타겟 하드웨어 전환의 시각적 신호**가 된다.

**Parked 후보**:
- `#36 openvino_int8_qat` (학습 파이프라인 붙으면) — Wave 6에서는 공간만 예약, 실행 안 함
- `#37 ort_cpu_int4_weight_only` (ORT 1.19+ matmul_nbits) — Vision 검증되면 추가

---

## Measurement Hygiene Additions

CPU 추론 고유의 variance 원인 때문에 기존 `measure.py` 프로토콜을 그대로 쓰면 신뢰할 수 없는 수치가 나온다. 다음을 **Task 6에서 보강**한다.

| 항목 | GPU 기준 (현재) | CPU 보강 |
|---|---|---|
| warmup iters | 100 | **200+** (JIT/커널 선택/캐시 priming) |
| measure iters | 100 | 300 (variance 흡수) |
| clock lock | `nvidia-smi -lgc` | **Linux**: `cpupower frequency-set -g performance`, **Windows**: `powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c` (High performance) |
| thread count | n/a | `session_options.intra_op_num_threads = recipe.measurement.thread_count` — **물리 코어 수로 고정** |
| NUMA | n/a | single-node만 (`numactl --cpunodebind=0 --membind=0` on Linux) |
| cooldown | 없음 | iter 사이 10ms sleep (옵션, thermal throttle 감지되면) |
| Result.env 추가 필드 | `gpu`, `cuda`, `tensorrt` | `cpu_model`, `cpu_cores_physical`, `cpu_flags` (VNNI/AMX/BF16 bool), `thread_count`, `governor`, `numa_node` |

---

## Task Dependency Graph

```
Task 1 (schema)   Task 2.5 (_weights_io 추출 — BLOCKER)
   │                   │
   ├── Task 2 (env_lock CPU) ───────┐
   │                                │
   └──────────► Task 3 (run_cpu skeleton — Task 2.5 필요)
                        │           │
                        ├── Task 4 (ORT static INT8)
                        │   │
                        │   └── Task 6 (hygiene) ────► Task 7 (recipes #30-33)
                        │
                        └── Task 5 (OpenVINO + NNCF) ──► Task 7 (recipes #34-35)
                                                                     │
                                                                     ├── Task 8 (batch + Makefile)
                                                                     │
                                                                     └── Task 9 (docs)
                                                                               │
                                                                               └── Task 10 (smoke E2E)
```

**병렬 가능**:
- Task 1 ∥ Task 2 ∥ Task 2.5 — 서로 다른 파일, 의존 관계 없음
- Task 4 (ORT static INT8) ∥ Task 5 (OpenVINO NNCF) — 서로 다른 프레임워크, 동일 `run_cpu.py` 파일이지만 함수 단위 분리

**필수 순서**:
- Task 2.5 → Task 3 (Task 3이 `_weights_io`에서 import)
- Task 3 → Task 4 / Task 5 (dispatcher 확장)
- Task 1 → Task 7 (recipe YAML이 확장된 schema 필요)

---

## Task 1: Schema 확장

**Why**: CPU 필드를 `Result.env`가 못 실으면 downstream `recommend.py`가 CPU 결과를 받아도 구분·정렬 못 한다. `technique.source`도 Literal 갱신 안 하면 레시피 로드에서 pydantic이 차단.

**Files**: `scripts/_schemas.py`, `tests/test_schemas.py` (신규 or 확장)

- [ ] **Step 1: Write the failing test** — `tests/test_schemas.py`에 CPU 레시피 validation 케이스 추가
  - `ort_cpu` / `openvino` source가 모두 `Recipe.model_validate`를 통과해야 함
  - `hardware.cpu: {model: "Intel Xeon 8480+", cores_physical: 56, ...}` 필드 설정 시 정상 파싱
  - `measurement.thread_count: 8` 필드 추가됐는지 확인
- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_schemas.py -x -q`
- [ ] **Step 3: Extend `_schemas.py`**
  - `TechniqueSpec.source`: `Literal["trt_builtin", "modelopt", "ort_quant", "brevitas", "ort_cpu", "openvino"]`
  - `RuntimeSpec.dtype`: **현재 값 확인 후** `bf16` 추가 → `Literal["fp32", "fp16", "bf16", "int8"]`. #31 recipe에 필수 (pydantic validation).
  - `RuntimeSpec.engine`: `openvino` 추가 → `Literal[..., "onnxruntime", "openvino"]` (현재 값 확인)
  - `HardwareSpec`에 `cpu_model: Optional[str]`, `cpu_cores_physical: Optional[int]`, `cpu_flags: Optional[list[str]]`, `numa_node: Optional[int]`, `governor: Optional[str]`
  - `MeasurementSpec.thread_count: Optional[int] = None` (None이면 physical cores auto-detect, YAML `null` → Python None)
  - `Result.env`에 cpu_model/cpu_cores_physical/cpu_flags 세 필드 추가 (optional, GPU 결과에서는 null)
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: 기존 21개 recipe 전부 재로드 확인**
  - `python -c "import sys; sys.path.insert(0, '.'); from scripts._schemas import load_recipe; import pathlib; [load_recipe(str(p)) for p in pathlib.Path('recipes').glob('*.yaml')]"`
  - 하나라도 실패하면 Literal 확장이 역호환 깼다는 뜻 — 즉시 조사
- [ ] **Step 5b: recommend.py 역호환 sanity**
  - `python scripts/recommend.py --results-dir results_qr --out /tmp/sanity.md --exclude brevitas_int8_entropy`
  - 기존 Wave 5 QR 리포트(Wave 5의 `report_qr.md`와 동등한 내용)가 그대로 재생성 확인
  - Result JSON의 env에 CPU 필드가 없어도(기존 GPU run들) 파싱·랭킹에 영향 없어야 함 (new optional fields는 하위호환 유지)
- [ ] **Step 6: Commit** — `feat(schemas): CPU fields on HardwareSpec/Result + ort_cpu/openvino sources`

---

## Task 2: `env_lock.py` CPU 지원

**Why**: 현재 `env_lock.py`는 nvidia-smi / CUDA 캡처만 한다. CPU 결과의 재현성과 순위 해석에 CPU 스펙·governor·thread 수가 필수. 이게 없으면 "Xeon 8480+ 결과와 i9-13900K 결과"를 구분 못 해 report.md가 무의미해진다.

**Files**: `scripts/env_lock.py`, `tests/test_env_lock.py`

- [ ] **Step 1: Write the failing test** — mock 환경에서 CPU 필드가 채워지는지 확인 (플랫폼별: Linux `/proc/cpuinfo`, Windows `wmic` / `Get-CimInstance`, macOS `sysctl`)
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement CPU detection**
  - `cpuinfo` 파이썬 패키지 사용 고려 (crossplatform) 또는 직접 파싱
  - `isa_features`: `flags` 파일에서 `avx2`, `avx512f`, `avx512_vnni`, `avx512_bf16`, `amx_tile` 존재 여부를 bool dict로 반환
  - `cores_physical`: hyperthreading 무시한 실제 코어 수
  - `governor`: Linux만 — `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
- [ ] **Step 4: CPU clock/governor lock 래퍼 함수 추가**
  - `lock_cpu_clock(enabled: bool) -> bool` — Linux `cpupower`, Windows `powercfg`, macOS 무시
  - 실패 시 `Result.notes`에 "could not lock CPU clock, thermal variance possible" 기록 (degrade-not-crash 원칙)
- [ ] **Step 5: Run test to verify it passes**
- [ ] **Step 6: Commit** — `feat(env_lock): CPU detection + clock/governor lock`

---

## Task 2.5: TRT 독립 공통 헬퍼 추출 (BLOCKER for Task 3)

**Why**: 현재 `scripts/run_trt.py`는 top-level에 `import tensorrt as trt` 등 CUDA/TRT imports를 가짐. `run_cpu.py`가 `run_trt.py`의 `_resolve_weights`·`_export_onnx`를 그냥 import 하면 **CPU-only 환경(CI, macOS, TRT 미설치 Windows)에서 ImportError**. CLAUDE.md에 추가하려는 규칙 "CPU runner must not import `tensorrt` at module load time"과 정면 충돌. 공통 로직을 TRT 독립 모듈로 먼저 추출해야 Task 3이 안전하게 재사용 가능.

**Files**: `scripts/_weights_io.py` (NEW), `scripts/run_trt.py` (refactor), `scripts/run_cpu.py` (Task 3에서 신규 import), `tests/test_weights_io.py` (NEW)

- [ ] **Step 1: 현재 `run_trt.py`에서 TRT 독립 가능 함수 식별**
  - `_resolve_weights(recipe)` — 순수 pydantic/pathlib + `ultralytics.YOLO` + 조건부 `modelopt` import
  - `_export_onnx(weights, imgsz, half, cache_dir, ...)` — 순수 `ultralytics.YOLO.export` 호출
  - `_load_yolo_for_restore(base_path)` — 순수 ultralytics import
  - `_apply_modelopt_sparsify`, `_apply_modelopt_qat` — modelopt 의존, CPU runner 안 씀 → **유지**
- [ ] **Step 2: Write failing test** — `tests/test_weights_io.py`
  - `from scripts._weights_io import resolve_weights, export_onnx` 성공
  - import 시 `tensorrt`, `pycuda`가 sys.modules에 로드되지 않음 (`import tensorrt` 같은 top-level 없는지)
  - `resolve_weights` / `export_onnx` 동작이 기존 run_trt 경로와 동일한지 (기존 test 재사용 / 얇은 smoke)
- [ ] **Step 3: Run test to verify it fails** (모듈 없음)
- [ ] **Step 4: Create `scripts/_weights_io.py`**
  - 위 3개 함수를 그대로 이식. `modelopt` import는 **함수 내부 lazy import** 유지
  - top-level imports 최소화: `pathlib`, `torch` (weight 조작), `ultralytics.YOLO` (skeleton 로딩)
  - **금지**: `import tensorrt`, `import pycuda`, `from scripts.run_trt import *`
- [ ] **Step 5: `run_trt.py`에서 해당 함수들 제거 후 re-export**
  - `from scripts._weights_io import resolve_weights, export_onnx`
  - 기존 호출 지점은 모두 동일한 이름으로 참조 가능 (alias 유지)
  - `_MAIN_TRAINED_YOLO` 전역 상태는 `run_trt.py`에 남겨둠 (TRT runner만 사용)
- [ ] **Step 6: Run test to verify it passes** + 기존 `pytest tests/ -q` 전체가 여전히 통과 (특히 `test_run_trt_trained_weights.py`)
- [ ] **Step 7: Import safety 검증**
  ```bash
  python -c "import sys; import scripts._weights_io; assert 'tensorrt' not in sys.modules; assert 'pycuda' not in sys.modules; print('OK: TRT-free import')"
  ```
- [ ] **Step 8: Commit** — `refactor(scripts): extract TRT-independent _weights_io module (prep for Wave 6 CPU)`

---

## Task 3: `run_cpu.py` skeleton

**Why**: CPU runner를 `run_trt.py`·`run_ort.py`에서 분리하는 이유: TRT/CUDA 런타임 import 없이도 CPU 경로가 동작해야 CI/Mac 노트북 환경에서도 `pytest tests/`가 깨지지 않는다.

**Files**: `scripts/run_cpu.py` (NEW), `tests/test_run_cpu_fp32.py` (NEW)

- [ ] **Step 1: Write the failing test** — YOLO26n을 ONNX로 export한 뒤 ORT CPU EP로 1회 inference 성공 여부만 검증 (정확도는 Task 7 통합 테스트에서)
- [ ] **Step 2: Run test to verify it fails** (파일 자체가 없어서 ImportError)
- [ ] **Step 3: Implement skeleton**
  - 기존 `run_trt.py`의 CLI 시그니처 재사용: `--recipe`, `--out`
  - **Task 2.5에서 추출한 `scripts._weights_io`에서 import** — 절대 `from scripts.run_trt import ...` 하지 않음 (TRT import 연쇄 방지)
    ```python
    from scripts._weights_io import resolve_weights, export_onnx  # TRT-free
    ```
  - `resolve_weights(recipe)` 재사용 (OMNI_WEIGHTS_OVERRIDE 동일)
  - `export_onnx(weights, imgsz, half=False)` — dynamic=True
  - FP32 경로: `ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])`
  - session_options: `graph_optimization_level=ORT_ENABLE_EXTENDED` (권장 기본값 — ENABLE_ALL은 공격적 layout transform이 YOLO26n attention block에서 드물게 그래프 break 유발, 문제 없는 모델만 ALL로 승격)
  - `measure.py`의 기존 warmup/measure loop 호출 (반복 로직은 재사용)
  - mAP eval: `eval_coco.py` 재사용 (ONNX 경로 이미 존재)
  - **top-level import 금지 목록**: `tensorrt`, `pycuda`, `scripts.run_trt` — 모듈 로드 시점에 전부 부재여도 작동해야 함
- [ ] **Step 3b: 공통 calibration loader 추가 (Task 4 + Task 5 공유)**
  - 신규 `_build_calib_numpy_array(recipe: Recipe) -> np.ndarray` in `run_cpu.py`
  - 출력 shape: `(N, 3, imgsz, imgsz)` float32, N = `recipe.technique.calibration_samples or 512`
  - **데이터 소스**: `recipe.technique.calibration_dataset` (e.g. `coco_val2017`)를 사용. Wave 5 QR 평가 패턴 유지 — `OMNI_WEIGHTS_OVERRIDE=best_qr.pt`로 eval weights만 QR 치환하고 **calibration 이미지는 환경변수 `OMNI_CALIB_YAML`(default: `coco_val2017`) 경로에서 로드**. QR val 133장만으로는 statistics 부족해 Wave 5에서 `coco_val_only.yaml`로 분리한 결정을 Wave 6에도 계승.
  - 대상 로더: `run_trt.py`의 기존 calib 로더가 TRT 독립이면 `_weights_io`로 함께 옮기거나 별도 `_calib_io.py` 고려
  - `run_ort.py`의 `_NumpyReader` wrapper 패턴 확인 후 재사용 여부 결정 (중복 구현 방지)
  - Task 4는 ORT용 `_NumpyReader(arr, input_name)` 래핑, Task 5는 NNCF `nncf.Dataset(generator, transform_func=...)` 래핑 — **array 자체는 한 곳에서 빌드**
  - 결정적 재현: `recipe.technique.calibration_seed` (default 42)로 샘플 선택 인덱스 고정. NNCF 내부 RNG는 제어 불가 가능 — 실측 시 소량 drift 허용
- [ ] **Step 3c: Spike 아티팩트 정리**
  - `scripts/_spike_wave6_r1.py`의 Stage 1/2/3 로직이 Task 5(OpenVINO)와 Task 4(ORT)로 이식 완료되면 spike 스크립트 삭제
  - `best_qr.onnx` (export 캐시) + `results_cpu/_ov_ir/best_qr_*.xml/.bin`는 `.gitignore`에 포함된 상태로 유지, 개발 루프에서 재사용
  - 참고용으로 보존하고 싶으면 `docs/spikes/2026-04-21-wave6-r1-spike.md`에 결과 요약만 남김 (스크립트 자체는 `run_cpu.py`로 흡수됐으므로 삭제)
- [ ] **Step 4: Dispatcher branch**
  - `recipe.technique.source == "ort_cpu"` && `dtype == "fp32"` → fp32 경로
  - 그 외 케이스는 `NotImplementedError` (Task 4/5에서 확장)
- [ ] **Step 5: Run test to verify it passes**
- [ ] **Step 6: Commit** — `feat(run_cpu): skeleton runner, ort_cpu fp32 path + shared calib loader`

---

## Task 4: ORT CPU static INT8 (VNNI)

**Why**: Dynamic INT8은 편하지만 static INT8이 CPU VNNI의 주 타겟. Wave 3에서 이미 얻은 symmetric QDQ 설정 지식(P0-A 패치)을 그대로 CPU EP에 적용 가능 — TRT는 QDQ를 해석해 engine 빌드하고 CPU EP는 MLAS 커널로 직접 실행만 한다는 점이 다를 뿐.

**Files**: `scripts/run_cpu.py`, `tests/test_run_cpu_int8.py` (NEW)

- [ ] **Step 1: Write the failing test** — int8 dynamic과 static 둘 다 session 생성·1회 forward pass 성공 확인
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `_quantize_ort_cpu_static`**
  ```python
  from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod
  quantize_static(
      model_input=str(clean_onnx),
      model_output=str(cached_int8_onnx),
      calibration_data_reader=_NumpyReader(calib_arr, input_name),
      quant_format=QuantFormat.QDQ,
      activation_type=QuantType.QInt8,
      weight_type=QuantType.QInt8,
      per_channel=True,
      reduce_range=False,  # CPU VNNI는 full-range OK, 구형 CPU는 True 필요할 수 있음
      calibrate_method={
          "minmax": CalibrationMethod.MinMax,
          "entropy": CalibrationMethod.Entropy,
          "percentile": CalibrationMethod.Percentile,
      }[recipe.technique.calibrator],
      extra_options={
          "ActivationSymmetric": True,
          "WeightSymmetric": True,
          "AddQDQPairToWeight": True,
          "DedicatedQDQPair": False,  # CPU EP는 False 권장 (TRT의 True와 반대)
      },
  )
  ```
  - **중요 차이**: `DedicatedQDQPair`는 TRT explicit quantization 요구사항이고 CPU EP MLAS에는 오히려 그래프가 복잡해져 커널 fusion 방해 가능 — ORT 문서 재확인 필요
  - `reduce_range=False`: CPU VNNI 없는 환경에서 saturation overflow가 나면 `True`로 폴백 고려
- [ ] **Step 3b: DedicatedQDQPair 실기 비교 (R3 해소)**
  - `DedicatedQDQPair=True`와 `False` 두 버전으로 quantize → 각각 `best_qr_int8_ded.onnx`, `best_qr_int8_fused.onnx` 캐시
  - 동일 input (random 1×3×640×640) 10회 warmup + 100회 latency 측정
  - 빠른 쪽을 `run_cpu.py` default로 채택, 느린 쪽과의 **percentage diff를 주석으로 기록**
  - `scripts/_schemas.py`의 recipe에 옵션 노출은 하지 않음 (결정된 값 하드코딩)
- [ ] **Step 4: Dynamic INT8 경로 추가**
  - `from onnxruntime.quantization import quantize_dynamic` — calibration 없이 weight-only + runtime activation scale
  - 더 단순하지만 일반적으로 static 대비 10–20% 느림
- [ ] **Step 5: `_prepare_cpu_session` 디스패치**
  - `dtype=="fp32"` → 원본 ONNX
  - `dtype=="int8" && calibrator is None` → dynamic quantize 결과
  - `dtype=="int8" && calibrator is not None` → static quantize 결과
  - 양자화 결과는 `results/_onnx/{recipe.name}_quant.onnx`로 캐시 (TRT 엔진 캐시와 동일 패턴)
- [ ] **Step 6: BF16 경로 (optional, AMX 필요)**
  - `onnxruntime.transformers.float16.convert_float_to_float16` 유사 — 하지만 CPU EP의 BF16 지원은 1.18+ 특정 ops에 한정
  - AMX 없는 CPU에서는 BF16 cast가 오히려 느려질 수 있어 **hardware gate**: `cpu_flags.amx_tile or cpu_flags.avx512_bf16 must be True`, 아니면 `meets_constraints=False` 기록하고 스킵
- [ ] **Step 7: Run test to verify it passes**
- [ ] **Step 8: Commit** — `feat(run_cpu): ORT CPU EP static/dynamic INT8 + BF16 path`

---

## Task 5: OpenVINO + NNCF PTQ

**Why**: OpenVINO는 Intel CPU에서 ORT CPU EP보다 **10–30% 빠른 케이스가 흔함** (특히 conv-heavy 모델) — 자체 커널 라이브러리가 더 타겟팅 되어 있어서. NNCF는 OpenVINO의 공식 양자화 툴로, ONNX에서 곧바로 INT8 IR(.xml/.bin) 생성. 별도 TRT-style explicit QDQ 조립이 필요 없다.

**Files**: `scripts/run_cpu.py` (확장), `pyproject.toml`, `tests/test_run_cpu_openvino.py` (NEW)

- [ ] **Step 1: Add OpenVINO extras to `pyproject.toml`**
  ```toml
  [project.optional-dependencies]
  cpu = ["openvino>=2024.3", "nncf>=2.9,<3.0"]
  ```
- [ ] **Step 2: Write the failing test** — ONNX → OpenVINO IR conversion + infer 1 sample
- [ ] **Step 3: Run test to verify it fails**
- [ ] **Step 4: Implement `_prepare_openvino_model`**
  ```python
  import openvino as ov
  # Core 재사용: 모듈 전역 싱글톤으로 ~300ms 초기화 비용 1회만 발생
  _OV_CORE: ov.Core | None = None
  def _get_core() -> ov.Core:
      global _OV_CORE
      if _OV_CORE is None:
          _OV_CORE = ov.Core()
      return _OV_CORE

  core = _get_core()
  ov_model = core.read_model(str(onnx_path))
  if recipe.runtime.dtype == "int8":
      import nncf
      calib_dataset = nncf.Dataset(calib_loader, transform_func=lambda x: x)
      ov_model = nncf.quantize(
          ov_model,
          calib_dataset,
          preset=nncf.QuantizationPreset.MIXED,  # activations INT8, sensitive layers FP32
          target_device=nncf.TargetDevice.CPU,
          subset_size=recipe.technique.calibration_samples or 300,  # NNCF default
      )
      ov.save_model(ov_model, str(cached_ir_path))

  # PerformanceHint는 **batch size별로 별도 compile**. bs=1은 LATENCY, bs>1은 THROUGHPUT.
  # 두 개의 compiled_model을 측정에 각각 사용 (bs=1 fps ≠ bs=8 fps 비교가 공정해짐).
  compiled_by_bs: dict[int, ov.CompiledModel] = {}
  for bs in recipe.measurement.batch_sizes:
      hint = "LATENCY" if bs == 1 else "THROUGHPUT"
      compiled_by_bs[bs] = core.compile_model(ov_model, "CPU", {
          "PERFORMANCE_HINT": hint,
          "INFERENCE_NUM_THREADS": str(recipe.measurement.thread_count or 0),
          "CPU_BIND_THREAD": "YES",  # 스레드 이주 방지 (latency 측정 필수)
      })
  ```
  - `transform_func` (NNCF 2.19+): `transform_fn`은 deprecated. spike에서 확인됨.
  - `subset_size`: NNCF 기본 300. 512로 재정의하면 calibration 시간만 길어지고 정확도 gain 미미 — default 따라가기.
- [ ] **Step 5: Inference wrapper**
  - OpenVINO의 `InferRequest` 인터페이스는 ORT와 다름 — input tensor dict, output은 result dict
  - `eval_coco.py`가 ORT session을 기대하므로 OpenVINO를 ORT-compatible callable로 래핑하는 어댑터 필요
  - 어댑터: `class OVRunnerAsORT: def run(self, output_names, input_dict) -> list[np.ndarray]`
- [ ] **Step 6: Dispatcher branch for `source == "openvino"`**
- [ ] **Step 7: Run test to verify it passes**
- [ ] **Step 8: Commit** — `feat(run_cpu): OpenVINO runtime + NNCF INT8 PTQ`

---

## Task 6: Measurement hygiene

**Why**: CPU variance는 GPU보다 체계적으로 크다. 같은 recipe를 10번 돌리면 GPU는 ±1% 안쪽, CPU는 thermal/scheduling 때문에 ±5% 이상 튈 수 있다. Warmup·iter·lock 기본값을 CPU 전용으로 조정하고 `_env.json`에 측정 환경 전체를 기록해야 recommend.py 해석 가능.

**Files**: `scripts/measure.py` (확장), `scripts/env_lock.py` (이미 Task 2), `scripts/run_cpu.py`

- [ ] **Step 1: Write test** — CPU recipe에서 warmup_iters/measure_iters가 default override 되는지
- [ ] **Step 2: measure.py에 CPU 전용 default 추가**
  - `if recipe.runtime.engine in ("onnxruntime", "openvino") and recipe.technique.source in ("ort_cpu", "openvino"):` 블록
  - `warmup_iters = max(recipe.measurement.warmup_iters, 200)`
  - `measure_iters = max(recipe.measurement.measure_iters, 300)`
  - per-iter `time.sleep(0.010)` (thermal cooldown, 옵션)
  - **Measurement loop timing 원칙**: latency 측정 구간은 **모델 forward + NMS만** 포함. 이미지 로딩, letterbox, normalize(uint8→float32), NHWC→NCHW transpose는 loop 밖에서 1회 변환 후 캐시된 np.ndarray 텐서를 재사용. 이 원칙은 GPU 경로와 동일하지만 CPU에서 preprocessing 비중이 상대적으로 커서 (~1-2ms) 명시적으로 지켜야 함.
- [ ] **Step 3: Thread count 자동 감지**
  - recipe가 `thread_count=null` (YAML `null` → Python `None`)이면 자동 감지 동작:
    - 1차: `psutil.cpu_count(logical=False)` (crossplatform)
    - 2차 fallback (psutil 없음): Linux `lscpu -p | grep -v '^#' | awk -F, '{print $2}' | sort -u | wc -l`
    - 3차 fallback (Windows): `os.cpu_count() // 2` (hyperthreading 가정)
  - 감지 결과를 `int`로 `session_options.intra_op_num_threads` 및 OpenVINO `INFERENCE_NUM_THREADS`에 전달
  - recipe가 명시적 정수면 그 값 사용 (auto-detect 건너뜀) — thread 스케일링 실험 시 유용
  - **`thread_count=0` 금지**: ORT 기본 0은 "logical cores 전부 사용"이라 hyperthreading까지 잡혀서 오히려 느림. pydantic validator로 0 거부 또는 최소 1.
- [ ] **Step 4: p50/p95/p99 + stddev 기록**
  - `LatencyStats.stddev_ms: Optional[float] = None` 추가 (default None)
  - **Decision**: GPU 결과에도 소급 적용 — 하지만 하위 호환 유지. 과거 JSON은 `stddev_ms=None` 상태 그대로 허용, 재실행 시 자동으로 채워짐. `recommend.py`는 `stddev_ms`가 있으면 리포트에 포함, 없으면 생략.
- [ ] **Step 5: Run tests**
- [ ] **Step 6: Commit** — `feat(measure): CPU warmup/iter defaults + thread count auto`

---

## Task 7: Recipe YAML #30–#35

**Why**: 레시피가 있어야 E2E 검증 가능. 기존 YAML 스타일(indent, 필드 순서)을 그대로 유지.

**Files**: `recipes/30_ort_cpu_fp32.yaml` ~ `recipes/35_openvino_int8_nncf.yaml`

- [ ] **Step 1: `30_ort_cpu_fp32.yaml`**
  ```yaml
  name: ort_cpu_fp32
  model:
    family: yolo26
    variant: n
    weights: yolo26n.pt
  runtime:
    engine: onnxruntime
    dtype: fp32
  technique:
    name: fp32
    source: ort_cpu
  hardware:
    gpu: null
    cpu_cores_physical: null  # auto
  measurement:
    dataset: coco_val2017
    num_images: 500
    warmup_iters: 200
    measure_iters: 300
    batch_sizes: [1, 8]
    input_size: 640
    thread_count: null  # auto → physical cores
    seed: 42
  constraints:
    max_map_drop_pct: 1.0
    min_fps_bs1: null  # FP32 baseline — fps threshold 없음 (어차피 feasible 비교 대상 아님)
  ```
  - **`min_fps_bs1` 계층 정책** (spike 수치 근거: Tiger Lake FP32 ~5fps, INT8 ~13fps):
    - FP32 / BF16 baselines (#30, #31, #34): `null` — baseline 역할, fps constraint 없음
    - INT8 recipes (#32, #33, #35): `10` — 실용 임계, VNNI 하드웨어에서 통과 기대
  - 이유: 동일 threshold를 전 recipe에 쓰면 FP32는 항상 FAIL, 리포트가 구조적으로 편향됨
- [ ] **Step 2: `31_ort_cpu_bf16.yaml`** — `dtype: bf16`, 하드웨어 요구 `requires_isa: ["avx512_bf16"]` 추가 (hardware gate 필드), `min_fps_bs1: null`
- [ ] **Step 3: `32_ort_cpu_int8_dynamic.yaml`** — `dtype: int8`, `calibrator: null`, `min_fps_bs1: 10`
- [ ] **Step 4: `33_ort_cpu_int8_static.yaml`** — `dtype: int8`, `calibrator: entropy`, `calibration_samples: 512`, `min_fps_bs1: 10`
- [ ] **Step 5: `34_openvino_fp32.yaml`** — `source: openvino`, `runtime.engine: openvino`, `min_fps_bs1: null`
- [ ] **Step 6: `35_openvino_int8_nncf.yaml`** — `source: openvino`, `calibrator: nncf`, `calibration_samples: 300` (NNCF default), `min_fps_bs1: 10`
- [ ] **Step 7: 22개 기존 recipe + 6개 신규 전부 `load_recipe` 통과 확인**
- [ ] **Step 8: Commit** — `feat(recipes): #30-35 CPU backends (ort_cpu, openvino)`

---

## Task 8: Batch scripts + Makefile + pyproject

**Why**: 단독 실행뿐 아니라 `make all`과 병렬로 CPU bank도 돌 수 있어야 한다. 단, 기본 `make all`에 CPU 레시피를 넣으면 **GPU 노트북에서도 CPU 부하가 발생**해 GPU 측정 variance에 영향. 별도 target `make cpu-all`이 올바른 분리.

**Files**: `Makefile`, `scripts/run_cpu_batch.sh` (NEW), `pyproject.toml`

- [ ] **Step 1: Makefile targets #30–#35**
  - `.PHONY`에 recipe-30~35 추가
  - 각 recipe-N 타겟은 `python scripts/run_cpu.py --recipe ... --out ...`
- [ ] **Step 2: `make cpu-all` 신규 타겟** — 위 6개 + `make cpu-report`
- [ ] **Step 3: `make cpu-report` 신규** — `python scripts/recommend.py --results-dir results_cpu --out report_cpu.md --exclude "openvino_int8_qat"`
- [ ] **Step 4: `scripts/run_cpu_batch.sh`** — GPU batch script의 CPU 버전. `OMNI_WEIGHTS_OVERRIDE=best_qr.pt` 동일 패턴. `results_cpu_qr/` 출력
- [ ] **Step 5: `.gitignore` 업데이트** — `results_cpu/_onnx/`, `results_cpu/_ov_ir/` 추가
- [ ] **Step 6: `pyproject.toml` extras**
  ```toml
  [project.optional-dependencies]
  cpu = ["openvino>=2024.3", "nncf>=2.9,<3.0", "psutil>=5.9"]
  ```
  `all` extras에도 포함
- [ ] **Step 7: Verify recipes still load + tests still pass**
- [ ] **Step 8: Commit** — `chore(infra): CPU batch scripts + Makefile targets + pyproject`

---

## Task 9: Documentation

**Why**: `docs/architecture.md`가 "current scope" 절에 GPU 전제로 적혀 있다. Wave 6 착지 시 해당 블록 갱신 안 하면 다음 세션의 Claude가 잘못된 전제로 리팩터.

**Files**: `docs/architecture.md`, `CLAUDE.md`, `README.md`

- [ ] **Step 1: `docs/architecture.md`**
  - "Current scope" 블록에 CPU 타겟 추가
  - 신규 "Wave 6 — CPU inference (2026-04-2X)" 섹션 추가 (Wave 5 스타일 매칭)
  - Backends 테이블 갱신 (ort_cpu, openvino 추가)
  - Extension hook 절에 `run_cpu.py` 디스패치 설명
  - **`cold_start_ms` 해석 주의 문구 추가** — backend별로 의미 다름:
    - GPU TRT: engine cache hit ~50ms / miss 수 분 (plan·calibration 포함)
    - ORT CPU: `InferenceSession` 생성 ~200ms (그래프 opt)
    - OpenVINO: `read_model` + `compile_model` 합쳐 **~500ms–1.5s** (LATENCY hint 시 특히)
    - `report.md` 랭킹에서 cold_start는 참고치로만 표시, latency/fps와 별도 축으로 해석
- [ ] **Step 2: `CLAUDE.md`**
  - "Current scope" 한 문장 갱신: "one NVIDIA GPU + x86_64 Intel CPU"
  - Critical conventions에 CPU 관련 규칙 1개 추가: "CPU runner must not import `tensorrt` at module load time"
- [ ] **Step 3: `README.md`**
  - Usage 섹션에 `pip install -e ".[cpu]"` + `make cpu-all` 추가
  - 타겟 매트릭스 테이블 추가 (recipe → 필수 하드웨어 ISA)
- [ ] **Step 4: Commit** — `docs(architecture): Wave 6 CPU inference section`

---

## Task 10: Smoke E2E 검증 (이름만 smoke, 실제론 6개 recipe 전수 실행)

**Why**: Wave 5에서 겪은 것처럼 실제 실행 전엔 버그 종류를 예측 못 한다. QR 데이터셋에서 6개 recipe 모두 돌려 `results_cpu_qr/*.json`과 `report_cpu_qr.md`를 산출하고, 각 recipe가 실제로 constraint 만족/실패 어느 쪽인지 확인해야 Wave 6 "active"로 선언 가능.

**Files**: (런타임만, 코드 변경 없음)

- [ ] **Step 1: Verify prerequisites** — `best_qr.pt`, `qr_barcode.yaml`, Python env에 `openvino`/`nncf` 설치됐는지
- [ ] **Step 2: Smoke run #30 (fastest: FP32)** — `python scripts/run_cpu.py --recipe recipes/30_ort_cpu_fp32.yaml --out /tmp/smoke_30.json`
- [ ] **Step 3: 결과 sanity check**
  - mAP > 0.95 (QR finetuned baseline이므로)
  - p50_ms 수치가 수십~수백 ms (CPU 기대 범위)
  - env.cpu_model 채워져 있음
  - env.thread_count 채워져 있음
- [ ] **Step 4: 전체 실행** — `bash scripts/run_cpu_batch.sh`
- [ ] **Step 5: Report 생성** — `python scripts/recommend.py --results-dir results_cpu_qr --out report_cpu_qr.md --exclude "openvino_int8_qat"`
- [ ] **Step 6: Result comparison** — GPU QR 결과(`report_qr.md`)의 top 3와 CPU top 3 나란히 놓고 권고 차이 확인
- [ ] **Step 7: 만약 #31 `ort_cpu_bf16`이 하드웨어 부재로 skip됐다면 `meets_constraints=False` + `notes`에 기록됐는지 확인** (degrade-not-crash 원칙)
- [ ] **Step 8: Commit results** — `feat(results): Wave 6 CPU eval + report (30-35)` (Wave 5 패턴 매칭)

---

## Known Risks & Decisions

### R1. OpenVINO + YOLO26n attention block 호환성 — **CLEARED 2026-04-21**
**Risk (원문)**: Wave 3 INC 경로에서 SmoothQuant가 YOLO26n attention block의 Reshape nodes를 stale 상태로 남겼다. OpenVINO NNCF도 유사한 그래프 변형을 하므로 동일 이슈가 재현될 가능성이 있음.
**Verification**: `scripts/_spike_wave6_r1.py` 실행 결과 (Intel i7-11375H Tiger Lake, AVX-512 VNNI):
- Stage 0 ONNX export: PASS (best_qr.onnx 9.7MB)
- Stage 1 OpenVINO FP32 read/compile/infer: PASS (196ms 첫 inference, no warmup)
- Stage 2 NNCF PTQ with MIXED preset + 32 random samples: PASS (16.4s, **no ignored_scope needed**)
- Stage 3 OpenVINO INT8 infer: PASS (74.3ms, output shape preserved `[1,300,6]`)
**Outcome**: Default NNCF MIXED preset이 YOLO26n attention block을 정상 처리 → `ignored_scope` mitigation 불필요. Task 5 Step 4의 `ignored_scope` 옵션 언급은 fallback으로만 유지.

### R2. BF16 하드웨어 gating
**Risk**: `#31 ort_cpu_bf16`은 AMX(SPR+) 또는 AVX-512 BF16(Cooper Lake+) 필요. 개발/CI 환경에서 이런 CPU가 없으면 recipe가 항상 skip되고 Wave 6 "완료" 판정이 모호해짐.
**Mitigation**: Task 2에서 `cpu_flags` 감지 → Task 4 Step 6에서 조기 스킵 + notes 기록. recommend.py에서 skipped recipe는 별도 섹션 표시.
**Decision**: #31은 optional로 선언. AMX 없는 환경에서 Wave 6 "active" 판정 가능.

### R3. ORT DedicatedQDQPair 옵션
**Risk**: TRT는 `DedicatedQDQPair=True` 필수, CPU EP는 False 권장이라는 내 이해는 ORT 문서로 **실기 확인 필요**.
**Mitigation**: Task 4 Step 3 착수 전에 ORT 1.17+ 문서(`onnxruntime/python/tools/quantization/quantize.py`) 재확인. 틀렸다면 True로 설정.

### R4. Windows `cpupower` 부재
**Risk**: Linux `cpupower frequency-set`는 Windows에 없다. `powercfg`로 "High performance" 전원 계획 전환 가능하지만 실제 주파수 고정은 아님.
**Mitigation**: Windows에서는 `notes`에 "frequency governor not locked, thermal variance possible" 기록하고 그대로 진행. Linux에서만 하드 락.
**Decision**: degrade-not-crash 유지.

### R5. macOS Apple Silicon
**Risk**: OpenVINO는 Apple Silicon 정식 지원 없음. ORT CPU EP는 돌아가지만 CoreML EP가 훨씬 빠름.
**Decision**: Wave 6 범위 밖. Wave 7에서 `coreml_cpu` source 추가 고려. `pyproject.toml` cpu extra가 macOS ARM에서 `openvino` 설치 실패하더라도 프로젝트 코어는 깨지지 않도록 optional import.

### R6. 의존성 버전 충돌 (torch 2.8 vs NNCF 2.19 권장 2.9 / modelopt setuptools)
**Risk**: Spike 중 발견 — NNCF 2.19.0이 `torch==2.9.*` 권장, 현재 `2.8.0+cu129`에서 경고만 뜸. 또한 `openvino` 설치 시 `nvidia-modelopt 0.43.0 requires setuptools>=80, you have 69.5.1` 충돌 경고. torch를 2.9로 올리면 Wave 5 `modelopt_qat`/`modelopt_sparsify` 경로가 회귀할 가능성. setuptools를 올리면 modelopt가 재설치될 수 있음.
**Mitigation**:
- torch 2.8.0+cu129 고정 유지 (Wave 5 검증된 버전)
- NNCF 경고는 무시 (spike에서 동작 확인됨)
- setuptools 69.5.1 유지 — modelopt 0.43.0이 최신 setuptools 요구를 명시했지만 실제 설치는 성공한 상태
- `pyproject.toml`의 `[cpu]` extra는 torch/modelopt 핀과 분리 (상호 간섭 방지)
**Decision**: torch 2.9 도입은 Wave 7에서 Wave 5/6 동시 회귀 테스트와 함께. Wave 6은 현재 환경 그대로 진행.

---

## Extension Gates (Wave 7+ candidates)

- **ARM NEON / SVE** — ORT CPU EP가 이미 ARM 지원, 레시피만 추가 가능. 측정 환경 (Graviton, Raspberry Pi 5, Apple M-series) 확보 필요.
- **Apple CoreML EP** — M-series 전용, ORT에 EP로 붙음. Neural Engine offload.
- **AMD CPU 튜닝** — Zen4+ VNNI는 이미 Wave 6 경로가 커버. 별도 튜닝은 필요 없을 수도.
- **INT4 weight-only** — ORT 1.19+ `matmul_nbits` — LLM에서 검증됐고 Vision 적용 연구 시작 단계.
- **CPU pruning** — Wave 5 training pipeline의 `prune_24` modifier는 GPU 2:4 Sparse Tensor Core 전용. CPU는 structured channel pruning이 유효 → 별도 modifier 필요.

---

## Success Criteria (Wave 6 "active" 선언 조건)

- [ ] `results_cpu_qr/{30..35}_*.json` 6개 파일 존재 (#31 BF16은 하드웨어 있을 때)
- [ ] `report_cpu_qr.md` 생성, ranking에 최소 5개 recipe 포함
- [ ] 최소 1개 recipe가 `meets_constraints=True` (mAP drop ≤ 1.0%p, fps ≥ min_fps_bs1)
- [ ] `docs/architecture.md` Wave 6 섹션 작성
- [ ] `pytest tests/ -q` 전 항목 pass
- [ ] `make cpu-all`이 GPU recipe와 독립적으로 실행 가능

---

## Non-goals (Wave 6 명시적 제외)

- ARM / Apple / AMD-specific 튜닝
- INT4 weight-only
- CPU-specific pruning/QAT 학습 파이프라인
- 분산 추론, batch size > 8
- 다른 모델 (YOLO26n 외)
