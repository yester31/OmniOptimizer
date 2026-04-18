# OmniOptimizer 개선 점검 — TensorRT 10 + NVIDIA Model Optimizer 문서 기반

**Audit date:** 2026-04-18  |  **TensorRT:** 10.16  |  **modelopt:** 0.43
**Scope:** `scripts/run_trt.py`, `scripts/run_ort.py`, `scripts/measure.py`,
`scripts/recommend.py`, 레시피 YAML.
**References:**
- TensorRT Developer Guide (10.x) — `docs.nvidia.com/deeplearning/tensorrt/latest/`
- NVIDIA Model Optimizer — `github.com/nvidia/model-optimizer/docs`

소스/버전별 API 변경을 교차 검증했습니다. 각 항목에 NVIDIA 문서 인용을 근거로
붙였고, 영향도는 이 저장소의 타깃(YOLO26n × RTX 3060 Laptop, bs=1/8) 기준입니다.

---

## Executive summary

| # | 개선 항목 | 영향도 | 공수 | 대상 파일 |
|---|---|---|---|---|
| H1 | **CUDA graph capture로 bs=1 latency 단축** | 🔥 매우 큼 | 중 | `run_trt.py::_make_trt_forward` |
| H2 | **Timing cache 공유 (engine build 시간 단축)** | 🔥 큼 | 소 | `run_trt.py::_build_engine` |
| H3 | **modelopt AutoTune / AutoQuantize 도입** | 🔥 큼 | 중~대 | `run_trt.py::_prepare_modelopt_onnx` + 신규 레시피 |
| M1 | TF32 플래그로 FP32 baseline 가속 | 중 | 소 | `run_trt.py::_build_engine` |
| M2 | `kOBEY_PRECISION_CONSTRAINTS` + `nodes_to_exclude` 강제 | 중 | 소 | `run_trt.py::_build_engine` |
| M3 | CUDA Event 기반 GPU time 측정 | 중 | 중 | `measure.py::measure_latency` |
| M4 | Polygraphy `--validate` / `--debug-precision` 보조 타겟 | 중 | 소 | `Makefile` + `scripts/debug_precision.py` 신규 |
| L1 | Random-normal calib fallback은 에러로 (무언의 정확도 훼손 방지) | 소 | 최소 | `run_trt.py::_build_engine`, `_prepare_modelopt_onnx` |
| L2 | Peak memory 측정 — torch + NVML 이중 보고 | 소 | 소 | `measure.py`, `_schemas.Result` |
| L3 | IInt8EntropyCalibrator2 경로는 deprecated 표식 | 소 | 최소 | 문서화 |

---

## H1. CUDA graph capture — bs=1 latency 단축

### 현재 상태
`run_trt.py::_make_trt_forward`는 매 반복마다 `context.execute_async_v3(stream.cuda_stream)`
를 직접 호출하고 `stream.synchronize()`로 완료를 기다립니다:

```python
def fwd():
    with torch.cuda.stream(stream):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
```

YOLO26n은 p50 ≈ 3.6 ms 수준이라 Python/TRT launch overhead (~0.3–0.8 ms)가 전체 latency의
10–20%를 차지합니다. bs=1에서 특히 두드러집니다.

### 근거 (NVIDIA 문서)

> "TensorRT supports CUDA graph capture for models that do not require
> mid-pipeline CPU interaction. By capturing the execution sequence into
> a graph instance, you can launch the graph directly for subsequent
> inferences, which can reduce CPU overhead compared to standard enqueue
> methods."
> — *TensorRT Performance > Best Practices > CUDA Graphs*

권장 패턴 (Python, 문서 코드):

```python
# Call execute_async_v3() once after an input shape change to update internal state.
context.execute_async_v3(stream)

# Capture a CUDA graph instance
cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureModeGlobal)
context.execute_async_v3(stream)
err, graph = cudaStreamEndCapture(stream)
err, instance = cudaGraphInstantiate(graph, 0)

for i in range(iterations):
    cudaGraphLaunch(instance, stream)
    cudaStreamSynchronize(stream)
```

### 권장 변경
`_make_trt_forward`에 1회 graph capture → 이후 `cudaGraphLaunch`로 대체. 조건:
- 입력 shape 변경 없는 고정 profile (현재 코드가 이미 고정 shape — OK)
- 동적 shape / 데이터 종속 제어 흐름 없음 (YOLO26 backbone은 이 조건 충족)

예상 이득: **bs=1 p50 −8%~−15%** (YOLO26n 규모 기준 경험치; measure 필요).

주의: capture 실패 시 (드문 edge case) `cudaStreamEndCapture`가 에러를 반환 — 이때는
기존 `execute_async_v3` 경로로 graceful fallback 유지.

### 공수
중. `cuda-python` 바인딩 추가 필요 (`pip install cuda-python`). `_make_trt_forward`에
capture/replay 분기 + try-fallback 추가.

---

## H2. Timing cache 재사용 — engine build 시간 단축

### 현재 상태
`_build_engine`는 timing cache를 설정하지 않습니다. 매 build마다 TRT가 모든 layer에 대해
tactic timing을 다시 프로파일합니다. YOLO26n fp16 engine build는 약 40–60초, int8은 90–120초.

### 근거

> "A layer-timing cache stores profiling information to avoid redundant
> timing operations during the build process. This cache is specific to
> the target device, CUDA version, TensorRT version, and specific
> builder configuration flags."
> — *TensorRT Performance > Best Practices > Timing Cache*

> "If a builder is not provided with an existing cache, it creates a
> temporary local cache that is discarded once the build completes."
> — 같은 문서

### 권장 변경
`results/_trt_timing.cache`를 프로젝트 타이밍 캐시로 추가:

```python
# 1) 캐시 로드 (없으면 새로 생성)
cache_path = Path("results/_trt_timing.cache")
cache_bytes = cache_path.read_bytes() if cache_path.exists() else b""
timing_cache = config.create_timing_cache(cache_bytes)
config.set_timing_cache(timing_cache, ignore_mismatch=False)

# 2) build 후 serialize 된 cache를 덮어쓰기
# (TRT 10: timing_cache.serialize() → bytes)
cache_path.write_bytes(bytes(timing_cache.serialize()))
```

예상 이득: **engine build 2회째 이후 40%~70% 시간 단축**. `make clean` 후 처음 빌드는
동일, 그 다음부터 이득. 개발 반복 속도에 직접 체감됨.

### 주의사항
- 캐시는 device / CUDA / TRT 버전별로 다름. `.cache` 파일에 `env.cuda + env.trt + gpu_name`
  서명을 prefix로 넣어서 잘못된 캐시 재사용을 막아야 함.
- `ignore_mismatch=False`가 기본 — 형식 불일치 시 TRT가 경고 후 캐시 무시. 안전함.

### 공수
소. `_build_engine`에 10~15줄 추가.

---

## H3. modelopt AutoTune / AutoQuantize — recipe #12의 진화

### 현재 상태
`_prepare_modelopt_onnx`는 `modelopt.onnx.quantization.quantize(...)`를 한 번 호출하고
사용자가 직접 준 `nodes_to_exclude` 리스트를 그대로 적용합니다. Recipe #12는 이 방식으로
4개 레이어(stem + cv2.\*)를 수동 지정했지만 Phase 3 측정에서 mAP 개선 0.02%p(노이즈)
+ bs=1 fps −25% 였음. 수동 exclude는 "어느 레이어가 실제로 민감한지"에 대한
근거 없이 경험칙에 의존.

### 근거: AutoTune (ONNX-level, 추론 엔진까지 측정)

> "Autotune is a specialized tool designed for automated Q/DQ placement
> optimization within ONNX models. By leveraging TensorRT latency
> measurements, it identifies optimal insertion schemes that
> significantly minimize inference time."
> — *modelopt docs/source/guides/9_autotune.rst*

CLI 예:
```bash
python -m modelopt.onnx.quantization.autotune \
    --onnx_path model.onnx \
    --output_dir ./results \
    --quant_type int8 \
    --schemes_per_region 50
```

Python API로 직접 호출도 가능하며 빌드마다 TRT engine을 내부에서 만들어 latency를
측정하고 최적 Q/DQ 스킴을 리전별로 선택합니다.

### 근거: AutoQuantize (torch-level, sensitivity 기반 mixed precision)

> "AutoQuantize intelligently applies more aggressive quantization
> formats ... to less sensitive layers and less aggressive ones to
> highly sensitive layers, or even skips quantization for extremely
> sensitive layers. This approach results in a better accuracy compared
> to models quantized uniformly ..."
> — *modelopt docs/source/guides/_pytorch_quantization.rst*

두 기법 모두 Recipe #12의 "수동 exclude" 문제를 체계적으로 해결. AutoTune은
**latency 최적화 초점**, AutoQuantize는 **accuracy 최적화 초점**.

### 권장 변경

#### 3a. 신규 recipe #13: `modelopt_int8_autotune`
```yaml
technique:
  source: modelopt
  name: int8_autotune
  autotune: default             # quick | default | extensive
  autotune_schemes_per_region: 20
```
구현: `_prepare_modelopt_onnx`가 `technique.autotune`이 켜지면 CLI 대신 Python API
호출(`modelopt.onnx.quantization.autotune.Autotuner`)로 연결.

#### 3b. 신규 recipe #14: `modelopt_int8_autoquantize` (선택)
torch-level이라 학습 데이터가 필요. v1.3 training 통합 시 같이 도입.

### 공수
중~대. AutoTune만 먼저 도입해도 큰 가치. YOLO26n + RTX 3060 기준 autotune 전체 돌리면
수 시간 (quick 모드는 수 십 분). 일회성 러닝.

---

## M1. TF32 플래그 — FP32 baseline 가속 (Ampere+)

### 현재 상태
`_build_engine`에서 FP32 경로(dtype 미지정)는 아무 builder flag도 안 세움. Ampere+
GPU에선 TF32가 **자동으로 활성화되지 않음** (TRT 10에선 기본 off, `kTF32` 명시 필요).

### 근거

> `BuilderFlag::kTF32` — "Enable TF32 precision."
> — *TensorRT C API namespace nvinfer1*

> "Set the FP16 builder flag to allow TensorRT to select lower-precision
> implementations. ... Use kTF32 similarly for TF32 tensor cores."
> — *TensorRT Precision Control*

TF32는 FP32와 동일한 dynamic range + 10-bit mantissa (vs FP32 23-bit) — 정확도 영향 거의
없이 matmul 최대 4x 빠름 (Ampere).

### 권장 변경
```python
if dtype == "fp32":
    config.set_flag(trt.BuilderFlag.TF32)
```

예상 이득: YOLO26n p50 **−5%~−10%** (Conv heavy 모델은 이득 작음, SPP/attention 많으면 큼).
정확도 변화 거의 0. recipe #5 (trt_fp16) 대비 비교 baseline으로서의 **의미가 더 명확**해짐.

### 공수
소. 한 줄.

---

## M2. `kOBEY_PRECISION_CONSTRAINTS` — nodes_to_exclude 강제

### 현재 상태
`_prepare_modelopt_onnx`는 `nodes_to_exclude`를 modelopt quantize에 전달하지만 TRT
engine build 시점에 이 제약이 **강제되지 않습니다**. TRT는 여전히 "더 빠른 INT8 tactic이
있으면 그걸 쓸" 자유가 있어, exclude 의도가 부분적으로만 반영됨.

### 근거

> `kOBEY_PRECISION_CONSTRAINTS` — "Configure the builder to strictly obey
> precision constraints, issuing an error if no implementation matches
> the requested precision."
>
> `kPREFER_PRECISION_CONSTRAINTS` — weaker variant (emit warning, fall
> back to best tactic).
> — *TensorRT Precision Control*

### 권장 변경
Recipe에 `nodes_to_exclude`가 있거나 `technique.obey_precision=true`가 명시되면:

```python
if recipe.technique.nodes_to_exclude:
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
```

효과: FP16에 남기겠다고 한 레이어가 실제로 FP16으로 빌드됨을 보장. mAP drop 감소 기대.

### 주의
너무 엄격하면 build fail 날 수 있음. `PREFER_*` 쪽이 더 안전한 기본값일 수도. 두
옵션 모두 노출하고 default는 `OBEY_*`.

### 공수
소.

---

## M3. CUDA Event 기반 GPU time 측정

### 현재 상태
`measure.py::measure_latency`는 wall-clock (`time.perf_counter()`) + `torch.cuda.synchronize()`:

```python
_cuda_sync()
t0 = time.perf_counter()
forward_fn()
_cuda_sync()
samples.append((time.perf_counter() - t0) * 1000.0)
```

CPU wall clock에는 Python/torch launch overhead가 포함됨. 순수 GPU 실행 시간을 분리하려면
CUDA event가 정석.

### 근거

TensorRT 공식 벤치마킹 가이드 (trtexec)는 CUDA event 기반 측정을 사용.

### 권장 변경 (streaming runner 한정)
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
forward_fn()  # TRT context.execute_async_v3 on its own stream
end.record()
end.synchronize()
ms = start.elapsed_time(end)
```

CPU launch overhead가 제외된 "순수 GPU time"이 나옴. 현재 wall-clock p50보다 통상
**0.3–0.8 ms 더 낮게** 나옴. H1(CUDA graph)과 결합하면 더 중요.

### 설계 고민
MLPerf/paper-replication 관점에서는 end-to-end wall-clock이 **사용자 경험 수치**. 둘
다 기록하고 report에 병기하는 게 정답:
- `latency_ms.p50_wall` (현재 수치, 유지)
- `latency_ms.p50_gpu` (신규)

`_schemas.Result.latency_ms`를 확장 필요.

### 공수
중. schema + measure + recommend.py 순위 기준 결정(어느 걸로 정렬할지).

---

## M4. Polygraphy `--validate` + `--debug-precision` 보조 타겟

### 현재 상태
정확도 이슈(recipe #11 real 2:4의 mAP 붕괴 같은 경우) 디버깅 시 "왜 망가졌는지" 분석할
계통적 도구가 없음. 현재는 ad-hoc 스크립트로 weight inspection.

### 근거

> "An experimental Polygraphy tool can assist in automatically
> identifying layers that should be run with higher precision to
> improve accuracy."
>
> ```bash
> polygraphy run <model.onnx> --debug-precision
> ```
>
> "Use the `--validate` option with Polygraphy to check for NaNs and
> Infs in layer outputs."
> — *TensorRT Performance Best Practices*

### 권장 변경
Makefile에 진단 타겟:
```make
.PHONY: diagnose-recipe-%
diagnose-recipe-%:
    polygraphy run results/_onnx/$(shell ls results/_onnx | grep $*.onnx | head -1) \
        --validate --log-file results/_diag/$*.validate.log
    polygraphy run ... --debug-precision --log-file results/_diag/$*.debug_precision.log
```

`make diagnose-recipe-12` → `results/_diag/12_*.log` 에 layer별 NaN/Inf 및 민감도
자동 탐지 결과.

### 공수
소. 보조 도구이므로 우선순위 낮음.

---

## L1. Random-normal calibration fallback 제거

### 현재 상태 (의도적 보수)
`_make_random_calibrator` + `_build_calib_numpy`의 `rng_np.standard_normal(...)` 분기
— `OMNI_COCO_YAML`이 없을 때 random 텐서로 fallback. 경고는 찍지만 실행은 계속.

주석에도 "Known to produce large mAP drops"라고 명시돼 있음.

### 문제
Phase 3에서 recipe-11의 mAP 붕괴를 디버깅할 때 첫 의심 지점 중 하나였음. 사일런트한
정확도 훼손 경로를 남겨두면 앞으로도 비슷한 시간 낭비.

### 권장 변경
`OMNI_COCO_YAML` 미설정 또는 파일 없음 → `run_trt.py`가 초기에 **명확한 에러로 종료**.
`--allow-random-calib` 같은 명시적 flag를 주어야만 fallback 허용.

### 공수
최소. 2~3줄 + Makefile에 `ALLOW_RANDOM_CALIB ?=` 등 추가.

---

## L2. Peak memory 측정 — torch caching allocator vs TRT allocator

### 현재 상태
`measure.py::_read_peak_mem_mb`는 `torch.cuda.max_memory_allocated()` 우선 + pynvml
fallback. TRT는 자체 allocator를 쓰므로 torch 측정치는 **backbone의 runtime 버퍼만**
잡히고 engine binding 버퍼는 누락 가능.

현 측정치 (38 MB)는 recipe 전반적으로 동일한데, YOLO26n INT8 engine 자체가 ~10 MB,
IO tensor가 ~6 MB/batch — 대략 맞지만 정확한 경계가 불분명.

### 권장 변경
- torch max_memory는 참고 지표로 유지
- NVML 측정은 "프로세스 전체 GPU 사용 메모리" 스냅샷(시작 전 vs 측정 중 delta)으로 교체
- Result schema:
  - `peak_gpu_mem_mb_torch`
  - `peak_gpu_mem_mb_nvml_delta`

### 공수
소~중. 스냅샷 delta 계산 로직 추가 + schema 확장.

---

## L3. IInt8EntropyCalibrator2는 deprecated 표식

### 현재 상태
`_make_coco_calibrator` / `_make_random_calibrator`는 TRT의 `IInt8EntropyCalibrator2`
C API를 상속하는 Python 클래스. Recipe #6 (trt_builtin) 전용.

### 근거

> "Version of calibration algorithm to use. **Deprecated in TensorRT
> 10.1, superseded by explicit quantization.**"
> — *TensorRT C API CalibrationAlgoType*

즉 NVIDIA는 "implicit quantization (Calibrator)" 대신 "explicit quantization (QDQ-ONNX)"를
권장. 우리 modelopt 경로(#8-#12)가 이미 explicit quantization임.

### 권장 변경
코드 변경 없음. 문서(README, CLAUDE.md)에 한 줄 주석:
> recipe #6 / #7은 TRT의 implicit INT8 경로 (IInt8EntropyCalibrator2)로,
> TRT 10.1에서 deprecated. Long-term 권장은 modelopt 경로 (#8-#12)의 QDQ
> explicit quantization. #6/#7은 레거시 비교 baseline으로만 유지.

### 공수
최소. 문서 한두 줄.

---

## 권장 실행 순서

1. **L1** (random-calib 에러 처리) — 10분, 버그 회피 즉효
2. **H2** (timing cache) — 30분, 개발 반복 속도 즉효
3. **M1** (TF32) — 5분 + 재측정
4. **M2** (OBEY_PRECISION_CONSTRAINTS) — 10분 + 재측정 (#12 mAP 개선 기대)
5. **H1** (CUDA graph) — 반나절, bs=1 latency 최대 이득
6. **M3** (CUDA event) — Result schema 변경 포함이라 브랜치로 분리
7. **L2, L3** — 문서/측정 정밀도 hygiene
8. **H3** (AutoTune) — 새 recipe #13 + modelopt 통합 작업, 별도 에픽
9. **M4** (polygraphy 진단 타겟) — 필요할 때 추가

---

## 바로 반영 안 하는 항목 (범위 밖 / 의도적 보류)

- **FP8 / NVFP4 / INT4**: RTX 3060 Laptop (SM 8.6)에서 지원 안 함. Hopper (SM 9.0+) 이상만.
- **WeightStreaming / DLA**: YOLO26n은 너무 작음.
- **Hardware compatibility level**: 크로스 GPU 빌드는 v2+ scope (CLAUDE.md와 일치).
- **kSTRIP_PLAN / kREFIT**: 사이즈 최적화 도구. 배포 파이프라인에서 필요 시 도입.
- **QAT / SAT**: 학습 코드 범위 밖.
