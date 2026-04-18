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

---

## v2 전문가 재검토 (2026-04-18, 같은 날 두 번째 패스)

첫 번째 패스 작성 후 숫자와 가정을 비판적으로 다시 봤습니다. 다음이 수정 사항.

### 수정 A — H1 (CUDA graph) 효과 과대 주장

**원본 주장:** "bs=1 p50 −8~−15%"
**근거 재점검:** YOLO26n p50 = 3.6 ms. Python + TRT launch overhead는 최신 드라이버에서
통상 0.1–0.3 ms/call. 15% (= 0.54 ms) 단축은 가능하지만 **상한**이고 평균은 1–4%
선이 현실적. nano 모델일수록 kernel launch 비율이 높지 않아 이득이 작음.

**수정:** H1의 기대 이득을 **bs=1 p50 −3~−8% (경험 상한 −12%)**로 하향. 여전히 가치 있으나
"🔥 매우 큼"은 과장. **"🔥 큼"** 수준으로 재분류.

**추가 정정 — 구현 경로:**
원본은 `cuda-python` 바인딩 추가를 제안. 더 나은 경로: **`torch.cuda.CUDAGraph`** (PyTorch
1.10+ 내장 API). 이미 의존성에 있음. `torch` 스트림과 TRT execution context가 같은 스트림을
쓰는 현 코드와 호환.

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=stream):
    context.execute_async_v3(stream.cuda_stream)
# 이후
def fwd():
    g.replay()
    stream.synchronize()
```

cuda-python 의존 제거 + 더 단순.

### 수정 B — H3 (AutoTune / AutoQuantize) 오클레임

**원본 주장:** "🔥 큼", recipe #12를 "진화"
**사실:**
1. **AutoTune은 latency 튜너**. "mAP drop을 줄이는" 도구가 아님. recipe #12의 목표(1%p mAP
   아래)는 AutoTune으로 달성할 수 없음. AutoTune은 "정확도는 기존 QDQ 그대로, latency만
   더 빠르게" 상황에 맞음. 우리 현재 수치가 recipe-09 기준 이미 fps 403 (bs=1) → 더 빨라져봤자
   한계 효용.
2. **AutoTune 실행 비용**: schemes_per_region × regions 회 TRT INT8 엔진 빌드. RTX 3060 Laptop
   YOLO26n INT8 빌드는 ~90–120s. quick 모드라도 수십 분, default 수 시간, extensive 반나절+.
3. **AutoQuantize는 training-dependent**. forward + loss + 레이블 필요. 현재 사용자 지침
   ("학습 코드 추가 전까지 ...")과 충돌. v1.3 training 통합 시만 유효.

**수정:**
- **H3 재분류: "AutoTune은 M급, AutoQuantize는 v1.3으로 제외"**
- AutoTune은 별도 실험 타겟으로만 도입 (`make autotune-recipe-09` 처럼 수동 실행). `make all`
  포함 금지.
- AutoQuantize는 학습 코드 합류 후 재평가.

### 수정 C — M2 (`kOBEY_PRECISION_CONSTRAINTS`) 적용 대상 오해

**원본 주장:** "`nodes_to_exclude`를 TRT가 실제 강제"
**사실:** modelopt의 `nodes_to_exclude`는 QDQ 주입 단계에서 **해당 노드에 Q/DQ를 안 넣는**
방식으로 동작. 결과 QDQ-ONNX에는 그 노드 주변에 QDQ가 없음. TRT는 QDQ 없는 영역을 **FP16/FP32
중 더 빠른 것**으로 자유롭게 선택. `kOBEY_PRECISION_CONSTRAINTS`는 **`network->setPrecision()`
API 호출에 대해 적용**, QDQ-ONNX 모델에는 직접 영향 제한적.

**수정:** M2는 효과 불확실로 **"요검증" 태그**. Recipe #12 mAP 개선 기대는 근거 약함. 대신:
- `modelopt.onnx.autocast.convert_to_mixed_precision(..., low_precision_type="fp16",
  nodes_to_exclude=...)`을 QDQ 주입 **이후**에 추가 실행해서 비-QDQ 노드를 명시적으로 FP16
  초기화자로 바꾸는 경로가 더 정확.
- 또는 build 시점에 **layer-level `setPrecision` API를 직접 호출**. ONNX 파싱 후 network를
  walk해서 제외 대상 레이어에 precision 강제. 구현 복잡도↑.

**권고:** M2는 초기 구현 대상에서 제외. 필요 시 별도 실험으로 분리.

### 수정 D — M1 (TF32) 기대 이득 정정 + 확인 필요

**원본 주장:** "p50 −5~−10%"
**사실:** YOLO는 Conv 위주에 3×3 kernel + 작은 채널 (nano). TF32는 GEMM과 큰 Conv에 이득이 큼.
nano 모델에선 **1–4%** 가 현실적. 또한 TRT 10에서 **TF32는 Ampere+에서 기본 on**일 가능성 (공식
문서 명확하지 않음). 먼저 현재 recipe #1(PyTorch fp32 baseline이 아닌 **TRT FP32가 현재 없음**)을
만들어서 flag on/off로 실측해야 근거가 생김.

**수정:** M1을 "TRT FP32 baseline 레시피 신규 추가 + TF32 flag 토글 실험"으로 구조화. 숫자 기대는
잠정 1–4%로 하향.

### 누락 발견 항목 (추가)

#### 신규 N1. Calibration 전처리 RGB/BGR 확인 (**검증 완료: 일치**)

**실측 (2026-04-18):** COCO val image 한 장을 `_letterbox`와 ultralytics
`LetterBox` + BGR→RGB + /255 양쪽에 돌려 비교 → `np.allclose=True`,
`max_abs_diff=0.0`. 즉 현 calibration 전처리는 ultralytics ONNX export가 기대하는
입력과 정확히 일치. **수정 불필요, 우려 해소.**

(원 N1 분석은 아래 유지.)



`_build_calib_numpy` → `_letterbox`는:
```python
rgb_chw = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
```
즉 BGR (cv2.imread) → RGB 로 반전. 반면 ultralytics YOLO의 ONNX export는 내부적으로 어떤 채널
순서를 기대하는지 **확인 안 됨**. 만약 mismatch면 calibration scale이 편향되어 mAP drop
원인이 됨. **검증 필요**:
```python
# ultralytics YOLO 기본 preprocessor 확인
from ultralytics.data.augment import LetterBox
# 또는 val 루프에서 실제 입력 텐서 추출해서 비교
```
공수 최소. 중요도 중 (mAP -0.5~-1%p 의심).

#### 신규 N2. Engine cache 버전 서명

`results/_engines/*.engine`은 파일명에 model hash는 있어도 **CUDA/TRT/driver 버전**이 없음.
사용자가 TRT 10.16 → 10.17 업그레이드하면 캐시된 엔진이 호환 안 될 수 있는데 파이프라인이
그대로 reuse. 실패 시 notes에 에러 남기고 `meets_constraints=False`가 되지만 사일런트 mAP 퇴행
리스크.

**수정:** engine 파일명에 `_trt{major}_{minor}` suffix 추가 또는 `env.tensorrt`를 포함한 hash 값
prefix. 공수 소.

#### 신규 N3. bs=8 엔진은 mAP 측정 안 됨 (문서화 필요)

현재 `eval_coco.py`는 `bs1_engine`만 사용. 만약 TRT tactic 선택이 bs=1과 bs=8에서 다르게
numerics가 발생하면 (드문 케이스지만 가능) bs=8 fps 리포트는 정확도 보장이 없음. README에
이 점을 명시해야 함. 코드 변경 아님.

#### 신규 N4. mAP recomputation on cache hit

`_export_onnx`와 `_build_engine`이 모두 cache-hit path를 가짐. cache hit이면 mAP eval 단계까지
스킵 안 하고 **매번 실행**. 빠른 반복 시 time-saver지만, 이미 ran 되었는지 불명확한 결과 JSON
덮어쓰기 위험. 결과 JSON에도 `env.trt/cuda` 버전 도장을 찍고, `recommend.py`가 stale detection.

### 정리된 수정 우선순위표

| # | 항목 | 영향 재평가 | 공수 | 코멘트 |
|---|---|---|---|---|
| H2 | Timing cache | 🔥 큼 (확실) | 소 | **제일 먼저 해도 되는 것** |
| L1 | Random calib fallback → error | 중 (버그 방지) | 최소 | 바로 반영 |
| N2 | Engine cache version signing | 중 (장기 안전성) | 소 | 같이 반영 |
| N1 | Calibration RGB/BGR 검증 | 중 (정확도) | 최소 | 3줄 스크립트 |
| H1 | CUDA graph (torch.cuda.CUDAGraph) | 중~큼 (상한 8-12%) | 중 | torch API 사용 |
| M3 | CUDA Event 측정 | 중 | 중 | schema 변경 포함, 별도 브랜치 |
| M1 | TF32 실험 (+ FP32 TRT baseline 레시피) | 소~중 | 소 | 측정으로 이득 확인 |
| M4 | polygraphy 진단 타겟 | 하 | 소 | 필요할 때 |
| L3 | deprecated 주석 | 최하 | 최소 | 한 줄 |
| L2 | Peak memory 이중 측정 | 하 | 소 | schema 확장 |
| M2 | `OBEY_PRECISION_CONSTRAINTS` | ~~중~~ → **요검증** | 중 | QDQ 경로에선 효과 불명 |
| H3 | AutoTune | 중 (실험용, 일회성) | 대 | `make all`에서 분리, 별도 |
| — | AutoQuantize | ~~중~~ → **v1.3** | 대 | 학습 필요 |

**결론:** 원본 순서 1–2번(L1, H2)은 유지, 그 다음은 N1/N2 우선, H1·M3는 큰 브랜치, H3는 별도 에픽.
M2와 AutoQuantize는 이번 라운드에서 뺌.

---

## v3 실행 플랜 — 병렬 분석 + 웨이브 (재정렬 반영)

v2 재검토를 기반으로 **파일/함수 단위 충돌 맵**을 그리고 실제로 병렬 가능한 작업을 확정.

### 파일/함수 충돌 매트릭스

| 대상 | 수정 작업 (Task ID) | 병렬 가능? |
|---|---|---|
| `run_trt.py::_build_engine` | H2 (#9), L1 (#10 일부), TF32 (#15) | ❌ 직렬 |
| `run_trt.py::_prepare_modelopt_onnx` | L1 (#10 일부) | (자연스레 #10 세트) |
| `run_trt.py::run()` (engine_tag) | N2 (#11) | ⚠ 같은 파일, 다른 함수 |
| `run_trt.py::_make_trt_forward` | H1 (#13) | ✅ 격리 함수 |
| `_schemas.py::LatencyStats` | M3 (#14) | ✅ 독립 |
| `measure.py::measure_latency` | M3 (#14), L2 peak mem (#16) | ❌ 묶어야 함 |
| `recommend.py` | M3 (#14) | (#14 세트) |
| `Makefile` | M4 diagnose (#16), AutoTune (#17) | ⚠ 같은 파일 |
| `README.md` | L3 deprecated (#16) | ✅ 독립 |
| `scripts/autotune.py` (신규) | AutoTune (#17) | ✅ 신규 |
| `recipes/00_trt_fp32.yaml` (신규) | TF32 (#15) | ✅ 신규 |

**핵심 관측:**
- `_build_engine`를 건드리는 #9/#10/#15는 직렬 필수. 마찰 피하려면 같은 PR로 묶거나 순서대로.
- `_make_trt_forward`는 다른 함수라 병렬 에이전트로 분리 가능.
- #14(schema) + #16(measure.py peak mem)은 같은 `measure.py`를 건드리므로 **#16-부분을 #14에 흡수**하는 게 깔끔.
- #12 RGB/BGR은 단발 진단 (수정 없을 수도), 어디서든 가능.
- #17 AutoTune은 신규 파일이라 완전 독립.

### 웨이브 플랜 (재정렬)

#### **Wave 1 — 고확률·저공수 배치 (~1–1.5시간, 두 트랙 병렬)**

**Track A (메인 스레드, `run_trt.py` 직렬):**
1. **#12 RGB/BGR 확인** (5분, 진단) — 결과에 따라 #10 validation 추가 여부 결정
2. **#11 Engine cache version suffix** (10분) — `engine_tag` 한 줄
3. **#9 Timing cache reuse** (30분) — `_build_engine` 상단에 캐시 로드/저장
4. **#10 Random calib fallback → error** (10분) — `_prepare_modelopt_onnx` + `_build_engine` 경로 검증

모두 `run_trt.py` 중심이므로 직렬이 자연스러움. 각 단계마다 `pytest tests/` + `make recipe-05`
스모크 체크.

**Track B (서브에이전트, 격리 함수):**
- **#13 CUDA graph via `torch.cuda.CUDAGraph`** (45–60분) — `_make_trt_forward`만 수정
- 입력: 문서 H1 섹션 + v2 수정. torch.cuda.CUDAGraph API 사용. capture 실패 시 기존 경로로 fallback.
- 검증: 재구축된 bs=1 엔진으로 recipe-05 실측 → 이전 대비 −3~−8% 달성 확인.

Track A와 B는 **같은 run_trt.py**를 건드리지만 **다른 함수** → 충돌 발생 시 rebase 용이.
Track B 먼저 완료되어 병합 → Track A는 최신 HEAD 위에서 작업. 또는 반대.

#### **Wave 2 — 측정 인프라 확장 (~2–3시간, 두 트랙 병렬)**

**Track C (메인 or 에이전트):**
- **#14 + #16(부분) CUDA Event + Peak mem 이중 측정 통합**
- 수정: `_schemas.LatencyStats`에 `p50_gpu` 추가, `measure.py::measure_latency`에 CUDA event 경로 +
  NVML delta peak mem, `recommend.py` 컬럼 한 개 추가, 기존 10개 Result JSON 재측정 or lazy null.
- 리스크: schema 변경이라 기존 JSON 로드 시 `p50_gpu=None` 허용해야 함 (Optional[float]).

**Track D (메인 스레드, 병렬):**
- **#15 TRT FP32 baseline + TF32 toggle** (30분 + 실측)
  - `recipes/00_trt_fp32.yaml` (TF32 off) + `recipes/00_trt_fp32_tf32.yaml` (TF32 on)
  - `_build_engine`에 FP32/TF32 처리 추가 (1줄)
  - `make recipe-00`, `make recipe-00-tf32` 실측 → 실제 이득 1–4% 근거 확보
- **#16(나머지) L3 deprecated note + M4 polygraphy diagnose target** — README/Makefile 작은 문구

Track C와 D는 파일 충돌 없음 (C는 `_schemas/measure/recommend`, D는 `recipes/_build_engine/README/Makefile`).

#### **Wave 3 — 별도 실험 에픽 (독립 브랜치, 시간 되는 날)**

**#17 AutoTune experimental target**
- `scripts/autotune.py` 신규 + `Makefile`에 `autotune-recipe-%` 타겟
- `make all`에서 분리 (수 시간 소요)
- 별도 브랜치 `epic/autotune`에서 진행
- 일회성 측정 → 결과는 `results/_autotune/`로

### 실행 타임라인 (이상적)

```
시간      00:00        01:30        04:00               훗날
         ┌────────────┬────────────┬──────────────────┬──────
Wave 1-A │ 12→11→9→10 │            │                  │
Wave 1-B │ 13 (agent) │            │                  │
         │            │─ merge ─   │                  │
Wave 2-C │            │ 14+16-부분 │                  │
Wave 2-D │            │ 15, 16-나머지│                 │
         │            │            │─ merge, publish ─│
Wave 3   │            │            │                  │ 17 AutoTune
         └────────────┴────────────┴──────────────────┴──────
```

총 ~4시간으로 H1,H2,L1,L2(부분),L3,M1,M3,M4,N1,N2 처리. H3(AutoTune)만 별도.

### 제외 재확인

- **M2 `OBEY_PRECISION_CONSTRAINTS`** — QDQ-ONNX 경로에선 효과 불명, 실험 가치 낮아 제외.
- **AutoQuantize** — `학습 코드` 부재로 v1.3 이후.
- **FP8/INT4/DLA/Cross-GPU** — v2+ scope (CLAUDE.md 일치).

### 리스크와 완화

| 리스크 | 완화 |
|---|---|
| Track A 중 `_build_engine` diff 충돌 | 순서대로 커밋, 각 커밋 후 `pytest` + smoke recipe 실행 |
| Track B의 CUDAGraph 캡처가 capture mode 제약으로 실패 | try/except fallback → 로그 경고 + 기존 경로 유지 |
| #14 schema 변경이 기존 JSON 로드 깨뜨림 | 새 필드는 `Optional[...] = None`로 추가, 기존 JSON load 영향 없음 |
| #17 AutoTune이 수 시간 잡아먹고 실패 | `make all`과 분리, 실패해도 main 파이프라인 영향 없음 |

---

## 바로 반영 안 하는 항목 (범위 밖 / 의도적 보류)

- **FP8 / NVFP4 / INT4**: RTX 3060 Laptop (SM 8.6)에서 지원 안 함. Hopper (SM 9.0+) 이상만.
- **WeightStreaming / DLA**: YOLO26n은 너무 작음.
- **Hardware compatibility level**: 크로스 GPU 빌드는 v2+ scope (CLAUDE.md와 일치).
- **kSTRIP_PLAN / kREFIT**: 사이즈 최적화 도구. 배포 파이프라인에서 필요 시 도입.
- **QAT / SAT**: 학습 코드 범위 밖.

---

## Wave 3 results — ONNX Runtime Quantization + Intel Neural Compressor (2026-04-18)

### 실험 목적
`trt_builtin`(−7.9%p) 및 `modelopt`(−1.6~1.9%p) 외에 두 개의 INT8 백엔드를
더 얹어 mAP-fps 공간을 넓혀본다. (1) ORT `quantize_static` (4종 calibrator),
(2) Intel Neural Compressor 2.6 (MinMax PTQ + SmoothQuant). QAT 스펙(#19)은
training 코드 부재로 parked.

### 측정치 (RTX 3060 Laptop, YOLO26n 640×640, bs=1)

| 레시피 | calibrator | mAP@0.5 | drop %p | fps(bs1) | meets 1%p? |
|---|---|---:|---:|---:|:---:|
| #13 ort_int8_minmax       | MinMax       | 0.494 | +5.94 | 536 | ✘ |
| #14 ort_int8_entropy      | Entropy      | 0.501 | +5.16 | 679 | ✘ |
| #15 ort_int8_percentile   | Percentile   | 0.528 | +2.50 | 647 | ✘ |
| #16 ort_int8_distribution | Distribution | 0.503 | +4.97 | 644 | ✘ |
| #17 inc_int8_ptq          | MinMax       |   —   |   —   |   — | ✘ (build fail) |
| #18 inc_int8_smoothquant  | SmoothQuant  |   —   |   —   |   — | ✘ (build fail) |
| (참조) #8 modelopt_int8_ptq  | max        | 0.521 | +3.19 | 400 | ✘ |
| (참조) #9 modelopt_int8_entropy | entropy | 0.537 | +1.64 | 409 | ✘ |
| (참조) #12 modelopt_int8_mixed | entropy+excludes | 0.537 | +1.64 | 419 | ✘ |

### 해석

**ORT 백엔드 — 예상 밖의 fps, 낮지만 측정 가능한 mAP**

- ORT 경로는 `quantize_static` 호출 전에 `quant_pre_process` (shape inference + folding)를 강제함. 결과 ONNX가 modelopt 경로 대비 얇아져 TRT 엔진이 더 효율적 — ORT 4종의 fps(bs=1)가 modelopt보다 30~70% 높게 나옴. 가속 자체는 양자화가 아닌 pre-processing 기여.
- **ORT percentile (#15)**은 Wave 3 최고 accuracy (drop +2.50%p). Entropy/Distribution (5%p대)보다 뚜렷이 낫고, MinMax(6%p)보다는 더 낫지만 modelopt entropy(1.64%p)에는 못 미침. 이유는 ORT percentile의 고정 99.999 threshold가 YOLO neck의 outlier 분포에 덜 robust.
- 공통적으로 Wave 3 recipes 중 `max_map_drop_pct=1.0` 충족 없음. modelopt도 마찬가지였으나 drop이 절반 수준 — "fast 하지만 덜 정확한" ORT vs "느리지만 더 정확한" modelopt 구도가 명확.

**INC 백엔드 — 본 조합에서 TRT 비호환 확정**

INC 2.6 + onnx 1.17 + TRT 10 조합을 YOLO26n에 적용하면 5개의 독립적 비호환이 연쇄 발생:

1. `onnx.mapping` module removed in onnx>=1.15 — INC 2.6 내부 참조 잔존
2. INC 기본 `QOperator` 포맷이 `com.microsoft.QLinearSigmoid`를 생성 — TRT 플러그인 미등록
3. Conv.bias를 INT32 QDQ로 폴딩 — TRT `DequantizeLayer`가 INT32 입력 거부
4. MinMax calibration이 `op_type_dict` scheme='sym' 요청에도 불구하고 activation output QDQ에 non-zero zero_point 산출 — TRT가 asymmetric QDQ 거부
5. SmoothQuant가 attention block의 weight shape를 바꾸지만 downstream `Reshape` 노드는 그대로 — TRT 빌드 시 `[8,2,32,400] → [8,128,20,20]` volume mismatch

1~4는 `scripts/run_trt.py`에 호환 레이어를 추가해 통과 시도했으나, 5는 INC가 SmoothQuant 시 attention subgraph를 재작성할 때 필요한 쌍 대응 graph rewrite를 수행하지 않아 해결 불가. INC 내부의 SmoothQuant + ONNX backend는 LLM/CNN 공통 경로로 검증돼있지만 **ViT-style attention + reshape** 워크로드에선 아직 beta 수준. 결국 #17/#18은 runner가 build 실패로 기록하고 report에 노출만 함 (`meets_constraints=False`).

### 결론 & follow-up

- 즉시 추천 변경 없음. `trt_fp16`이 여전히 유일한 제약 충족 레시피로 report 1위.
- Wave 3의 양자화 비교가 실제로 도움이 되는 것은 "더 빠른 INT8이 필요하지만 1%p drop은 이미 포기" 시나리오 — 이 때 `ort_int8_percentile` (fps 647, drop 2.5%p)이 modelopt_int8_mixed (fps 419, drop 1.6%p)를 대체할 후보.
- INC 경로는 **본 저장소에서 deprecated** 표식. 재시도는 (a) INC 3.x가 onnx-rewrite를 attention-aware로 고친 뒤 또는 (b) PyTorch-level INC SmoothQuant → torch.onnx.export QDQ 경로(modelopt의 torch path 구조와 동일)로 재구현.
- SmoothQuant 시도 자체는 계속 가치 있음: modelopt는 SmoothQuant를 CNN path에 공식 지원하지 않으므로, Phase 5에서 `modelopt.torch.quantization`의 fake-quant → SmoothQuant 어댑터 수기 구현을 고려.
