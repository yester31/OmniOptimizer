# Recipe 변환 프로세스 — 전체 파이프라인 설명

**Purpose**: 각 recipe 가 `yolo26n.pt` 에서 측정 가능한 런타임 아티팩트까지 어떻게 변환되는지 단계별로 설명. 디버깅 / 신규 recipe 추가 / 파이프라인 재현 시 참조용.

**Entry point**: `python scripts/run_{backend}.py --recipe recipes/{id}_{name}.yaml --out results*/…json`

## 공통 전처리 — 모든 backend 공유

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Recipe YAML → Recipe pydantic 객체 (scripts/_schemas.py::load_recipe)  │
│            ↓                                                            │
│  _split.calib_yaml() / eval_yaml() 로 calib/eval 이미지 분리            │
│            ↓                                                            │
│  _resolve_weights(recipe) — training 여부 판별                          │
│    ├─ training=None        → str path 그대로 반환                       │
│    ├─ prune_24 trained     → trained_weights/{name}.pt                  │
│    └─ modelopt_* trained   → YOLO instance (mto.restore() 적용)         │
└─────────────────────────────────────────────────────────────────────────┘
```

**핵심 모듈**:
- `scripts/_schemas.py` — Recipe / Result pydantic 계약
- `scripts/_weights_io.py::_resolve_weights` — 학습 산출물 선택
- `scripts/_weights_io.py::_export_onnx` — 모든 ONNX 경로의 진입점
- `scripts/_split.py` — COCO val 을 calib / eval 로 결정적 분할 (seed 42)
- `scripts/measure.py::measure_latency` — warmup + percentile 측정

## 1. PyTorch eager (#01 `pytorch_fp32`)

```
yolo26n.pt
    ↓ ultralytics.YOLO(weights)
    ↓ inner = m.model (nn.Module)
    ↓ inner.to(device=cuda, dtype=torch.float32)
    ↓ torch.inference_mode() 컨텍스트
    ↓ torch.randn((bs, 3, 640, 640)) 입력
    ↓ measure_latency (warmup 100 + measure 100 iter, CUDA events)
    ↓
결과: latency_ms + throughput_fps
```

**파일**: `scripts/run_pytorch.py`
- Line 56: `_make_forward` — inference_mode + 고정 입력 closure
- Line 83-102: cold_start 로딩 시간 측정
- Line 104-122: triton_key shim (torch 2.8 + triton 3.x 호환)

**특징**:
- ONNX 변환 없음. `.pt` 직접 로드.
- 엔진 컴파일 없음. eager execution.
- 참고: `torch.compile` (archived #02) 는 Windows 에서 MSVC 의존.

## 2. ORT TRT EP (#04 `ort_trt_fp16`)

```
yolo26n.pt
    ↓ _export_onnx(yolo, imgsz=640, half=False, dynamic=True)
    ↓ ONNX: results/_onnx/yolo26n_640_fp32_bs1.onnx
    ↓ _add_tensorrt_dll_dir() — tensorrt_libs/ 를 DLL 검색 경로에 추가
    ↓ onnxruntime.InferenceSession(
    │    providers=[
    │       ("TensorrtExecutionProvider", {
    │         "trt_engine_cache_enable": True,
    │         "trt_engine_cache_path": results/_trt_cache/,
    │         "trt_fp16_enable": True,
    │       }),
    │       "CUDAExecutionProvider",
    │       "CPUExecutionProvider",
    │    ])
    ↓ session guard: get_providers()[0] == TensorrtExecutionProvider 검증
    ↓ 첫 run() 시 TRT EP 가 내부적으로 engine 생성 → trt_cache 저장
    ↓ measure_latency
```

**파일**: `scripts/run_ort.py`
- Line 16-42: `_add_tensorrt_dll_dir` — `os.add_dll_directory()` (Wave 11 fix, DLL path blocker 해결)
- Line 77: `_export_onnx` — ultralytics YOLO.export(format='onnx') 래퍼
- Line 95-135: `_make_session` — provider 옵션 주입 + session 검증

**특징**:
- ORT 가 TRT 를 wrapper 로 호출. fps 는 native TRT 대비 ~43% 수준 (구조적 오버헤드).
- AMD / Intel GPU 에도 동일 인터페이스 (Wave 9 DirectML EP 참조).

## 3. TensorRT native — FP32 / FP16 / BF16 (#00, #00-tf32, #05, #40, #41)

```
yolo26n.pt
    ↓ _export_onnx(half=False, dynamic=False)  # bs-specific
    ↓ ONNX: yolo26n_640_fp32_bs1.onnx
    ↓
    ↓ [TRT build path: scripts/run_trt.py::_build_engine]
    ↓
    ↓ trt.Builder(logger) + create_network(EXPLICIT_BATCH)
    ↓ OnnxParser.parse()
    ↓ config = create_builder_config()
    ↓ config.set_memory_pool_limit(WORKSPACE, 4 GiB)
    ↓ config.set_timing_cache(cache)  # results/_trt_timing.cache
    ↓
    ↓ [dtype 분기]
    ├─ fp32 + tf32 mode → config.set_flag(BuilderFlag.TF32)
    ├─ fp32            → (default, 플래그 없음)
    ├─ fp16            → config.set_flag(BuilderFlag.FP16)
    └─ bf16            → config.set_flag(BuilderFlag.BF16)   [Wave 14 A2]
         └─ sparsity="2:4" 동시 지정 시 → NotImplementedError (guard)
    ↓
    ↓ [선택: Wave 14 A1] builder_optimization_level=5
    ↓   exhaustive autotune (build 3-5× 시간)
    ↓
    ↓ optimization_profile 고정 (bs, 3, 640, 640)
    ↓ builder.build_serialized_network(network, config)  # 타이밍 측정
    ↓ 엔진: results/_engines/{stem}_{dtype}{_tf32}{_opt5}_bs{N}_trt10.16_cuda12.9.engine
    ↓
    ↓ torch.cuda IExecutionContext (pycuda 미사용 — 공유 CUDA context)
    ↓ measure_latency (CUDA events)
    ↓
    ↓ ultralytics.YOLO(engine_path).val(data=eval_yaml) — mAP 계산
```

**파일**: `scripts/run_trt.py`
- Line 519: `_build_engine` — 전체 빌드 로직
- Line 583-600: dtype 분기 (fp32/fp16/bf16)
- Line 646-652: `builder_optimization_level` 주입 (Wave 14)
- Line 674-720: `_make_trt_forward` — torch.cuda 기반 enqueue_v3

**특징**:
- `_trt_timing.cache` 를 레시피 간 공유 → 중복 layer tactic 재사용.
- BF16 on sm_86: Myelin 이 일부 BF16 tactic skip (log warning), FP16 대비 fps 낮음.
- TF32: FP32 기본값 대비 ~1-3% 개선, 정확도 거의 무손실.

## 4. TensorRT INT8 built-in calibrator (#06 `trt_int8_ptq`, #07 `trt_int8_sparsity`)

```
yolo26n.pt
    ↓ _export_onnx(half=False)
    ↓ ONNX: yolo26n_640_fp32_bs1.onnx  (#3 경로와 동일한 FP32 ONNX)
    ↓
    ↓ _build_engine(dtype="int8", quant_preapplied=False)
    ↓   config.set_flag(BuilderFlag.INT8)
    ↓   calibrator = _make_coco_calibrator(
    │       shape=(bs, 3, 640, 640),
    │       n_samples=512,
    │       val_yaml=coco_val_only.yaml,
    │   )
    ↓   config.int8_calibrator = calibrator
    ↓   (선택) sparsity="2:4" → config.set_flag(SPARSE_WEIGHTS)
    ↓
    ↓ TRT 가 calibration images 를 forward 시켜 activation range 수집
    ↓ 최종 scale 결정 → engine 내 INT8 kernel tactic 선택
    ↓ builder.build_serialized_network()
    ↓ 엔진 저장
```

**파일**: `scripts/run_trt.py`
- Line 411: `_make_coco_calibrator` — EntropyCalibratorV2 서브클래스
- Line 473: `_make_random_calibrator` — 폴백 (mAP 크게 떨어짐, `OMNI_ALLOW_RANDOM_CALIB=1` 필요)
- Line 590-638: INT8 분기 로직

**특징**:
- TRT 내장 entropy calibrator — QDQ ONNX 를 만들지 않음. 빌드 시점에 scale 계산.
- Sparsity: `BuilderFlag.SPARSE_WEIGHTS` 는 **weight 에 2:4 pattern 이 이미 있어야** 실효. Post-training application 은 mAP 1.5%p drop (sparsity-aware training 필요).

## 5. TensorRT + modelopt ONNX-level INT8 (#08-#12, #17, #42)

```
yolo26n.pt
    ↓ _export_onnx(half=False)
    ↓ clean ONNX (FP32, no Q/DQ)
    ↓
    ↓ [scripts/run_trt.py::_prepare_modelopt_onnx]
    ↓
    ↓ calibrator = recipe.technique.calibrator
    ↓   (max | entropy | percentile | mixed | *_asymmetric | *_sparse24)
    ↓
    ↓ [suffix 해석] "entropy_asymmetric" → base="entropy" + use_zero_point=True  [Wave 14 A5]
    ↓ [suffix 해석] "_sparse24"           → sparsity_preprocess="2:4" 선행
    ↓
    ↓ calib_data = _build_calib_numpy(val_yaml, samples=512, seed=42)
    ↓   — (512, 3, 640, 640) float32 numpy 배열
    ↓
    ↓ modelopt.onnx.quantization.quantize(
    │    onnx_path=clean.onnx,
    │    quantize_mode="int8",
    │    calibration_method=base_calibrator,   # "entropy"
    │    calibration_data=calib_data,
    │    use_zero_point=use_zero_point,         # True = asymmetric
    │    output_path=qdq.onnx,
    │    high_precision_dtype="fp16",           # Detect head 잔류 FP16
    │ )
    ↓ QDQ ONNX: results/_onnx/{stem}_640_modelopt_{method}{_asym}{_sparse24}_bs1.onnx
    ↓   — Q/DQ pair 223개 주입 (activation scalar + weight per-channel)
    ↓
    ↓ _build_engine(dtype="int8", quant_preapplied=True)
    ↓   config.set_flag(BuilderFlag.INT8)
    ↓   config.set_flag(BuilderFlag.FP16)     # modelopt 의 mixed 기본
    ↓   int8_calibrator 미지정 (scale 이 ONNX 안에 있음)
    ↓
    ↓ 엔진 빌드 → 측정
```

**파일**: `scripts/run_trt.py`
- Line 122-205: `_prepare_modelopt_onnx` — 전체 modelopt 경로
- Line 146-157: suffix 해석 (asymmetric / sparse24)
- Line 157-175: sparsity_preprocess 우선 적용 → clean ONNX 에 0 패턴 주입
- Line 182-203: `moq_quantize()` 호출

**특징**:
- modelopt 0.43+ API: `use_zero_point=False` 가 symmetric 기본. `True` 로 asymmetric 전환 (Wave 14).
- Histogram-heavy calibrator (entropy/percentile) 는 RAM-hungry — sample 수를 128-512 범위에서 조정.
- `nodes_to_exclude` 로 민감 layer 는 FP16 으로 유지 가능.

## 6. TensorRT + modelopt FastNAS + INT8 (#23, #24)

**구조적 변화**: pruning 이 앞에 추가됨.

```
yolo26n.pt (사전 학습된 FastNAS-prune 가중치)
    ↓ trained_weights/{recipe.name}.pt 로드 (scripts/train.py 가 생성)
    ↓
    ↓ FastNAS 후보 서브그래프가 이미 선택된 상태
    ↓ FLOPs −15.7% (FX-trace 가능 layer 한정)
    ↓
    ↓ [선택: #24] modelopt.torch.sparsity 2:4 FT + mask-preservation callback
    ↓
    ↓ _export_onnx(yolo_with_pruned_weights, half=False)
    ↓ clean ONNX (작음 — 5MB vs baseline 38MB)
    ↓
    ↓ [경로 4와 동일] modelopt.onnx.quantize(entropy)
    ↓ QDQ ONNX: Q/DQ 223 쌍 (그래프 구조는 보존, 채널 수만 줄어듦)
    ↓
    ↓ _build_engine(dtype="int8", quant_preapplied=True)
    ↓ 엔진: 5 MB (−88%)
```

**파일**:
- `scripts/train.py` — FastNAS 사전 학습 엔트리
- `scripts/run_trt.py::_resolve_weights` — trained_weights/ 경로 분기
- `scripts/_modifiers/modelopt_fastnas.py` — prune + FT 로직

**특징**:
- FX-trace 불가능한 YOLO 동적 branch 는 prune 대상 제외 → 감소율 15.7% 상한.
- mAP −4%p trade-off → 엣지/VRAM 제약 시에만 이득.

## 7. TensorRT + ORT QDQ (#13-#16)

modelopt 대신 onnxruntime.quantization 이 QDQ ONNX 를 만듦.

```
yolo26n.pt
    ↓ _export_onnx(half=False) → clean ONNX
    ↓
    ↓ [scripts/run_trt.py::_prepare_ort_quant_onnx]
    ↓
    ↓ onnxruntime.quantization.shape_inference.quant_pre_process(
    │    auto_merge=True,  # YOLO attention block 용 dynamic reshape 병합
    │ )
    ↓ preproc ONNX (shape 추론 완료)
    ↓
    ↓ calib_data: _NumpyReader(CalibrationDataReader) 스트리밍
    ↓   — 128 samples (histogram 메모리 제약)
    ↓
    ↓ onnxruntime.quantization.quantize_static(
    │    calibrate_method=MinMax | Entropy | Percentile | Distribution,
    │    quant_format=QDQ,
    │    activation_type=QInt8,
    │    weight_type=QInt8,
    │    per_channel=True,
    │    extra_options={
    │       "ActivationSymmetric": True,   # TRT 요구사항
    │       "WeightSymmetric": True,
    │    },
    │ )
    ↓ QDQ ONNX
    ↓
    ↓ _build_engine(dtype="int8", quant_preapplied=True)  # [경로 5 와 동일]
    ↓ 엔진 빌드 → 측정
```

**파일**: `scripts/run_trt.py`
- Line 208: `_prepare_ort_quant_onnx` — 전체 ORT quant 경로
- Line 247-261: histogram sample cap (128)
- Line 284-301: `quant_pre_process` 전처리
- Line 334-346: `quantize_static` 호출

**특징**:
- MinMax 는 메모리 효율적 (512 sample OK). Entropy/Percentile/Distribution 은 128 이 상한.
- TRT 는 symmetric activation/weight 강제 — `ActivationSymmetric=True` 필수.

## 8. ORT CPU FP32 (#30, #38)

```
yolo26n.pt
    ↓ _export_onnx(half=False, dynamic=True)
    ↓ ONNX: results_cpu/_onnx/yolo26n_640_fp32_dyn.onnx
    ↓
    ↓ ort.SessionOptions()
    ↓   graph_optimization_level = ORT_ENABLE_EXTENDED
    ↓   intra_op_num_threads = 4  (physical cores)
    ↓   inter_op_num_threads = 1
    ↓
    ↓ ort.InferenceSession(providers=["CPUExecutionProvider"])
    ↓ measure_latency (perf_counter + iter_cooldown_ms 옵션)
```

**파일**: `scripts/run_cpu.py`
- Line 123-131: `_build_ort_session_options`
- Line 134: `_prepare_ort_cpu_fp32`
- Line 70-116: `_resolve_thread_count` — psutil → /proc/cpuinfo → `os.cpu_count()//2` 폴백

## 9. ORT CPU INT8 dynamic (#32)

```
yolo26n.pt
    ↓ FP32 ONNX (위와 동일)
    ↓
    ↓ onnxruntime.quantization.quantize_dynamic(
    │    weight_type=QUInt8,
    │    per_channel=False,
    │    reduce_range=False,
    │ )
    ↓ INT8 ONNX (weight-only, activation 은 runtime 에 동적 quantize)
    ↓
    ↓ InferenceSession(CPUExecutionProvider)
    ↓ measure_latency
```

**파일**: `scripts/run_cpu.py::_prepare_ort_cpu_int8_dynamic` (Line 236)

**특징**: Per-channel + QInt8 조합은 MLAS ConvInteger 가 미구현 → 반드시 `per_channel=False` + `QUInt8`. Calibration 불필요.

## 10. ORT CPU INT8 static (#33) — Wave 11 fix

```
yolo26n.pt
    ↓ FP32 ONNX
    ↓
    ↓ quant_pre_process(auto_merge=True)
    ↓ preproc ONNX
    ↓
    ↓ [Wave 11 Task 5] nodes_to_exclude 패턴 확장
    ↓   recipe YAML: nodes_to_exclude: [/model.23/]
    ↓   → preproc ONNX 에서 /model.23/* prefix 매칭 183 nodes 열거
    ↓
    ↓ calib_reader = _NumpyReader(streaming 128 samples)
    ↓
    ↓ onnxruntime.quantization.quantize_static(
    │    calibrate_method=Entropy,
    │    nodes_to_exclude=expanded_excludes,  # <-- Detect head 전체 제외
    │    quant_format=QDQ,
    │    activation_type=QInt8, weight_type=QInt8,
    │    per_channel=True,
    │    extra_options={
    │       "ActivationSymmetric": True,
    │       "WeightSymmetric": True,
    │       "AddQDQPairToWeight": True,
    │       "DedicatedQDQPair": False,   # CPU MLAS fusion
    │    },
    │ )
    ↓ INT8 QDQ ONNX (Detect head 는 FP32 그대로)
    ↓
    ↓ InferenceSession(CPUExecutionProvider)
    ↓ measure_latency
```

**파일**: `scripts/run_cpu.py::_prepare_ort_cpu_int8_static` (Line 284-421)

**Wave 11 교훈**: Detect head Q/D/Q 가 동일 scale 을 TopK/GatherElements/Mod 등 index 텐서에도 적용하면 mAP 0. `/model.23/*` 전부 제외로 mAP 0.000 → 0.983.

## 11. ORT CPU BF16 (#31) — HW-gated

```
yolo26n.pt
    ↓ _collect_cpu_info() → cpu_flags 수집
    ↓
    ↓ [HW gate]
    ↓ if not (amx_tile OR avx512_bf16 in flags):
    ↓    raise NotImplementedError("host CPU lacks BF16 ISA")
    ↓    → Result.notes 기록, meets_constraints=False
    ↓
    ↓ [아직 미구현 경로]
    ↓ raise NotImplementedError("BF16 inference on ORT CPU not implemented")
```

**파일**: `scripts/run_cpu.py::_prepare_cpu_session` Line 636-658

**현재 상태**: Tiger Lake i7-11375H 는 AVX512_BF16 / AMX 없음 → HW gate 에서 skip. Sapphire Rapids+ 확보 시 BF16 path 구현 + 측정 가능.

## 12. OpenVINO FP32 (#34, #37)

```
yolo26n.pt
    ↓ _export_onnx(half=False, dynamic=True)
    ↓ ONNX
    ↓
    ↓ openvino.Core().read_model(onnx_path)
    ↓ ov_model
    ↓ openvino.save_model(ov_model, ir_xml)
    ↓ IR: results_cpu/_ov_ir/{recipe}_fp32.xml + .bin
    ↓
    ↓ compiled = core.compile_model(
    │    ov_model, "CPU",
    │    config={
    │       "PERFORMANCE_HINT": "LATENCY" if bs==1 else "THROUGHPUT",
    │       "INFERENCE_NUM_THREADS": "4",
    │    },
    │ )
    ↓
    ↓ OVRunnerAsORT(compiled, input_name, output_names)
    ↓   — openvino.CompiledModel 을 ort.InferenceSession 인터페이스로 감싸기
    ↓ measure_latency (공통 runner 사용)
```

**파일**: `scripts/run_cpu.py`
- Line 427-466: `OVRunnerAsORT` — adapter 패턴
- Line 481-511: `_compile_openvino` — batch-size aware hint
- Line 511: `_prepare_openvino_fp32`

**특징**:
- IR 캐시: 2회차부터 `read_model` 에 xml 직접 읽어 ONNX 재파싱 skip.
- `INFERENCE_NUM_THREADS` 를 명시해 hyperthread 포함 로직 방지.

## 13. OpenVINO + NNCF INT8 (#35, #36)

```
yolo26n.pt
    ↓ _export_onnx(half=False)
    ↓ ONNX
    ↓
    ↓ ov_fp32 = core.read_model(onnx_path)
    ↓
    ↓ [NNCF PTQ]
    ↓ calib_gen = _iter_calib_samples(val_yaml, n=300, imgsz=640, seed=42)
    ↓   — streaming (1, 3, 640, 640) numpy yield
    ↓ nncf_calib = nncf.Dataset(calib_gen, transform_func=lambda x: x)
    ↓
    ↓ ov_int8 = nncf.quantize(
    │    ov_fp32,
    │    nncf_calib,
    │    preset=QuantizationPreset.MIXED,
    │    target_device=TargetDevice.CPU,
    │    subset_size=300,
    │ )
    ↓ IR (quantized): {recipe}_int8.xml
    ↓
    ↓ compile_model(ov_int8, "CPU")
    ↓ OVRunnerAsORT → measure_latency
```

**파일**: `scripts/run_cpu.py::_prepare_openvino_int8_nncf` (Line 554)

**특징**:
- NNCF MIXED preset: activation=symmetric, weight=symmetric, attention block fallback.
- FastNAS 변형 (#36) 은 pruning 을 train.py 가 먼저 수행 → pruned ONNX 에 NNCF 적용.
- 현재 accuracy eval 은 FP32 ONNX 기반 placeholder (ultralytics 가 OV IR val 직접 지원 안 함 — Wave 6 Task 10 후속).

## 공통 후처리 — measurement + report

```
runner 공통:
    per_bs = {bs: measure_latency(forward_fn, warmup, measure)}
    throughput = fps = 1000 / p50_ms
    accuracy = ultralytics.YOLO(artifact).val(data=eval_yaml)
    Result(...).model_dump_json() → {results|results_qr|results_cpu_qr}/*.json
         ↓
    make report / make cpu-report
         ↓
    scripts/recommend.py rank + format_report → report*.md
```

## Dependency 요약

| Stage | 외부 툴 | 설치 |
|---|---|---|
| Weight load | ultralytics 8.4.27 | `pip install ultralytics` |
| ONNX export | torch.onnx.export (ultralytics 내부) | `torch 2.8.0+cu129` |
| TRT engine | tensorrt 10.16.0.72 | `pip install tensorrt` |
| modelopt quant | nvidia-modelopt 0.43+ | `pip install --extra-index-url https://pypi.nvidia.com nvidia-modelopt` |
| ORT quant (GPU + CPU) | onnxruntime-gpu 1.22 / onnxruntime 1.22 | `pip install onnxruntime-gpu` |
| OpenVINO | openvino 2026.1.0 | `pip install openvino` |
| NNCF | nncf | `pip install nncf` |

## 확장 가이드 — 신규 recipe 추가 시

1. YAML 작성: `recipes/{NN}_{name}.yaml` (규칙: `_schemas.py::Recipe` 검증 통과)
2. 기존 runner 경로 재사용 가능 여부 판단 (dtype / source / calibrator 조합)
3. 새 경로가 필요하면 해당 `run_*.py` 의 dispatcher 분기 추가
4. Makefile `recipe-{NN}` target + `all` 리스트 추가
5. `scripts/run_qr_batch.sh` / `run_cpu_batch.sh` ORDER 배열에 추가
6. `tests/test_schema_*.py` 에 schema 검증 테스트 추가 (신규 필드일 경우)
7. `make recipe-{NN}` 단일 실행 → Result JSON 검증
8. `make report` → 랭킹 확인

## 참고

- 전체 recipe 요약: [`docs/recipe-bank-summary.md`](recipe-bank-summary.md)
- Architecture: [`docs/architecture.md`](architecture.md)
- Wave 11 Task 5 (#33 fix): [`docs/plans/2026-04-22-wave11-recipe-debug-cleanup.md`](plans/2026-04-22-wave11-recipe-debug-cleanup.md)
- Wave 14 (asymmetric / opt_level / BF16): [`docs/plans/2026-04-22-wave14-trt-optimization.md`](plans/2026-04-22-wave14-trt-optimization.md)
- Schema: `scripts/_schemas.py`
- Runner roots: `scripts/run_{pytorch,ort,trt,cpu}.py`
