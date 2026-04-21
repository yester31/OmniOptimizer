# OmniOptimizer Architecture

Extracted from CLAUDE.md to keep the root file short. Cross-reference this
document when planning non-trivial changes.

## What this project is

**OmniOptimizer** is an auto-search tool for vision model inference optimization.
Give it a model + a target GPU + constraints (max mAP drop, min fps), and it runs
a bank of (runtime × technique) recipes end-to-end, then recommends the best one.

## Current scope (post Wave 6 CPU inference, 2026-04-21)

- **Model**: YOLO26n (Ultralytics).
- **Hardware**: one NVIDIA GPU (Ampere+ for sparsity recipes / TF32) **+
  x86_64 Intel CPU** for Wave 6 recipes (AVX2 minimum, AVX-512 VNNI
  recommended, AMX / AVX-512 BF16 optional for #31).
- **Runtimes × Techniques** — 28 recipes defined, **20 active GPU + 6 CPU + 2 parked**:
  - PyTorch eager FP32 (#01), `torch.compile` FP16 (#02).
  - ONNX Runtime CUDA EP (#03) / TensorRT EP (#04), both FP16.
  - Native TensorRT: FP32 (#00), FP32+TF32 (#00-tf32), FP16 (#05), INT8 PTQ (#06).
  - INT8 `modelopt` (ONNX-path PTQ): max (#08), entropy (#09), percentile (#10),
    mixed precision (#12).
  - INT8 `ort_quant` (`onnxruntime.quantization.quantize_static`): minmax (#13),
    entropy (#14), percentile (#15), distribution (#16).
  - INT8 `brevitas` (PyTorch-native eager PTQ → QDQ ONNX): percentile (#20),
    MSE (#21). Entropy (#22) parked — Brevitas 0.10.x has no entropy/KL
    activation observer, and silently fell back to default stats, producing
    byte-identical ONNX to percentile. GPTQ dropped — requires Brevitas
    graph-mode, which fx-traces YOLO26n and fails on ultralytics' Python-flow
    forward.
  - **Parked**: #22 brevitas_int8_entropy only (no supporting observer in
    Brevitas 0.10.x). #07 trt_int8_sparsity and #11 modelopt_int8_sparsity
    are now **active** (Wave 5 training pipeline). #17 modelopt_int8_qat is
    new and **active** (Wave 5). `#19 inc_int8_qat` was dropped, not parked —
    INC backend was removed in Wave 3 (9d064ca).
  - **CPU recipes (Wave 6)** — driven by `scripts/run_cpu.py`, kept out of
    the default `make all` so GPU measurement variance isn't polluted by
    CPU thermal load on laptops:
    - `#30 ort_cpu_fp32` — ONNX Runtime CPU EP, FP32 baseline.
    - `#31 ort_cpu_bf16` — BF16, gated on `amx_tile || avx512_bf16`. Skipped
      on hosts without either ISA (recorded in Result.notes).
    - `#32 ort_cpu_int8_dynamic` — `quantize_dynamic`, QUInt8, per-tensor
      (CPU MLAS gotcha: QInt8+per_channel emits unsupported ConvInteger(10)).
    - `#33 ort_cpu_int8_static` — `quantize_static` QDQ, symmetric act/weight,
      entropy calibrator, 128 samples (down from 512 — Windows paging OOMed
      on bulk np.stack).
    - `#34 openvino_fp32` — Intel OpenVINO runtime, FP32 IR.
    - `#35 openvino_int8_nncf` — NNCF MIXED preset, 300 samples, no
      ignored_scope (R1 spike CLEARED on YOLO26n attention blocks).
- **Metrics**: p50/p95/p99 latency (wall-clock + CUDA Event GPU-only),
  **stddev_ms** (wall-clock variance, Wave 6), fps (bs=1, 8), peak GPU mem
  (torch + NVML delta), mAP@0.5 / mAP@0.5-0.95, model size, cold-start.

  **cold_start_ms reads differently per backend** — TRT engine cache hit
  ≈ 50 ms / miss minutes (plan + calibration); ORT CPU InferenceSession
  ≈ 200 ms (graph opt); OpenVINO `read_model` + `compile_model`
  ≈ 500 ms–1.9 s (LATENCY hint, measured i7-11375H). Use cold_start as
  a reference, not a ranking axis.

Original v1 plan approved 2026-04-17 (7 recipes). Wave 1/2 added perf hygiene,
modelopt, CUDA graph capture, timing cache reuse, FP32/TF32 baselines, CUDA-event
GPU timing, NVML mem delta, polygraphy diagnose. Wave 3 added ORT + INC backends.

Design doc (personal): `~/.gstack/projects/yester31-OmniOptimizer/yeste-main-design-20260417-093458.md`

## Directory layout (flat-by-design)

```
recipes/                YAML inputs — one file per (runtime × technique) combination
scripts/
  _schemas.py           pydantic models: Recipe, Result, LatencyStats, etc.
  _weights_io.py        TRT-free weight/ONNX/calibration helpers (Wave 6)
  env_lock.py           GPU + CPU snapshot; optional GPU/CPU clock lock
  measure.py            warmup + latency percentiles + stddev + CUDA events + cold-start
  eval_coco.py          mAP eval (ultralytics val for v1; generic path stubbed)
  run_pytorch.py        recipes #01, #02
  run_ort.py            recipes #03, #04 (onnxruntime EPs; exports ONNX from .pt)
  run_trt.py            recipes #00, #05..#22 (builds .engine, dispatches INT8 backends)
  run_cpu.py            recipes #30..#35 (ORT CPU EP + OpenVINO, Wave 6)
  recommend.py          reads results/*.json, ranks, writes report.md
results/                GPU JSON (one per recipe) + _env.json + _onnx/ + _engines/
results_cpu/            CPU JSON (Wave 6) + _onnx/ + _ov_ir/
docs/
  architecture.md       this file
  improvements/         audit docs
  plans/                TDD-style implementation plans
Makefile                make all / make recipe-XX / make cpu-all / make report
Dockerfile              nvcr.io/nvidia/pytorch:24.05-py3 base
```

**Design premise**: v1 stays flat. No abstract `Runtime` base class on purpose — an
adapter layer is v2 work, deferred until a second model or a second GPU forces it.
Pull this forward only when concrete duplication (a new runtime without a home, code
that keeps drifting) justifies it.

## Common commands

```bash
# End-to-end GPU: run all active recipes and emit report.md
# (22 GPU recipes defined; #22 parked, so `make all` runs 20.)
make all

# End-to-end CPU (Wave 6): runs #30-#35 → results_cpu/ → report_cpu.md
make cpu-all

# One recipe at a time
make recipe-01    # PyTorch FP32 baseline
make recipe-06    # TensorRT INT8 PTQ (trt_builtin)
make recipe-09    # TensorRT INT8 PTQ (modelopt, entropy)
make recipe-15    # TensorRT INT8 PTQ (ort_quant, percentile)
make recipe-30    # ORT CPU EP FP32 (Wave 6)
make recipe-35    # OpenVINO INT8 NNCF (Wave 6)

# Re-generate the report from existing results/
python scripts/recommend.py --results-dir results --out report.md
python scripts/recommend.py --results-dir results_cpu --out report_cpu.md

# QR-fine-tuned checkpoint bank (Wave 5 + Wave 6)
bash scripts/run_qr_train_batch.sh   # GPU training (Wave 5)
make cpu-qr                           # CPU inference bank → report_cpu_qr.md

# Clean artefacts
make clean

# Docker (reproducible environment)
docker build -t omnioptimizer:v1 .
docker run --rm --gpus all -v "$PWD":/workspace/omnioptimizer omnioptimizer:v1 \
    bash -c "make all && cat report.md"

# Install locally (pip). tensorrt needs NVIDIA's wheel index separately.
pip install -e ".[all]"   # torch+onnx+trt+brevitas+modelopt+cpu
pip install -e ".[cpu]"   # Wave 6 CPU only: openvino + nncf + py-cpuinfo + psutil
pip install --extra-index-url https://pypi.nvidia.com tensorrt

# Fast sanity checks (no GPU needed)
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('scripts').glob('*.py')]"
python -c "import sys; sys.path.insert(0, '.'); from scripts._schemas import load_recipe; import pathlib; [load_recipe(str(p)) for p in pathlib.Path('recipes').glob('*.yaml')]"
```

## Extended conventions

Beyond the critical rules kept in CLAUDE.md, these guide runner development:

- **Extension hook via `technique.source`.** The INT8/quantization backend is
  selected by `TechniqueSpec.source` in the recipe YAML. Current dispatch
  targets split along the runner boundary:
  - **GPU (run_trt.py)**: `trt_builtin` (TRT's entropy calibrator +
    `SPARSE_WEIGHTS`), `modelopt` (ONNX-path QDQ via
    `modelopt.onnx.quantization.quantize`), `ort_quant`
    (`onnxruntime.quantization.quantize_static`), `brevitas` (PyTorch-level
    PTQ via `brevitas.graph.quantize` + `export_onnx_qcdq`).
  - **CPU (run_cpu.py, Wave 6)**: `ort_cpu` (CPU EP +
    `quantize_dynamic`/`quantize_static`), `openvino` (OV runtime +
    NNCF PTQ for INT8).

  Adding a new **GPU** backend: one `_prepare_*_onnx` helper in
  `run_trt.py` that emits QDQ ONNX, one dispatcher branch in
  `_prepare_onnx`, and an entry in `_SOURCE_TAG` for engine cache filename
  shortening. `_build_engine` and `_make_trt_forward` don't change — they
  consume QDQ ONNX identically regardless of source.

  Adding a new **CPU** backend: one `_prepare_<source>_<dtype>` helper in
  `run_cpu.py` returning `(runner, input_name, output_names, cold_ms,
  onnx_path)`, and a dispatcher branch in `_prepare_cpu_session`. If the
  runner isn't an ORT session, wrap it with an ORT-compatible adapter
  (see `OVRunnerAsORT` for the minimal surface — `.run`, `.get_inputs`,
  `.get_outputs`). **Never import `tensorrt` or `pycuda` at module load
  time** — the CPU-only test (`test_run_cpu_imports_without_tensorrt`)
  will fail and CI on non-GPU runners will break.
- **QDQ ONNX → TRT compatibility checklist** (painfully learned in Wave 3):
  - Symmetric activation + weight quantization (zero_point=0).
  - No bias quantization — TRT computes INT32 bias internally from
    `activation_scale × weight_scale`.
  - `QuantFormat.QDQ`, not `QOperator` (TRT has no plugin for
    `com.microsoft.QLinearSigmoid` etc.).
  - Per-channel weights on `axis=0` for Conv; per-tensor for activations.
  - Run ORT's `quant_pre_process` (shape inference + folding) before
    `quantize_static` — otherwise histogram calibrators OOM on attention graphs.
## Wave 5 — Training pipeline (2026-04-20)

QAT/sparsity 레시피는 recipe YAML의 ``technique.training`` 섹션으로 학습
파라미터를 기술하고, ``scripts/train.py`` 가 modifier별 전후훅을 실행.

- ``#07 trt_int8_sparsity``: ``prune_24`` modifier — magnitude 2:4 pruning
  + SAT 60 epochs (forward_pre_hook + mask 영구 적용 후 plain state_dict 저장).
- ``#11 modelopt_int8_sparsity``: ``modelopt_sparsify`` modifier —
  ``modelopt.torch.sparsity.sparsify(sparse_magnitude, 2:4)`` 60 epochs +
  ``mto.save/restore``.
- ``#17 modelopt_int8_qat``: ``modelopt_qat`` modifier — ``mtq.quantize(INT8_DEFAULT_CFG)``
  fake quant 삽입 + 30 epochs QAT at ``lr0=1e-4`` (AMP=False, scale 안정).

설계 원칙:
- Modifier 플러그인 (`scripts/_modifiers/{prune_24,modelopt_sparsify,modelopt_qat}.py`)
  각자 ``apply(yolo, spec)`` + ``finalize(yolo, spec, out_pt)`` 노출. prune_24는
  PRE_TRAIN_HOOK=True 플래그로 ultralytics on_train_start 콜백 안에서 적용.
- In-memory ``yolo.model`` (또는 학습 후 ``yolo.trainer.model``) 직접 직렬화 →
  ultralytics EMA-based best.pt 선택 회피로 validation leak 차단.
- modelopt modifier는 AMP 비활성 + ``modelopt.torch.opt.save/restore``로 wrapped
  state (QuantConv, QuantLinear) 보존. prune_24는 plain ultralytics checkpoint 포맷.
- Silent no-op guard: 세 modifier 모두 ``apply()``가 실제로 0 레이어에 적용된
  경우 ``RuntimeError`` 발생. 2:4 패턴은 finalize에서 assert.
- 출력: ``trained_weights/{recipe.name}.pt`` (+ ``.train.json`` 메타데이터).
  Git 추적 대상 아님 (``.gitignore``).

재현은 ``bash scripts/run_qr_train_batch.sh`` (학습 ~2h) → ``bash scripts/run_qr_batch.sh``
(평가). Smoke dry-run: ``OMNI_TRAIN_SMOKE=1 bash scripts/run_qr_train_batch.sh``
(1 epoch + 10% data).

설계 스펙: ``docs/superpowers/specs/2026-04-20-qr-training-pipeline-design.md``.
Implementation plan: ``docs/superpowers/plans/2026-04-20-qr-training-pipeline.md``.

### Waves 1-6 현황

| Wave | 범주 | 레시피 수 | 상태 |
|---|---|---|---|
| Wave 1 | TRT builtin (FP32/FP16/INT8 PTQ) | #00-#06 | active |
| Wave 2 | modelopt INT8 PTQ | #08-#12 | active |
| Wave 3 | ONNX Runtime INT8 | #13-#16 | active (INC는 dropped) |
| Wave 4 | Brevitas eager PTQ | #20-#21 | active (#22 parked: no entropy observer) |
| Wave 5 | Training pipeline (QAT/sparsity) | #07, #11, #17 | active (2026-04-20) |
| Wave 6 | CPU inference (ORT CPU + OpenVINO) | #30-#35 | active (2026-04-21) |

**Active 26 + Parked 1 = 27 총 레시피** (#31 BF16은 hardware-gated: AMX/AVX-512 BF16
없는 호스트에서는 자동 skip, recipe 정의는 유지).

## Wave 6 — CPU inference (2026-04-21)

x86_64 Intel CPU를 GPU와 대등한 측정 타겟으로 추가. 기존 GPU 런타임 코드는
건드리지 않고, 새 `scripts/run_cpu.py`가 `technique.source in {ort_cpu,
openvino}` 두 경로를 디스패치. `scripts/_weights_io.py`에서 TRT 독립 공통
헬퍼(`_resolve_weights`, `_export_onnx`, `_letterbox`, `_iter_calib_samples`,
`_build_calib_numpy`)를 제공해 CPU runner가 TRT/CUDA imports 없이 동작.

- **ORT CPU EP (`#30-#33`)**: fp32 / bf16(gated) / int8 dynamic / int8 static.
  Static INT8 설정은 Wave 3 P0-A 패치를 그대로 이식 — symmetric activation +
  weight, per-channel weight, `DedicatedQDQPair=False` (CPU MLAS는 fused QDQ
  선호, TRT explicit quantization와 반대).
- **OpenVINO (`#34-#35`)**: fp32 IR + NNCF MIXED preset INT8. `OVRunnerAsORT`
  어댑터로 `openvino.CompiledModel`을 ORT InferenceSession처럼 감싸 기존
  `measure_latency` 포워드 클로저가 그대로 동작. `_get_ov_core()` 싱글톤으로
  Core init 비용 (~300ms)을 프로세스당 1회로 상각.
- **Measurement hygiene**: CPU 추론은 GPU보다 variance가 크다. 기본값을
  warmup 200 / measure 300 으로 상향(GPU는 100/100), `LatencyStats.stddev_ms`
  필드로 분산을 공개하고, `MeasurementSpec.iter_cooldown_ms`로 per-iter sleep
  옵션 제공 (thermal throttle 완화). `measure_latency`는 cooldown을 **타이밍
  창 바깥**에서 실행해 측정값이 오염되지 않음.
- **Environment lock**: `env_lock.py`가 Linux `/proc/cpuinfo`, Windows
  `py-cpuinfo`, macOS `sysctl`로 cpu_model / cpu_cores_physical / cpu_flags
  수집. py-cpuinfo의 `avx512vnni` ↔ Linux `avx512_vnni` 같은 표기 차이는
  `_FLAG_ALIASES`로 정규화해 Result.env.cpu_flags가 플랫폼에 관계없이
  일관. `lock_cpu_clock(True)`: Linux `cpupower frequency-set`, Windows
  `powercfg /setactive <High performance>`, macOS는 degrade-not-crash로 note만.
- **Batch + report 분리**: `make cpu-all`(results_cpu/) 및
  `scripts/run_cpu_batch.sh` (results_cpu_qr/)가 GPU `make all`과 별도 디렉터리
  사용. 같은 랩톱에서 두 배치를 번갈아 돌려도 JSON 충돌 없음.

재현:
- 유닛: `pytest tests/test_run_cpu_*.py tests/test_env_lock.py tests/test_measure_hygiene.py`
- Smoke: `python scripts/run_cpu.py --recipe recipes/30_ort_cpu_fp32.yaml --out /tmp/smoke_30.json`
- 풀 배치: `make cpu-all` (또는 QR 데이터셋: `make cpu-qr`)

설계 스펙: `docs/plans/2026-04-21-wave6-cpu-inference.md`.
R1 spike (OpenVINO + YOLO26n attention) CLEARED 기록도 해당 plan 파일에 포함.

- **Windows-specific gotchas**:
  - `torch.cuda` only for CUDA context sharing (never `pycuda.autoinit`).
  - Engine cache filenames include short `_SOURCE_TAG` prefixes (`_ort`, `_inc`)
    because `_neural_compressor` pushes paths past MAX_PATH.
  - INC 2.x requires `setuptools<81` (still imports `pkg_resources`).
- **INC (Intel Neural Compressor) status**: recipes #17/#18 currently fail the
  TRT build for YOLO26n's attention block (SmoothQuant rewrites weight shapes
  but leaves downstream Reshape nodes stale). The dispatch path stays for
  future reactivation (INC 3.x or torch-level SmoothQuant reimpl). See
  `docs/improvements/2026-04-18-trt-modelopt-audit.md` Wave 3 section.
