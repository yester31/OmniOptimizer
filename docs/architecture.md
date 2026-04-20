# OmniOptimizer Architecture

Extracted from CLAUDE.md to keep the root file short. Cross-reference this
document when planning non-trivial changes.

## What this project is

**OmniOptimizer** is an auto-search tool for vision model inference optimization.
Give it a model + a target GPU + constraints (max mAP drop, min fps), and it runs
a bank of (runtime × technique) recipes end-to-end, then recommends the best one.

## Current scope (post Wave 5 training pipeline, 2026-04-20)

- **Model**: YOLO26n (Ultralytics).
- **Hardware**: one NVIDIA GPU (Ampere+ for sparsity recipes / TF32).
- **Runtimes × Techniques** — 21 recipes defined, **20 active + 1 parked** in `make all`:
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
- **Metrics**: p50/p95/p99 latency (wall-clock + CUDA Event GPU-only), fps (bs=1, 8),
  peak GPU mem (torch + NVML delta), mAP@0.5 / mAP@0.5-0.95, model size, cold-start.

Original v1 plan approved 2026-04-17 (7 recipes). Wave 1/2 added perf hygiene,
modelopt, CUDA graph capture, timing cache reuse, FP32/TF32 baselines, CUDA-event
GPU timing, NVML mem delta, polygraphy diagnose. Wave 3 added ORT + INC backends.

Design doc (personal): `~/.gstack/projects/yester31-OmniOptimizer/yeste-main-design-20260417-093458.md`

## Directory layout (flat-by-design)

```
recipes/                YAML inputs — one file per (runtime × technique) combination
scripts/
  _schemas.py           pydantic models: Recipe, Result, LatencyStats, etc.
  env_lock.py           GPU/driver/CUDA snapshot; optional nvidia-smi -lgc
  measure.py            warmup + latency percentiles + CUDA events + cold-start + peak mem
  eval_coco.py          mAP eval (ultralytics val for v1; generic path stubbed)
  run_pytorch.py        recipes #01, #02
  run_ort.py            recipes #03, #04 (onnxruntime EPs; exports ONNX from .pt)
  run_trt.py            recipes #00, #05..#19 (builds .engine, dispatches INT8 backends)
  recommend.py          reads results/*.json, ranks, writes report.md
results/                JSON output (one per recipe) + _env.json + _onnx/ + _engines/
docs/
  architecture.md       this file
  improvements/         audit docs
  plans/                TDD-style implementation plans
Makefile                make all / make recipe-XX / make report
Dockerfile              nvcr.io/nvidia/pytorch:24.05-py3 base
```

**Design premise**: v1 stays flat. No abstract `Runtime` base class on purpose — an
adapter layer is v2 work, deferred until a second model or a second GPU forces it.
Pull this forward only when concrete duplication (a new runtime without a home, code
that keeps drifting) justifies it.

## Common commands

```bash
# End-to-end: run all active recipes and emit report.md
# (21 defined; #22 parked; #07/#11/#17 active via Wave 5, so `make all` runs 20.)
make all

# One recipe at a time
make recipe-01    # PyTorch FP32 baseline
make recipe-06    # TensorRT INT8 PTQ (trt_builtin)
make recipe-09    # TensorRT INT8 PTQ (modelopt, entropy)
make recipe-15    # TensorRT INT8 PTQ (ort_quant, percentile)
make recipe-18    # TensorRT INT8 PTQ (neural_compressor, SmoothQuant)

# Re-generate the report from existing results/
python scripts/recommend.py --results-dir results --out report.md

# Clean artefacts
make clean

# Docker (reproducible environment)
docker build -t omnioptimizer:v1 .
docker run --rm --gpus all -v "$PWD":/workspace/omnioptimizer omnioptimizer:v1 \
    bash -c "make all && cat report.md"

# Install locally (pip). tensorrt needs NVIDIA's wheel index separately.
pip install -e ".[all]"
pip install --extra-index-url https://pypi.nvidia.com tensorrt

# Fast sanity checks (no GPU needed)
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('scripts').glob('*.py')]"
python -c "import sys; sys.path.insert(0, '.'); from scripts._schemas import load_recipe; import pathlib; [load_recipe(str(p)) for p in pathlib.Path('recipes').glob('*.yaml')]"
```

## Extended conventions

Beyond the critical rules kept in CLAUDE.md, these guide runner development:

- **Extension hook via `technique.source`.** The INT8/quantization backend is
  selected by `TechniqueSpec.source` in the recipe YAML. Current dispatch targets:
  `trt_builtin` (TRT's entropy calibrator + `SPARSE_WEIGHTS`), `modelopt`
  (ONNX-path QDQ via `modelopt.onnx.quantization.quantize`), `ort_quant`
  (`onnxruntime.quantization.quantize_static`), `brevitas` (PyTorch-level PTQ
  via `brevitas.graph.quantize` + `export_onnx_qcdq`). Adding a new backend means: one `_prepare_*_onnx` helper in
  `run_trt.py` that emits QDQ ONNX, one dispatcher branch in `_prepare_onnx`,
  and an entry in `_SOURCE_TAG` for engine cache filename shortening.
  `_build_engine` and `_make_trt_forward` do not change — they consume QDQ ONNX
  identically regardless of source.
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

### Waves 1-5 현황

| Wave | 범주 | 레시피 수 | 상태 |
|---|---|---|---|
| Wave 1 | TRT builtin (FP32/FP16/INT8 PTQ) | #00-#06 | active |
| Wave 2 | modelopt INT8 PTQ | #08-#12 | active |
| Wave 3 | ONNX Runtime INT8 | #13-#16 | active (INC는 dropped) |
| Wave 4 | Brevitas eager PTQ | #20-#21 | active (#22 parked: no entropy observer) |
| Wave 5 | Training pipeline (QAT/sparsity) | #07, #11, #17 | active (2026-04-20) |

**Active 20 + Parked 1 = 21 총 레시피**.

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
