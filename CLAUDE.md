# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**OmniOptimizer** is an auto-search tool for vision model inference optimization.
Give it a model + a target GPU + constraints (max mAP drop, min fps), and it runs
a bank of (runtime × technique) recipes end-to-end, then recommends the best one.

Current scope (as of Wave 3, 2026-04-18):

- **Model**: YOLO26n (Ultralytics)
- **Hardware**: one NVIDIA GPU (Ampere+ for sparsity recipes / TF32)
- **Runtimes × Techniques** (21 recipes): PyTorch eager FP32, `torch.compile` FP16,
  ONNX Runtime on CUDA EP and TensorRT EP (FP16), native TensorRT FP32/TF32/FP16/INT8.
  INT8 covers 4 backends — `trt_builtin` (native calibrator), `modelopt`
  (NVIDIA ModelOpt ONNX-path, max/entropy/percentile/mixed-precision),
  `ort_quant` (onnxruntime.quantization, 4 calibration methods),
  `neural_compressor` (Intel INC PTQ + SmoothQuant). Plus 2:4 sparsity and
  QAT recipes parked pending training pipeline.
- **Metrics**: p50/p95/p99 latency (wall-clock + CUDA Event GPU-only), fps (bs=1, 8),
  peak GPU mem (torch + NVML delta), mAP@0.5 / mAP@0.5-0.95, model size, cold-start.

Original v1 plan approved 2026-04-17 (7 recipes). Wave 1/2 landed perf hygiene
and modelopt/TF32; Wave 3 added ORT + INC backends. See
`docs/improvements/2026-04-18-trt-modelopt-audit.md` for the full audit trail
and `docs/plans/2026-04-18-wave3-ort-inc.md` for the most recent plan.

Design doc (personal): `~/.gstack/projects/yester31-OmniOptimizer/yeste-main-design-20260417-093458.md`
Approved plan: `~/.claude/plans/synchronous-skipping-hamming.md`

## Common commands

```bash
# End-to-end: run all active recipes and emit report.md
# (21 defined; #7/#11/#19 parked, so make all runs 18.)
make all

# One recipe at a time
make recipe-01    # PyTorch FP32 baseline
make recipe-06    # TensorRT INT8 PTQ (trt_builtin)
make recipe-09    # TensorRT INT8 PTQ (modelopt, entropy)
make recipe-15    # TensorRT INT8 PTQ (ort_quant, percentile)
make recipe-18    # TensorRT INT8 PTQ (neural_compressor, SmoothQuant)
...

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

# Fast sanity checks that don't need a GPU
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('scripts').glob('*.py')]"
python -c "import sys; sys.path.insert(0, '.'); from scripts._schemas import load_recipe; import pathlib; [load_recipe(str(p)) for p in pathlib.Path('recipes').glob('*.yaml')]"
```

## Architecture (v1, flat-by-design)

```
recipes/                YAML inputs — one file per (runtime × technique) combination
scripts/
  _schemas.py           pydantic models: Recipe, Result, and their sub-models
  env_lock.py           GPU/driver/CUDA snapshot; optional nvidia-smi -lgc
  measure.py            warm-up loop + latency percentiles + cold-start + peak mem
  eval_coco.py          mAP eval (ultralytics val for v1; generic path is stubbed)
  run_pytorch.py        recipes #1, #2
  run_ort.py            recipes #3, #4 (onnxruntime EPs; exports ONNX from .pt)
  run_trt.py            recipes #5, #6, #7 (builds .engine, optional INT8 + 2:4 sparsity)
  recommend.py          reads results/*.json, ranks, writes report.md
results/                JSON output (one per recipe) + _env.json + _onnx/ + _engines/
Makefile                `make all` = seven recipes + report
Dockerfile              nvcr.io/nvidia/pytorch:24.05-py3 + project installed
```

Design premise: v1 stays flat. No abstract `Runtime` base class on purpose — the
plan calls out that an adapter layer is v2 work, once a second model or a second
GPU forces it. Pull this forward only when something concrete (a new runtime
without a home, or duplicated code that keeps drifting) justifies it.

## Important conventions

- **Recipes drive behaviour.** Every knob (dtype, calibrator, warm-up counts,
  clock lock) lives in the YAML, not in argparse flags. Runner scripts only take
  `--recipe` and `--out`.
- **Result JSON is the contract.** Anything added to `Result` in `_schemas.py`
  must flow through the runners and through `recommend.py`. Do not log metrics
  that don't round-trip through the schema.
- **Measurement hygiene is load-bearing.** The audience is MLPerf / paper-replication
  folks. Never remove `warmup_iters`, `measure_iters`, or percentile reporting to
  "simplify" something. Average-only latency is a regression.
- **Degrade, don't crash.** If TensorRT isn't installed, if 2:4 sparsity isn't
  supported on the device, if a specific batch size fails to build — the runner
  records the failure in `Result.notes` and `meets_constraints=False` so
  `recommend.py` still produces a report. `make all` must not abort.
- **TRT runner uses torch.cuda, not pycuda.** `run_trt.py` shares torch's CUDA
  context for all device allocations (calibrator and forward). Mixing
  `pycuda.autoinit` with the torch context caused illegal-memory-access crashes
  on Windows. Keep it torch-only unless you have a concrete reason to switch.
- **Extension hook for nvidia-modelopt (v1.1+).** `TechniqueSpec.source` selects
  the quantization/sparsity backend. v1 ships `trt_builtin` (TRT's entropy
  calibrator + `SPARSE_WEIGHTS` flag). When adding modelopt: quantize at the
  torch level, export QDQ-annotated ONNX, feed it to the existing `_build_engine`
  (INT8 flag not needed — QDQ nodes carry the scales). `_make_trt_forward`
  stays unchanged regardless of quantization source.

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
