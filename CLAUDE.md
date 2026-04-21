# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository. Keep it short; detail lives in `docs/`.

## What this project is

**OmniOptimizer** auto-searches vision model inference optimizations. Give it a
model + target GPU + constraints (max mAP drop, min fps); it runs a bank of
(runtime × technique) recipes end-to-end and recommends the winner.

Current scope: YOLO26n, one NVIDIA GPU + x86_64 Intel CPU, 28 recipes across
GPU (`trt_builtin`, `modelopt`, `ort_quant`, `brevitas`) and CPU (`ort_cpu`,
`openvino`) backends plus FP32 / TF32 / FP16 / BF16 / INT8 variants.
Intel Neural Compressor was evaluated (Wave 3) and removed — see the audit
doc below for the incompatibility matrix. Full scope + architecture + commands in
[`docs/architecture.md`](docs/architecture.md). Latest audit:
[`docs/improvements/2026-04-18-trt-modelopt-audit.md`](docs/improvements/2026-04-18-trt-modelopt-audit.md).
Latest plan: [`docs/plans/2026-04-21-wave6-cpu-inference.md`](docs/plans/2026-04-21-wave6-cpu-inference.md).

## Critical conventions (load-bearing — violating these causes regressions)

- **Recipes drive behaviour.** Every knob (dtype, calibrator, warmup counts,
  clock lock) lives in the YAML. Runner scripts only take `--recipe` and `--out`.
- **Result JSON is the contract.** Anything added to `Result` in `_schemas.py`
  must flow through the runners and `recommend.py`. Do not log metrics that
  don't round-trip through the schema.
- **Measurement hygiene is non-negotiable.** Audience is MLPerf / paper-replication.
  Never drop `warmup_iters`, `measure_iters`, or percentile reporting to "simplify".
  Average-only latency is a regression.
- **Degrade, don't crash.** Missing TRT, unsupported sparsity, failed build —
  record in `Result.notes` with `meets_constraints=False`; `make all` must finish.
- **TRT runner uses `torch.cuda`, not `pycuda`.** Shared torch CUDA context is
  mandatory; `pycuda.autoinit` caused illegal-memory-access on Windows.
- **`scripts/run_cpu.py` must not import `tensorrt` at module load.** CPU-only
  environments (CI, macOS, TRT-less Windows) rely on this to keep
  `pytest tests/` green. openvino / onnxruntime.quantization imports live
  inside dispatcher branches, and TRT-free helpers live in
  `scripts/_weights_io.py`. Guarded by `test_run_cpu_imports_without_tensorrt`.

See `docs/architecture.md` for extension hooks, QDQ→TRT compat checklist,
and Windows-specific gotchas.

## Health Stack

- test: pytest tests/ -q
- typecheck: mypy scripts --ignore-missing-imports --no-strict-optional
- shell: (not installed — skellcheck 없음, 스킵)
- lint: (ruff 미설정 — 스킵)
- deadcode: (vulture 없음 — 스킵)

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the
Skill tool as your FIRST action. Do NOT answer directly, do NOT use other tools
first. The skill has specialized workflows that produce better results than
ad-hoc answers.

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
