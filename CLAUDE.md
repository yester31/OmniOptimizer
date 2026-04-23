# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository. Keep it short; detail lives in `docs/`.

## What this project is

**OmniOptimizer** auto-searches vision model inference optimizations. Give it a
model + target GPU + constraints (max mAP drop, min fps); it runs a bank of
(runtime × technique) recipes end-to-end and recommends the winner.

Current scope: YOLO26n, one NVIDIA GPU + x86_64 Intel CPU, 31 active recipes
across GPU (`trt_builtin`, `modelopt`, `ort_quant`, `modelopt_fastnas`) and
CPU (`ort_cpu`, `openvino`, `openvino_fastnas`) backends plus FP32 / TF32 /
FP16 / BF16 / INT8 variants.
Wave 14 (TRT tuning) SHIPPED 2026-04-22 — #40 `trt_fp16_opt5` (fps 645.2,
+48% over #05 FP16 via `builder_optimization_level=5`) + #41 `trt_bf16` +
#42 `modelopt_int8_asymmetric` (**NEW TOP at fps 770.5**, dethrones
entropy 763.9 via `use_zero_point=True`).
Wave 15 (audit-driven tuning) SHIPPED 2026-04-23 with mixed outcome:
D1 no-regret switches landed (OpenVINO `CACHE_DIR` persistent kernel
cache, ORT TRT EP `trt_builder_optimization_level=5` + timing cache
with backward-compat fallback). D3 `MeasurementSpec.build_ceiling_s`
schema added for per-recipe build-time-warning config. **D2 opt_level=5
opt-in to #09/#12/#42 ROLLED BACK** after measurement — INT8 modelopt
recipes sit near the tactic-selection ceiling at opt_level=3, and
opt_level=5 trades bs=8 throughput (-9 to -60%) for no bs=1 gain. Wave
14 #40's +48% was FP16-specific (headroom-dependent). See
[`docs/plans/2026-04-23-wave15-audit-driven-tuning.md`](docs/plans/2026-04-23-wave15-audit-driven-tuning.md)
and
[`docs/improvements/2026-04-23-wave15-results.md`](docs/improvements/2026-04-23-wave15-results.md).
Wave 16 (post-T1 fact-driven scope) SHIPPED 2026-04-23 as thin scope:
**D1 `Result.build_ceiling_breached` bool field** round-trips Wave 15
D3's ceiling signal through Result JSON so `recommend.py` can surface
breaches in a `## Build-Time Ceiling Breaches` report section. T1
(`scripts/audit_capabilities.py` + `results/_capabilities.json`
snapshot) froze four capability facts; findings killed three of four
originally planned workstreams: YOLO26n has **no MHA pattern** (4
MatMul, 0 feed Softmax), i7-11375H lacks AVX-512_BF16 / AMX
(hardware-blocked `#43_openvino_bf16` deferred to Wave 17+),
NNCF IgnoredScope work un-needed. T7 (ONNX cache key
`nodes_to_exclude` hash) closed pre-existing cache-poisoning bug
where #09 and #12 shared `best_qr_640_modelopt_entropy_bs1.onnx`.
End-to-end validation caught a D1 latent crash on cached-engine
path (`build_time_s=None` vs `_ceiling` TypeError); fix refactored
tracker into `_advance_ceiling_tracker` helper with explicit None
handling and sticky-True semantics across the bs loop. See
[`docs/plans/2026-04-23-wave16-plan.md`](docs/plans/2026-04-23-wave16-plan.md)
and
[`docs/improvements/2026-04-23-wave16-d1-validation.md`](docs/improvements/2026-04-23-wave16-d1-validation.md).
Intel Neural Compressor was evaluated (Wave 3) and removed — see the audit
below. Brevitas recipes (#20-#22) archived 2026-04-22 (redundant with
`modelopt_int8_entropy`; Q/DQ asymmetry ruled out as cause of fps deficit).
Wave 11 archived #02 torchcompile_fp16 (torch.compile MSVC blocker on
Windows) + #03 ort_cuda_fp16 (CUDA EP can't run YOLO26n NMS ops).
See [`docs/improvements/2026-04-22-brevitas-qdq-audit.md`](docs/improvements/2026-04-22-brevitas-qdq-audit.md).
doc below for the incompatibility matrix. Full scope + architecture + commands in
[`docs/architecture.md`](docs/architecture.md). Latest audit:
[`docs/improvements/2026-04-18-trt-modelopt-audit.md`](docs/improvements/2026-04-18-trt-modelopt-audit.md).
Wave 6 CPU inference shipped 2026-04-21 — see
[`docs/plans/_shipped/2026-04-21-wave6-cpu-inference.md`](docs/plans/_shipped/2026-04-21-wave6-cpu-inference.md)
and `report_cpu_qr.md` at repo root for the 6-recipe ranking.
Waves 7 and 8 both ARCHIVED 2026-04-21 after Task 0 spikes hit the
same class of blocker: YOLO26n's end-to-end NMS pipeline (ReduceMax /
TopK / GatherElements / anchors attribute mutation) rejected by
external converters (torch.export, pnnx) and wheels (XNNPACK missing
in onnxruntime-gpu). See
[`docs/improvements/2026-04-21-wave7-r3-r5-spike-results.md`](docs/improvements/2026-04-21-wave7-r3-r5-spike-results.md)
and
[`docs/improvements/2026-04-21-wave8-r1-spike-results.md`](docs/improvements/2026-04-21-wave8-r1-spike-results.md).

Waves 12 and 13 ARCHIVED 2026-04-21 pre-execution after `/gsd-plan-phase`
cross-verify: Wave 12 (INT4 weight-only) defeated by NVIDIA docs
("WoQ is GEMM-only, no Conv support" — YOLO26n is Conv-dominant);
Wave 13 (ONNX autocast/autotune) plan relied on a nonexistent
`modelopt.onnx.autotune` module and mismatched `convert_to_f16`
signature. See
[`docs/improvements/2026-04-21-int4-ampere-not-feasible.md`](docs/improvements/2026-04-21-int4-ampere-not-feasible.md)
and
[`docs/improvements/2026-04-21-wave13-api-discovery-blocker.md`](docs/improvements/2026-04-21-wave13-api-discovery-blocker.md).

Wave 10 (FastNAS structured pruning) **SHIPPED 2026-04-22**
(reopened after initial archive). FastNAS pruning yields only
15.7% FLOPs cut (FX trace excludes YOLO backbone/neck) but
combined with `modelopt.onnx.quantize` INT8 entropy produces
two competitive recipes:
- `modelopt_fastnas_int8` — Rank 4, fps 716.3, mAP 0.947,
  engine **5 MB (−88% vs baseline 38 MB)**
- `modelopt_fastnas_sp_int8` — Rank 5, fps 697.4, mAP 0.948
  (2:4 sparsity + fine-tune with mask-preservation callback)

Relative to baseline #1 `modelopt_int8_entropy` (fps 763.9):
FastNAS maintains **93.8% speed while using 12.4% of engine
size** — primary value is edge/embedded/VRAM-constrained
deployment. Constraint `max_map_drop_pct: 5.0` (vs baseline
1.5%p) reflects this trade-off.
See [`docs/plans/_shipped/2026-04-21-wave10-modelopt-fastnas-pruning.md`](docs/plans/_shipped/2026-04-21-wave10-modelopt-fastnas-pruning.md),
[`docs/improvements/2026-04-21-wave10-r1-spike-results.md`](docs/improvements/2026-04-21-wave10-r1-spike-results.md),
and [`docs/improvements/2026-04-22-wave10-pruning-extended-eval.md`](docs/improvements/2026-04-22-wave10-pruning-extended-eval.md).

**Critical reopening lesson**: initial archive was caused by
ultralytics val (e2e with preprocess/postprocess) vs
`scripts/run_trt.py` (pure CUDA kernel, warmup=100,
measure=100) protocol mismatch — fps looked 4× lower than
reality. Phase 7 fair bench via `scripts.measure.measure_latency`
exposed the bias; Phase 8 `modelopt.onnx.quantization.quantize`
(not torch-level `mtq.quantize`) on the ONNX directly resolved
the Detect head "Missing scale" issue.

Active GPU plan: **Wave 9 DirectML EP** (ORT-native, bypasses
converter issues, covers AMD GPU / Intel Arc / Windows NPU).
Plan draft pending.

Other candidates (ordered by friction): Wave 8 rescope with
`end2end=False` export / Wave 6 close-out (#33 mAP=0 debug).

**Plan-writing conventions (learned 2026-04-21/22)**:
1. External library APIs must be grounded in the repo's actual
   `__init__.py` / source code, not web docs or ChatGPT answers.
   Wave 13 failed because docs-based plans cited phantom modules.
2. FastNAS / NAS frameworks depend heavily on `torch.fx` symbolic
   trace compatibility. Dynamic control flow + ModuleList-heavy
   detection models (YOLO family) exclude entire backbones from
   search space. Confirm trace feasibility per-module before
   committing to pruning ratio targets.
3. ModelOpt `mto.save` / `mto.restore` are tuned for
   architecture-invariant modes (QAT, sparsity). For
   architecture-changing modes (pruning), prefer ultralytics-style
   full-model pickle.
4. **Compare apples-to-apples**: ultralytics `yolo.val()` is e2e
   (preprocess + inference + postprocess); `scripts/run_trt.py`
   uses `scripts.measure.measure_latency` (pure CUDA kernel,
   warmup=100, measure=100, CUDA events). Never compare fps
   across protocols — rebuild one side to match before making
   ship/archive decisions.
5. **`modelopt.onnx.quantization.quantize` is the go-to path** for
   INT8 QDQ on custom-trained models. Torch-level `mtq.quantize`
   followed by `torch.onnx.export` crashes with C-level segfault
   on QuantConv2d in modelopt 0.43. The ONNX-level path injects
   Q/DQ nodes into a clean FP32 ONNX without wrapping modules.

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
