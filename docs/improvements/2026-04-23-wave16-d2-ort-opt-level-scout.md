# Wave 16 D2 scout — ORT `graph_optimization_level` audit

**Date**: 2026-04-23
**Status**: NULL RESULT (audit-only, no measurement needed). Wave 16 D2 closed.
**Related**: `docs/plans/2026-04-23-wave16-plan.md` D2, T1 findings in `results/_capabilities.json`.

## Motivation

T1 audit (TODOS, snapshot in PR #7) recorded that ORT 1.22 `ORT_ENABLE_ALL` (level 99) adds NCHWc layout rewrites and MLAS-specific fusion passes beyond `ORT_ENABLE_EXTENDED` (level 2). On Tiger Lake i7-11375H without matching NCHWc kernels for every op, the ALL-level passes could run without paying off — wasted build time.

Wave 16 plan D2 proposed measuring representative CPU recipes at EXTENDED vs ALL. If EXTENDED were within ±2% fps AND ≥20% faster build, flip the level.

## Finding: no CPU recipe uses ALL

Code audit reveals the premise is already correct in the shipped code:

```python
# scripts/run_cpu.py:123-131
def _build_ort_session_options(recipe: Recipe):
    """Common ORT SessionOptions: EXTENDED graph opts, explicit thread count."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.intra_op_num_threads = _resolve_thread_count(recipe)
    so.inter_op_num_threads = 1
    return so
```

All CPU recipes (#30 `ort_cpu_fp32`, #32 `ort_cpu_int8_dynamic`, #33 `ort_cpu_int8_static`) go through this single helper. They use `ORT_ENABLE_EXTENDED`, not `ORT_ENABLE_ALL`. **Nothing to flip.**

The plan's D2 scope was based on an incorrect reading of the runtime config. No measurement needed.

## Finding: GPU ORT path uses ALL — but the delta is architecturally low-impact

```python
# scripts/run_ort.py:137-138
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

Recipe #04 `ort_trt_fp16` uses ORT_ENABLE_ALL with provider chain `[TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider]`. When TRT EP claims the subgraph (which it does for the bulk of YOLO26n), ORT's optimizer only runs on nodes that fall through — a tiny residual set.

The NCHWc layout rewrites that ALL adds over EXTENDED are x86 CPU-focused. They don't interact with TRT's own kernel selection. In the residual CPU-EP fallback path, the layout rewrite could in principle waste build time, but the fallback op count is small enough that the delta is negligible.

**Not worth measuring** unless a suspicious build-time-vs-runtime ratio ever points here. If measured in a future wave, the test matrix is:
- #04 `ort_trt_fp16` at `ORT_ENABLE_ALL` (current) vs `ORT_ENABLE_EXTENDED`
- Compare `build_time_s` and fps at bs1/bs8
- Expect: fps delta < 0.5%, build_time delta < 5%. If either exceeds, worth investigating.

## Conclusions

- Wave 16 D2 closed as **null result** based on code audit. No measurement needed.
- CPU recipes are already at the right level. No action.
- GPU ORT recipes use ALL but the delta is theoretically low-impact; file as "measure if suspicious" in a future wave rather than pre-emptive.
- T1's observation about ALL-vs-EXTENDED was academically correct but did not translate to a recipe change on this codebase.

## Lessons

- **Audit the runtime config before writing a measurement plan.** The plan assumed "CPU recipes use ALL" without grepping the runner. A 30-second grep would have reframed D2 before it was written.
- **T1 findings still earned their keep** — they surfaced the ALL-vs-EXTENDED semantics, which is now documented and re-usable. The next person tempted to flip ORT opt levels has a paper trail.
- **Not every scout needs measurement.** Sometimes the code already answers the question.

## Follow-ups

None. Wave 16 D2 complete (null result).
