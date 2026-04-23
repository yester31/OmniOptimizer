# Wave 15 — opt_level=5 per-recipe A/B results (ROLLBACK)

**Date**: 2026-04-23
**Branch**: `feat/wave15-tuning`
**Plan**: `docs/plans/2026-04-23-wave15-audit-driven-tuning.md`
**Measurement protocol**: QR dataset (OMNI_WEIGHTS_OVERRIDE=best_qr.pt, OMNI_COCO_YAML=qr_barcode.yaml, OMNI_CALIB_YAML=coco_val_only.yaml), `scripts/run_trt.py` native TRT path, warmup=100, measure=100, CUDA events
**Outcome**: **ROLLBACK all 3 measured recipes**. Wave 15 D2 delivers a negative result.

---

## 1. Headline

opt_level=5 on modelopt INT8 recipes (#09, #12, #42) on RTX 3060 Laptop **regresses performance** vs the opt_level=3 baseline. Every D2 recipe failed the accept criteria (fps +3% AND mAP -0.3%p AND build ≤ 1200s). Root cause: INT8 modelopt recipes already sit near the tactic-selection ceiling at opt_level=3; exhaustive autotune picks different bs=8 kernels that regress throughput while the bs=1 path stays flat. This is the opposite of what Wave 14 #40 (FP16) showed, because #40's FP16 baseline had 50%+ tactic headroom on bs=1 that INT8 recipes don't.

Net Wave 15 D2 impact on the recipe bank: **zero** — 3 baselines preserved, no regressions introduced. Wave 15's other deliverables (D1 no-regret switches, D3 build_ceiling_s schema, D4 tests) still ship.

---

## 2. D2 scope (vs plan)

| Recipe | In plan? | Measured? | Reason |
|---|---|---|---|
| `#05 trt_fp16` | listed | **SKIPPED** | `#40 trt_fp16_opt5` already encodes opt_level=5 FP16 (Wave 14). Modifying #05 in-place would duplicate #40 semantics. |
| `#09 modelopt_int8_entropy` | listed | ✅ | clean A/B (QR baseline fps 763.9) |
| `#12 modelopt_int8_mixed` | listed | ✅ | clean A/B (QR baseline fps 760.0) — note: shares ONNX cache with #09 due to pre-existing naming bug (§4) |
| `#17 modelopt_int8_qat` | listed | **DEFERRED** | requires 30-epoch QAT training before measurement; split to Wave 16 |
| `#42 modelopt_int8_asymmetric` | listed | ✅ | clean A/B (QR baseline fps 770.5, current TOP) |

3 of 5 planned recipes actually measured. #05 and #17 excluded with explicit structural reasons (not rollback).

---

## 3. Measured results (QR dataset, RTX 3060 Laptop)

| # | Recipe | opt | fps bs=1 | fps bs=8 | mAP50 | build_s | Δ fps bs=1 | Δ fps bs=8 | Δ mAP50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 09 | entropy           | 3 | 763.9 | 1078.5 | 0.98689 |   N/A | baseline | baseline | baseline |
| 09 | entropy           | 5 | 757.6 |  428.8 | 0.98689 | 445.6 | **-0.8%** | **-60.2%** | 0 |
| 12 | mixed             | 3 | 760.0 |  840.8 | 0.98689 |   N/A | baseline | baseline | baseline |
| 12 | mixed             | 5 | 742.6 |  429.3 | 0.98689 |  ~42  | **-2.3%** | **-48.9%** | 0 |
| 42 | asymmetric        | 3 | 770.5 | 1080.0 | 0.98689 |  18.1 | baseline | baseline | baseline |
| 42 | asymmetric        | 5 | 756.7 |  986.9 | 0.98689 |  48.0 | **-1.8%** | **-8.6%** | 0 |

mAP identical to 4 decimal places across opt_level variants — quantized weights + calibration scales are opt_level-independent (as expected — opt_level only changes kernel tactic selection, not Q/DQ numerics).

---

## 4. Accept / rollback per recipe

Criteria (per plan §4): fps delta ≥ +3% AND mAP delta ≥ -0.3%p AND build_time_s ≤ 1200.

| # | fps bs=1 gate | mAP gate | build gate | Verdict |
|---|---|---|---|---|
| 09 | ❌ -0.8% (want ≥ +3%) | ✅ | ✅ 446s ≤ 1200 | **ROLLBACK** |
| 12 | ❌ -2.3% | ✅ | ✅ | **ROLLBACK** |
| 42 | ❌ -1.8% | ✅ | ✅ 48s | **ROLLBACK** |

Rollback action taken: removed `builder_optimization_level: 5` and `build_ceiling_s: 1200` from the 3 recipe YAMLs. Baseline JSONs restored from `results_qr/_pre_wave15/` to `results_qr/`. Failed opt_level=5 engine files (`results/_engines/*_int8_opt5_modelopt_*.engine`) preserved as artifacts — delete manually if disk pressure.

---

## 5. Analysis — why Wave 14 #40 won but Wave 15 D2 lost

### Wave 14 #40 (FP16) baseline vs opt5
- #05 FP16 opt_level=3: bs1 **435.1**, bs8 **864.4**
- #40 FP16 opt_level=5: bs1 **645.2** (+48%), bs8 **460.9** (-47%)

### Wave 15 D2 pattern (INT8 modelopt)
- baseline bs1: 760-770 range (already 1.75x #05's FP16 baseline)
- opt_level=5 bs1: 742-758 (-1 to -2%)
- baseline bs8: 840-1080 (wide variance)
- opt_level=5 bs8: 429-987 (partially collapses for #09/#12, mild for #42)

**Mechanism**: `builder_optimization_level` controls TRT's tactic search space, not numeric precision. At opt_level=5 TRT explores tactics with different memory layouts, kernel fusion patterns, and SM occupancy tradeoffs. For small Conv-dominant YOLO26n on RTX 3060 Laptop (sm_86):
- FP16 at opt_level=3 was tactic-starved — many profitable tactics hadn't been explored. opt_level=5 found a bs1-winner that happened to trade bs8 throughput.
- INT8 modelopt at opt_level=3 was already near the Pareto frontier. opt_level=5 found "the bs1 winner shifts the Pareto tradeoff toward bs1-at-the-cost-of-bs8" — but because #09/#12/#42 already extracted the bs1 ceiling, the trade makes both worse.

**General rule this confirms**: opt_level=5 is **headroom-dependent**. It works when the baseline tactic selection has room to grow. It regresses when the model is already at its practical kernel ceiling. Without per-recipe A/B we'd have shipped a net bank regression.

### Why the bs8 collapse is so severe on #09/#12 specifically

#09 and #12 share the same quantized ONNX (`best_qr_640_modelopt_entropy_bs1.onnx`, see §6) and therefore got the same engine build tactics selected. The bs=8 engine at opt_level=5 apparently committed to a memory-bound batch-parallel kernel that is ~2.5x slower than the opt_level=3 choice. #42 (asymmetric) uses a different ONNX and landed a milder regression. Conclusion: the bs8 collapse isn't universal to INT8 + opt_level=5 — it's **bs=8 tactic selection on the specific quantized graph is unstable under exhaustive autotune on this GPU**.

---

## 6. ONNX cache caveat (pre-existing, documented not fixed)

`scripts/run_trt.py::_prepare_modelopt_onnx` (line 154-157) names the cached QDQ ONNX from `{calibrator}{asym?}{sparse?}` only — it does NOT incorporate `technique.nodes_to_exclude` into the cache key. Consequence:

- #09 (entropy, no excludes) generates `best_qr_640_modelopt_entropy_bs1.onnx`
- #12 (entropy, excludes Detect head CV2 branches) looks up the same cache key and gets #09's ONNX — so its configured excludes are silently ignored.

Evidence: both `results_qr/_pre_wave15/09_*.json` and `results_qr/_pre_wave15/12_*.json` report `mAP50=0.9868882125060877` (21 significant digits match). If excludes had applied to #12, the mAP would differ.

Scope decision: **not fixed in Wave 15.** The A/B for #12 is actually "entropy (no excludes) opt3 vs entropy (no excludes) opt5" which still cleanly isolates the opt_level variable. Fix candidate for Wave 16 alongside modelopt param schema expansion — see TODOS.md.

---

## 7. D3 (build_ceiling_s) validation

The plan raised each D2 recipe's `build_ceiling_s` to 1200. Actual build_time_s observed:

| # | build_time_s (opt5) | ceiling | within ceiling? |
|---|---:|---:|---|
| 09 | 445.6 | 1200 | ✅ |
| 12 | ~42 (engine cache hit on #09 build, new only for bs=8 engine re-run) | 1200 | ✅ |
| 42 | 48.0 | 1200 | ✅ |

D3 schema worked as designed — ceiling was diagnostic, no fatal path triggered. For Wave 16 reference: opt_level=5 on INT8 modelopt for YOLO26n takes 30-450s depending on cache state, well under the 1200s ceiling. The original 600s default would also have been sufficient; 1200 was precautionary.

Since the three recipes rolled back, `build_ceiling_s` is no longer set on any recipe in the bank — the schema field exists but is unused in practice. The MeasurementSpec field stays in case future Wave 16+ recipes opt into it (e.g., QAT-with-opt5, autotune-with-trtexec).

---

## 8. D2 verdict

**ROLLBACK: 3 / 3 measured recipes.** Wave 15 D2 is a negative result. The hypothesis "opt_level=5 generalizes from Wave 14 #40 FP16 to INT8 modelopt" is falsified.

Wave 15 ships **D1 + D3 + D4 + D5 only**:
- D1.1 OV CACHE_DIR (no-regret, accuracy zero-impact)
- D1.2 ORT TRT EP opt_level=5 + timing cache (backward-compat fallback; remains a gain candidate for #04 whose FP16 baseline has the same "headroom" property as Wave 14 #40)
- D3 `MeasurementSpec.build_ceiling_s` schema (enables future opt-in, no current user)
- D4 12 tests (179 suite total, 0 regressions)
- D5 plan doc + this results doc + recipes/README.md + TODOS.md

---

## 9. Artifacts

- **Recipe YAMLs**: 3 rolled back (#09, #12, #42) — see commit history on `feat/wave15-tuning`
- **Baselines archived**: `results_qr/_pre_wave15/{09,12,42}_*.json` (preserved for future retro)
- **Measured new JSONs (ROLLED BACK)**: overwritten baseline JSONs were restored from archive; the opt_level=5 measurement JSONs are discarded from `results_qr/` but the engine files remain at `results/_engines/best_qr_*_int8_opt5_modelopt_*.engine` as evidence
- **Run log**: `logs/wave15/d2_run.log`
- **Comparison helper**: `scripts/_compare_wave15.py` (Wave 15-specific, remove in Wave 16 cleanup)

---

## 10. Follow-ups (Wave 16 candidates)

1. **#04 ORT TRT EP re-measurement with D1.2 changes** — FP16 baseline has Wave 14-style headroom (fps 211, 3x below native #40's 645). Expected to benefit from opt_level=5 like #40 did. Measure and confirm.
2. **#17 QAT opt_level=5** — same caveat applies (INT8 near ceiling) but QAT variant hasn't been tested. Requires training pass first.
3. **Headroom-aware opt-in** — define a rule for when opt_level=5 is worth trying. Rough heuristic: "if baseline bs1 < 0.6x the best recipe's bs1, headroom exists; otherwise skip". Document in `recipes/README.md` under Wave 15 convention block.
4. **ONNX cache key bug** — add `nodes_to_exclude` hash to the cache filename tag (see §6). Quick win, blocks correct #12 behavior.
5. **bs=8 kernel selection investigation** — #09/#12 bs=8 collapse from 1078 to 429 is specific and unexplained at the tactic level. A focused spike using TRT profiling (`trtexec --verbose --profilingVerbosity=detailed`) would identify the regressed kernel and possibly surface a workaround (e.g., `BuilderFlag.REJECT_EMPTY_ALGORITHMS`, tiling hints).
6. **Revisit opt_level=5 via explicit A/B recipes** if headroom analysis (item 3) shows other recipes are candidates — use the per-recipe `_opt5.yaml` clone pattern (Option B from the original AskUserQuestion) instead of in-place modification, so A/B stays inspectable.

---

## 11. Lessons learned (capture for future Waves)

- **Measurement-first discipline works**. Without the accept/rollback gate, Wave 15 would have shipped a 3-recipe regression. The plan's explicit gates (fps +3%, mAP -0.3%p, build ≤ ceiling) paid for themselves on the first measurement cycle.
- **Don't generalize single-knob wins across tactics × precision × model families**. Wave 14 #40's +48% was FP16-specific and baseline-specific. Applying the same knob to a different precision family (INT8 modelopt) without A/B assumed generalization that didn't hold.
- **Headroom analysis belongs in the plan**. The Wave 15 plan should have included a "why we expect this to win" per recipe — Baseline fps vs "hypothetical ceiling for this dtype on this GPU". Without that, we were hoping rather than predicting.
- **Negative results are shippable deliverables**. This doc, the rollback commits, and the MeasurementSpec.build_ceiling_s schema together form a complete Wave 15 shipping artifact even though the "headline" change rolled back.
