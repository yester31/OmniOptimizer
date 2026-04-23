# Recipe Bank Numbering Convention

Recipes are numbered by backend family. Numbers never get reused after a recipe
is archived — they stay retired so result-JSON history keeps a stable identity.

| Range | Backend family | Runner |
|-------|----------------|--------|
| `00-17` | GPU — TRT native / PyTorch / ORT-CUDA / modelopt | `run_trt.py`, `run_pytorch.py`, `run_ort.py` |
| `18-19` | Reserved | — |
| `20-22` | **Retired** (brevitas, archived 2026-04-22) | — |
| `23-24` | GPU — modelopt FastNAS (pruning + INT8) | `run_trt.py` |
| `25-29` | Reserved | — |
| `30-35` | CPU — ORT CPU EP / OpenVINO | `run_cpu.py` |
| `36-38` | CPU — FastNAS (pruning) variants | `run_cpu.py` |
| `39` | Reserved | — |
| `40-49` | GPU — TRT tuning variants (Wave 14+ opt_level / dtype / zero-point knobs) | `run_trt.py` |

## Wave 15 convention — `builder_optimization_level` opt-in

Recipes may set `runtime.builder_optimization_level: 5` explicitly to request
exhaustive TRT tactic autotune. Leaving the field unset (default) selects
TRT's own default (3). Per-recipe opt-in is preferred over a global schema
default so `results/*.json` and `results_qr/*.json` retain MLPerf-style
reproducibility — the recipe file fully specifies the build configuration.

When opt-in, also raise `measurement.build_ceiling_s` above 600 (1200 is
a safe ceiling for INT8 + opt_level=5 on YOLO26n) so the diagnostic warning
only fires when a build genuinely hangs.

When adding a recipe:

1. Pick the next unused number in the matching range.
2. The numeric prefix must match the runner that dispatches it (e.g. CPU
   recipes must start with `30+`, not slot into the `00-17` GPU range).
3. Add a `recipe-NN` target to `Makefile` and a per-file entry to
   `scripts/run_qr_batch.sh` / `scripts/run_coco_batch.sh` if the recipe
   should run in batch.
4. If archiving, move to `recipes/_archived/` rather than deleting. Do NOT
   reuse the number for a new recipe.
