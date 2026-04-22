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

When adding a recipe:

1. Pick the next unused number in the matching range.
2. The numeric prefix must match the runner that dispatches it (e.g. CPU
   recipes must start with `30+`, not slot into the `00-17` GPU range).
3. Add a `recipe-NN` target to `Makefile` and a per-file entry to
   `scripts/run_qr_batch.sh` / `scripts/run_coco_batch.sh` if the recipe
   should run in batch.
4. If archiving, move to `recipes/_archived/` rather than deleting. Do NOT
   reuse the number for a new recipe.
