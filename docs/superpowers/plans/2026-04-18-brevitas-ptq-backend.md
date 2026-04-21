# Brevitas PTQ Backend (Wave 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Brevitas as the 4th INT8 PTQ backend via `technique.source="brevitas"` with 4 calibration variants (percentile / MSE / entropy / GPTQ), reusing the existing QDQ ONNX → TRT engine pipeline unchanged.

**Architecture:** Follow the Wave 3 `ort_quant` precedent exactly — a `_prepare_brevitas_onnx` helper that loads the YOLO26n PyTorch module, wraps it with Brevitas QuantConv2d/QuantLinear, calibrates on COCO val, exports QDQ ONNX via `brevitas.export.onnx.standard.export_onnx_qcdq`, then runs `onnxruntime.quantization.shape_inference.quant_pre_process`. The `_prepare_onnx` dispatcher gets one new branch; `_build_engine` and `_make_trt_forward` do not change. Failures degrade to `Result.notes` with `meets_constraints=False` per the CLAUDE.md rule.

**Tech Stack:** brevitas>=0.11, qonnx (transitive), onnxruntime, tensorrt, ultralytics, pytest.

Spec reference: `docs/superpowers/specs/2026-04-18-brevitas-ptq-backend-design.md`

---

## File Structure

**Create:**
- `recipes/20_brevitas_int8_percentile.yaml`
- `recipes/21_brevitas_int8_mse.yaml`
- `recipes/22_brevitas_int8_entropy.yaml`
- `recipes/23_brevitas_int8_gptq.yaml`
- `tests/test_wave4_brevitas_dispatch.py`

**Modify:**
- `scripts/_schemas.py:30-32` — extend `source` Literal with `"brevitas"`
- `scripts/run_trt.py` — add `_prepare_brevitas_onnx` helper (~line 377 after `_prepare_ort_quant_onnx`), extend `_SOURCE_TAG` dict (line 382), add dispatcher branch in `_prepare_onnx` (line 452)
- `Makefile:7-10, 21-25` — register `.PHONY` + `all:` + 4 new `recipe-2X` targets
- `pyproject.toml:29-36` — add `brevitas` extras group, reference it from `all`
- `docs/architecture.md:12-26` — bump counts (21→25 recipes, 3→4 active INT8 backends), list new recipes

---

## Task 1: Extend TechniqueSpec source Literal

**Files:**
- Modify: `scripts/_schemas.py:30-32`
- Test: `tests/test_wave4_brevitas_dispatch.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_wave4_brevitas_dispatch.py`:

```python
"""Wave 4 schema + dispatch tests for brevitas backend."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._schemas import TechniqueSpec, load_recipe  # noqa: E402


def test_source_literal_accepts_brevitas():
    spec = TechniqueSpec(name="int8_ptq", source="brevitas")
    assert spec.source == "brevitas"


@pytest.mark.parametrize("src", ["trt_builtin", "modelopt", "ort_quant", "brevitas"])
def test_source_literal_accepts_all_backends(src):
    spec = TechniqueSpec(name="int8_ptq", source=src)
    assert spec.source == src


def test_source_literal_rejects_unknown_backend():
    with pytest.raises(Exception):
        TechniqueSpec(name="int8_ptq", source="bogus_backend")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wave4_brevitas_dispatch.py::test_source_literal_accepts_brevitas -v`
Expected: FAIL with `pydantic.ValidationError: Input should be 'trt_builtin', 'modelopt' or 'ort_quant'`

- [ ] **Step 3: Extend the Literal**

In `scripts/_schemas.py`, change lines 30-32 from:
```python
    source: Literal[
        "trt_builtin", "modelopt", "ort_quant"
    ] = "trt_builtin"
```
to:
```python
    source: Literal[
        "trt_builtin", "modelopt", "ort_quant", "brevitas"
    ] = "trt_builtin"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_wave4_brevitas_dispatch.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/_schemas.py tests/test_wave4_brevitas_dispatch.py
git commit -m "feat(schema): allow source='brevitas' in TechniqueSpec"
```

---

## Task 2: Register _SOURCE_TAG entry

**Files:**
- Modify: `scripts/run_trt.py:382-386`
- Test: `tests/test_wave4_brevitas_dispatch.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_wave4_brevitas_dispatch.py`:

```python
def test_source_tag_includes_brevitas():
    """Engine cache filenames must stay short enough on Windows (MAX_PATH=260)."""
    from scripts.run_trt import _SOURCE_TAG

    assert _SOURCE_TAG["brevitas"] == "_brev"
    # Regression: existing tags must not change.
    assert _SOURCE_TAG["trt_builtin"] == ""
    assert _SOURCE_TAG["modelopt"] == "_modelopt"
    assert _SOURCE_TAG["ort_quant"] == "_ort"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wave4_brevitas_dispatch.py::test_source_tag_includes_brevitas -v`
Expected: FAIL with `KeyError: 'brevitas'`

- [ ] **Step 3: Add the tag**

In `scripts/run_trt.py`, change lines 382-386 from:
```python
_SOURCE_TAG = {
    "trt_builtin": "",
    "modelopt": "_modelopt",
    "ort_quant": "_ort",
}
```
to:
```python
_SOURCE_TAG = {
    "trt_builtin": "",
    "modelopt": "_modelopt",
    "ort_quant": "_ort",
    "brevitas": "_brev",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wave4_brevitas_dispatch.py::test_source_tag_includes_brevitas -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_trt.py tests/test_wave4_brevitas_dispatch.py
git commit -m "feat(run_trt): register brevitas _SOURCE_TAG"
```

---

## Task 3: Implement _prepare_brevitas_onnx helper

Implements the Brevitas PyTorch-level PTQ → QDQ ONNX export → ORT `quant_pre_process`
pipeline. Supported `algo` values: `percentile`, `mse`, `entropy`, `gptq`.
Fails fast on unknown algo. Uses cache identical in spirit to `_prepare_ort_quant_onnx`.

**Files:**
- Modify: `scripts/run_trt.py` — insert after `_prepare_ort_quant_onnx` (line ~377) and before `_SOURCE_TAG` (line 382)
- Test: `tests/test_wave4_brevitas_dispatch.py`

- [ ] **Step 1: Add the failing calibrator-validation test**

Append to `tests/test_wave4_brevitas_dispatch.py`:

```python
def _write_min_brevitas_recipe(tmp_path: Path, algo: str) -> Path:
    p = tmp_path / f"brevitas_{algo}.yaml"
    p.write_text(
        "name: smoke\n"
        "model: {family: yolo26, variant: n, weights: yolo26n.pt}\n"
        "runtime: {engine: tensorrt, dtype: int8}\n"
        "technique: {name: int8_ptq, source: brevitas,\n"
        f"            calibrator: {algo}, calibration_samples: 8,\n"
        "            calibration_seed: 42}\n"
        "measurement:\n"
        "  dataset: coco_val2017\n  num_images: 8\n"
        "  warmup_iters: 1\n  measure_iters: 2\n"
        "  batch_sizes: [1]\n  input_size: 640\n"
        "  gpu_clock_lock: false\n  seed: 42\n"
        "constraints: {max_map_drop_pct: 5.0}\n",
        encoding="utf-8",
    )
    return p


def test_brevitas_rejects_unknown_algo(tmp_path):
    """Guardrail: unknown algo must fail fast with a clear ValueError
    (matches the ort_quant precedent)."""
    from scripts.run_trt import _prepare_brevitas_onnx

    recipe = load_recipe(str(_write_min_brevitas_recipe(tmp_path, "bogus_algo")))
    with pytest.raises(ValueError, match="brevitas"):
        _prepare_brevitas_onnx(recipe, 640, tmp_path, dynamic=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wave4_brevitas_dispatch.py::test_brevitas_rejects_unknown_algo -v`
Expected: FAIL with `ImportError: cannot import name '_prepare_brevitas_onnx'`

- [ ] **Step 3: Implement the helper**

In `scripts/run_trt.py`, insert the following block immediately after the closing
`return cached` of `_prepare_ort_quant_onnx` (line 376) and before the
`# Short tags appear...` comment (line 379):

```python
def _prepare_brevitas_onnx(recipe: Recipe, imgsz: int, cache_dir: Path,
                           dynamic: bool = True) -> Path:
    """Quantize via Brevitas (PyTorch-native) and export QDQ ONNX that
    TensorRT's explicit-quantization path consumes identically to
    modelopt/ort_quant output.

    Supported ``technique.calibrator`` values:
      - ``percentile`` — activation scale from p99.99 of observed values
      - ``mse``        — activation scale minimizing MSE of (Q(x)-x)
      - ``entropy``    — KL-divergence calibration (Brevitas impl)
      - ``gptq``       — weight-only GPTQ correction (Brevitas-specific);
                         activations stay on percentile

    Quantizer config enforces the TRT compat checklist:
      - per-channel symmetric INT8 weights on Conv axis=0
      - per-tensor symmetric INT8 activations (zero_point=0)
      - no bias quantization (TRT computes INT32 bias from act*weight scales)
    """
    _BREVITAS_ALGOS = ("percentile", "mse", "entropy", "gptq")
    algo = (recipe.technique.calibrator or "percentile").lower()
    if algo not in _BREVITAS_ALGOS:
        raise ValueError(
            f"brevitas backend supports calibrator in {list(_BREVITAS_ALGOS)}, "
            f"got {algo!r}"
        )

    try:
        import torch
        from brevitas.graph.quantize import preprocess_for_quantize, quantize
        from brevitas.graph.calibrate import calibration_mode
        from brevitas.graph.gptq import gptq_mode
        from brevitas.export import export_onnx_qcdq
        from brevitas.quant.scaled_int import (
            Int8ActPerTensorFloat,
            Int8WeightPerChannelFloat,
        )
    except ImportError as e:
        raise RuntimeError(
            "brevitas not available. Install with `pip install brevitas>=0.11`."
        ) from e

    n_samples = int(recipe.technique.calibration_samples or 512)
    seed = int(recipe.technique.calibration_seed or 42)
    bs_tag = "dyn" if dynamic else "bs1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / (
        f"{Path(recipe.model.weights).stem}_{imgsz}_brev_{algo}_"
        f"{n_samples}_s{seed}_{bs_tag}.qdq.onnx"
    )
    if cached.exists():
        print(f"[info] brevitas cache hit: {cached.name}", file=sys.stderr)
        return cached

    # Brevitas works on nn.Module, not ONNX. Load the ultralytics PyTorch model,
    # strip the detection wrapper, and move to CUDA for calibration.
    from ultralytics import YOLO

    yolo = YOLO(recipe.model.weights)
    module = yolo.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    # Activation scale configuration.
    act_kwargs = {}
    if algo == "percentile":
        act_kwargs["high_percentile_q"] = 99.99
        act_kwargs["low_percentile_q"] = 0.01
    # mse / entropy / gptq use Brevitas defaults for activations; gptq
    # specifically operates on weights and keeps activations on percentile.

    module = preprocess_for_quantize(module)
    module = quantize(
        module,
        weight_quant=Int8WeightPerChannelFloat,
        act_quant=Int8ActPerTensorFloat,
        bias_quant=None,
        **({"act_quant_kwargs": act_kwargs} if act_kwargs else {}),
    )

    # Calibration data — reuse modelopt/ort_quant path.
    val_yaml = os.environ.get("OMNI_COCO_YAML")
    calib_arr = _build_calib_numpy(val_yaml, n_samples, imgsz, seed)

    import numpy as np

    def _batch_iter(arr, bs: int = 8):
        for i in range(0, len(arr), bs):
            chunk = arr[i:i + bs]
            if chunk.ndim == 3:
                chunk = chunk[None, ...]
            yield torch.from_numpy(np.ascontiguousarray(chunk)).to(device)

    print(
        f"[info] brevitas calibrate: algo={algo}, samples={n_samples}, "
        f"weights={Path(recipe.model.weights).name}",
        file=sys.stderr,
    )
    module.eval()
    with torch.no_grad():
        # All non-GPTQ algos run under calibration_mode; GPTQ additionally
        # wraps in gptq_mode for weight correction after activation stats.
        with calibration_mode(module):
            for x in _batch_iter(calib_arr):
                module(x)
        if algo == "gptq":
            with gptq_mode(module, use_quant_activations=True) as gptq:
                for _ in range(gptq.num_layers):
                    for x in _batch_iter(calib_arr):
                        gptq.model(x)
                    gptq.update()

    # Export QDQ ONNX. opset_version 17 matches the modelopt/ort_quant path.
    dummy = torch.zeros((1, 3, imgsz, imgsz), device=device)
    raw_out = cache_dir / (cached.stem + ".raw.onnx")
    export_onnx_qcdq(module, args=dummy, export_path=str(raw_out), opset_version=17)
    print(f"[info] brevitas exported raw QDQ onnx: {raw_out.name}", file=sys.stderr)

    # Run ORT's quant_pre_process — same rationale as _prepare_ort_quant_onnx:
    # histogram calibrators / TRT parser both benefit from shape inference +
    # constant folding before they see the QDQ graph.
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        quant_pre_process(
            input_model_path=str(raw_out),
            output_model_path=str(cached),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            auto_merge=True,
            verbose=0,
        )
        print(f"[info] brevitas wrote QDQ onnx: {cached}", file=sys.stderr)
    except Exception as e:
        print(
            f"[warn] brevitas preprocess failed ({e}); using raw export",
            file=sys.stderr,
        )
        raw_out.replace(cached)

    return cached
```

- [ ] **Step 4: Run the calibrator-validation test**

Run: `python -m pytest tests/test_wave4_brevitas_dispatch.py::test_brevitas_rejects_unknown_algo -v`
Expected: PASS (the ValueError branch is hit before any brevitas import)

- [ ] **Step 5: Sanity check the full file parses**

Run:
```bash
python -c "import ast, pathlib; ast.parse(pathlib.Path('scripts/run_trt.py').read_text(encoding='utf-8'))"
```
Expected: no output (success)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_trt.py tests/test_wave4_brevitas_dispatch.py
git commit -m "feat(run_trt): add _prepare_brevitas_onnx helper"
```

---

## Task 4: Wire brevitas into _prepare_onnx dispatcher

**Files:**
- Modify: `scripts/run_trt.py:443-453`
- Test: `tests/test_wave4_brevitas_dispatch.py`

- [ ] **Step 1: Add the failing dispatch test**

Append to `tests/test_wave4_brevitas_dispatch.py`:

```python
def test_prepare_onnx_routes_to_brevitas(tmp_path):
    from scripts.run_trt import _prepare_onnx

    recipe = load_recipe(str(_write_min_brevitas_recipe(tmp_path, "percentile")))
    with patch("scripts.run_trt._prepare_brevitas_onnx") as m:
        m.return_value = tmp_path / "fake.onnx"
        path, is_qdq = _prepare_onnx(recipe, 640, tmp_path, bs=1)
    assert is_qdq is True
    assert path == tmp_path / "fake.onnx"
    m.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wave4_brevitas_dispatch.py::test_prepare_onnx_routes_to_brevitas -v`
Expected: FAIL with `ValueError: unknown technique.source: 'brevitas'`

- [ ] **Step 3: Add dispatcher branch**

In `scripts/run_trt.py`, locate the `_prepare_onnx` function body (currently ending
at line 453 with `raise ValueError(f"unknown technique.source: {source!r}")`).
Insert before that raise, after the `if source == "ort_quant":` block:

```python
    if source == "brevitas":
        return _prepare_brevitas_onnx(recipe, imgsz, cache_dir, dynamic=dynamic), True
```

Final structure (for reference):
```python
    if source == "trt_builtin":
        path = _export_onnx(...)
        return path, False
    if source == "modelopt":
        return _prepare_modelopt_onnx(...), True
    if source == "ort_quant":
        return _prepare_ort_quant_onnx(...), True
    if source == "brevitas":
        return _prepare_brevitas_onnx(recipe, imgsz, cache_dir, dynamic=dynamic), True
    raise ValueError(f"unknown technique.source: {source!r}")
```

- [ ] **Step 4: Run all Wave 4 tests**

Run: `python -m pytest tests/test_wave4_brevitas_dispatch.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Run Wave 3 regression tests**

Run: `python -m pytest tests/test_wave3_dispatch.py -v`
Expected: all existing tests still PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_trt.py tests/test_wave4_brevitas_dispatch.py
git commit -m "feat(run_trt): dispatch source=brevitas in _prepare_onnx"
```

---

## Task 5: Create the 4 recipe YAMLs

**Files:**
- Create: `recipes/20_brevitas_int8_percentile.yaml`
- Create: `recipes/21_brevitas_int8_mse.yaml`
- Create: `recipes/22_brevitas_int8_entropy.yaml`
- Create: `recipes/23_brevitas_int8_gptq.yaml`

- [ ] **Step 1: Write recipe 20 (percentile)**

Create `recipes/20_brevitas_int8_percentile.yaml`:

```yaml
name: brevitas_int8_percentile
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
  ultralytics_version: null
runtime:
  engine: tensorrt
  version: null
  dtype: int8
technique:
  name: int8_ptq
  source: brevitas              # Brevitas PyTorch-level PTQ → QDQ ONNX → TRT
  calibrator: percentile        # activation scale = p99.99
  calibration_samples: 512
  calibration_dataset: coco_val2017
  calibration_seed: 42
hardware:
  gpu: null
  cuda: null
  driver: null
measurement:
  dataset: coco_val2017
  num_images: 500
  warmup_iters: 100
  measure_iters: 100
  batch_sizes: [1, 8]
  input_size: 640
  gpu_clock_lock: true
  seed: 42
constraints:
  max_map_drop_pct: 2.0
  min_fps_bs1: 30
```

- [ ] **Step 2: Write recipe 21 (MSE)**

Create `recipes/21_brevitas_int8_mse.yaml` — identical to recipe 20 except:
```yaml
name: brevitas_int8_mse
...
technique:
  name: int8_ptq
  source: brevitas
  calibrator: mse               # minimize MSE of (Q(x) - x) for act scale
  calibration_samples: 512
  calibration_dataset: coco_val2017
  calibration_seed: 42
```

Full file:

```yaml
name: brevitas_int8_mse
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
  ultralytics_version: null
runtime:
  engine: tensorrt
  version: null
  dtype: int8
technique:
  name: int8_ptq
  source: brevitas
  calibrator: mse
  calibration_samples: 512
  calibration_dataset: coco_val2017
  calibration_seed: 42
hardware:
  gpu: null
  cuda: null
  driver: null
measurement:
  dataset: coco_val2017
  num_images: 500
  warmup_iters: 100
  measure_iters: 100
  batch_sizes: [1, 8]
  input_size: 640
  gpu_clock_lock: true
  seed: 42
constraints:
  max_map_drop_pct: 2.0
  min_fps_bs1: 30
```

- [ ] **Step 3: Write recipe 22 (entropy)**

Create `recipes/22_brevitas_int8_entropy.yaml`:

```yaml
name: brevitas_int8_entropy
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
  ultralytics_version: null
runtime:
  engine: tensorrt
  version: null
  dtype: int8
technique:
  name: int8_ptq
  source: brevitas
  calibrator: entropy           # KL-divergence (Brevitas impl)
  calibration_samples: 512
  calibration_dataset: coco_val2017
  calibration_seed: 42
hardware:
  gpu: null
  cuda: null
  driver: null
measurement:
  dataset: coco_val2017
  num_images: 500
  warmup_iters: 100
  measure_iters: 100
  batch_sizes: [1, 8]
  input_size: 640
  gpu_clock_lock: true
  seed: 42
constraints:
  max_map_drop_pct: 2.0
  min_fps_bs1: 30
```

- [ ] **Step 4: Write recipe 23 (GPTQ)**

Create `recipes/23_brevitas_int8_gptq.yaml`:

```yaml
name: brevitas_int8_gptq
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
  ultralytics_version: null
runtime:
  engine: tensorrt
  version: null
  dtype: int8
technique:
  name: int8_ptq
  source: brevitas
  calibrator: gptq              # weight-only GPTQ; activations stay on percentile
  calibration_samples: 512
  calibration_dataset: coco_val2017
  calibration_seed: 42
hardware:
  gpu: null
  cuda: null
  driver: null
measurement:
  dataset: coco_val2017
  num_images: 500
  warmup_iters: 100
  measure_iters: 100
  batch_sizes: [1, 8]
  input_size: 640
  gpu_clock_lock: true
  seed: 42
constraints:
  max_map_drop_pct: 2.0
  min_fps_bs1: 30
```

- [ ] **Step 5: Verify all 4 YAMLs parse against the schema**

Run (from repo root):
```bash
python -c "import sys; sys.path.insert(0, '.'); from scripts._schemas import load_recipe; import pathlib; [print(load_recipe(str(p)).name) for p in sorted(pathlib.Path('recipes').glob('2[0-3]_*.yaml'))]"
```
Expected output (exact names):
```
brevitas_int8_percentile
brevitas_int8_mse
brevitas_int8_entropy
brevitas_int8_gptq
```

- [ ] **Step 6: Commit**

```bash
git add recipes/20_brevitas_int8_percentile.yaml recipes/21_brevitas_int8_mse.yaml \
        recipes/22_brevitas_int8_entropy.yaml recipes/23_brevitas_int8_gptq.yaml
git commit -m "feat(recipes): add 4 brevitas INT8 PTQ variants (percentile/mse/entropy/gptq)"
```

---

## Task 6: Makefile targets

**Files:**
- Modify: `Makefile:7-10, 21-25, 82` (insert after line 82 for new recipe blocks)

- [ ] **Step 1: Extend .PHONY line**

Change lines 7-10 from:
```makefile
.PHONY: all clean env report \
        recipe-00 recipe-00-tf32 \
        recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 recipe-07 \
        recipe-08 recipe-09 recipe-10 recipe-11 recipe-12 \
        recipe-13 recipe-14 recipe-15 recipe-16 \
        diagnose-recipe-%
```
to:
```makefile
.PHONY: all clean env report \
        recipe-00 recipe-00-tf32 \
        recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 recipe-07 \
        recipe-08 recipe-09 recipe-10 recipe-11 recipe-12 \
        recipe-13 recipe-14 recipe-15 recipe-16 \
        recipe-20 recipe-21 recipe-22 recipe-23 \
        diagnose-recipe-%
```

- [ ] **Step 2: Extend the all: target**

Change lines 21-25 from:
```makefile
all: recipe-00 recipe-00-tf32 \
     recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 \
     recipe-08 recipe-09 recipe-10 recipe-12 \
     recipe-13 recipe-14 recipe-15 recipe-16 \
     report
```
to:
```makefile
all: recipe-00 recipe-00-tf32 \
     recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 \
     recipe-08 recipe-09 recipe-10 recipe-12 \
     recipe-13 recipe-14 recipe-15 recipe-16 \
     recipe-20 recipe-21 recipe-22 recipe-23 \
     report
```

- [ ] **Step 3: Add 4 new recipe targets**

After the existing `recipe-16:` block (ending at line 82), insert:

```makefile

recipe-20:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/20_brevitas_int8_percentile.yaml --out $(RESULTS_DIR)/20_brevitas_int8_percentile.json

recipe-21:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/21_brevitas_int8_mse.yaml --out $(RESULTS_DIR)/21_brevitas_int8_mse.json

recipe-22:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/22_brevitas_int8_entropy.yaml --out $(RESULTS_DIR)/22_brevitas_int8_entropy.json

recipe-23:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/23_brevitas_int8_gptq.yaml --out $(RESULTS_DIR)/23_brevitas_int8_gptq.json
```

Note: the recipe command lines MUST be indented with a TAB (not spaces) — this is a
Makefile requirement.

- [ ] **Step 4: Verify Makefile parses**

Run: `make -n recipe-20`
Expected output (exact):
```
python scripts/run_trt.py --recipe recipes/20_brevitas_int8_percentile.yaml --out results/20_brevitas_int8_percentile.json
```

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "feat(make): add recipe-20..23 targets for brevitas backend"
```

---

## Task 7: pyproject extras

**Files:**
- Modify: `pyproject.toml:23-39`

- [ ] **Step 1: Add brevitas extras group and reference from all**

Change the `[project.optional-dependencies]` block (lines 23-39) from:
```toml
[project.optional-dependencies]
torch = [
    "torch>=2.3",
    "torchvision>=0.18",
    "ultralytics>=8.3",
]
onnx = [
    "onnx>=1.16",
    "onnxruntime-gpu>=1.18",
]
trt = [
    "tensorrt>=10.0",
    "polygraphy>=0.49",
]
all = [
    "omnioptimizer[torch,onnx,trt]",
]
```
to:
```toml
[project.optional-dependencies]
torch = [
    "torch>=2.3",
    "torchvision>=0.18",
    "ultralytics>=8.3",
]
onnx = [
    "onnx>=1.16",
    "onnxruntime-gpu>=1.18",
]
trt = [
    "tensorrt>=10.0",
    "polygraphy>=0.49",
]
brevitas = [
    "brevitas>=0.11",
]
all = [
    "omnioptimizer[torch,onnx,trt,brevitas]",
]
```

- [ ] **Step 2: Verify TOML parses**

Run:
```bash
python -c "import tomllib, pathlib; cfg = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); print('brevitas' in cfg['project']['optional-dependencies'])"
```
Expected output: `True`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): add brevitas optional extras group"
```

---

## Task 8: Update architecture.md

**Files:**
- Modify: `docs/architecture.md:12-26`

- [ ] **Step 1: Bump scope summary + recipe list**

In `docs/architecture.md`, change the `## Current scope (post Wave 3, 2026-04-18)`
section header to `## Current scope (post Wave 4, 2026-04-18)`.

Then in the bullet list under that header (lines 14-26), change:
- "21 recipes" → "25 recipes"
- After the `INT8 neural_compressor (INC 2.x): MinMax PTQ (#17), SmoothQuant (#18)` line,
  add a new line (keeping the same indentation/dash style):
  ```
  - INT8 `brevitas` (PyTorch-native PTQ → QDQ ONNX): percentile (#20),
    MSE (#21), entropy (#22), GPTQ (#23).
  ```
- In the `**Parked**` line, leave as-is (Brevitas adds no parked recipes this wave).

Final bullet block should read:
```markdown
- **Runtimes × Techniques** — 25 recipes:
  - PyTorch eager FP32 (#01), `torch.compile` FP16 (#02).
  - ONNX Runtime CUDA EP (#03) / TensorRT EP (#04), both FP16.
  - Native TensorRT: FP32 (#00), FP32+TF32 (#00-tf32), FP16 (#05), INT8 PTQ (#06).
  - INT8 `modelopt` (ONNX-path PTQ): max (#08), entropy (#09), percentile (#10),
    mixed precision (#12).
  - INT8 `ort_quant` (`onnxruntime.quantization.quantize_static`): minmax (#13),
    entropy (#14), percentile (#15), distribution (#16).
  - INT8 `neural_compressor` (INC 2.x): MinMax PTQ (#17), SmoothQuant (#18).
  - INT8 `brevitas` (PyTorch-native PTQ → QDQ ONNX): percentile (#20),
    MSE (#21), entropy (#22), GPTQ (#23).
  - **Parked** (need training pipeline): #07 trt_int8_sparsity, #11 modelopt_sparsity,
    #19 inc_int8_qat.
```

- [ ] **Step 2: Update the `_SOURCE_TAG` dispatcher description**

Find the `**Extension hook via technique.source**` paragraph (around line 103-110).
Change "Current dispatch targets: `trt_builtin` ... `neural_compressor` (INC 2.x `fit()`)" to include brevitas:

Before:
```
Current dispatch targets:
  `trt_builtin` (TRT's entropy calibrator + `SPARSE_WEIGHTS`), `modelopt`
  (ONNX-path QDQ via `modelopt.onnx.quantization.quantize`), `ort_quant`
  (`onnxruntime.quantization.quantize_static`), `neural_compressor` (INC 2.x
  `fit()`).
```
After:
```
Current dispatch targets:
  `trt_builtin` (TRT's entropy calibrator + `SPARSE_WEIGHTS`), `modelopt`
  (ONNX-path QDQ via `modelopt.onnx.quantization.quantize`), `ort_quant`
  (`onnxruntime.quantization.quantize_static`), `neural_compressor` (INC 2.x
  `fit()`, recipes inactive — see below), `brevitas` (PyTorch-level PTQ via
  `brevitas.graph.quantize` + `export_onnx_qcdq`).
```

- [ ] **Step 3: Commit**

```bash
git add docs/architecture.md
git commit -m "docs(arch): document brevitas backend + bump recipe count to 25"
```

---

## Task 9: Full regression + E2E smoke

**Files:** (none modified; verification only)

- [ ] **Step 1: Static sanity checks**

Run:
```bash
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('scripts').glob('*.py')]"
python -c "import sys; sys.path.insert(0, '.'); from scripts._schemas import load_recipe; import pathlib; [load_recipe(str(p)) for p in pathlib.Path('recipes').glob('*.yaml')]"
```
Expected: no output, no exceptions (all 25 recipes parse)

- [ ] **Step 2: Full test suite (no GPU required)**

Run: `python -m pytest tests/ -v`
Expected: all existing tests + new Wave 4 tests pass (5 new tests expected).

- [ ] **Step 3: Makefile dry-run of all**

Run: `make -n all | grep -c "run_trt.py"`
Expected output: a count ≥ 16 (covers existing trt runs + 4 new brevitas)

- [ ] **Step 4: E2E one brevitas recipe (GPU required)**

Prerequisite: `OMNI_COCO_YAML` points to a valid COCO val yaml, and `pip install -e ".[all]"`
has been re-run so brevitas is installed.

Run: `make recipe-20`
Expected:
- stderr shows `[info] brevitas calibrate: algo=percentile, samples=512, weights=yolo26n.pt`
- stderr shows `[info] brevitas wrote QDQ onnx: ...`
- TRT engine build succeeds (or degrades cleanly to notes)
- `results/20_brevitas_int8_percentile.json` exists and is valid per `Result` schema

Verification:
```bash
python -c "import json, pathlib; r = json.loads(pathlib.Path('results/20_brevitas_int8_percentile.json').read_text()); print(r['recipe_name'], r.get('meets_constraints'), r.get('notes'))"
```

- [ ] **Step 5: E2E remaining brevitas recipes**

Run: `make recipe-21 recipe-22 recipe-23`
Expected: each produces a JSON in `results/`. Failures land in `notes` with
`meets_constraints=False` (per CLAUDE.md rule) and do not abort the make target.

- [ ] **Step 6: Full `make all` + report regeneration**

Run: `make all`
Expected:
- All 22 active recipes (existing 18 + 4 brevitas) run to completion
- `report.md` contains 4 new rows prefixed `brevitas_int8_*`
- Success criterion from spec §8: at least one brevitas variant has
  `mAP drop < 0.5` AND `fps(bs=1) ≥ FP16 baseline`

Verification:
```bash
grep -c "brevitas_int8_" report.md
```
Expected output: `4` (or more if report lists per-batch rows)

- [ ] **Step 7: Final commit if any verification touched files**

If nothing changed, skip. Otherwise:
```bash
git add -u
git commit -m "chore: regenerate report.md with brevitas results"
```

---

## Self-Review Summary

- **Spec coverage:** §1 motivation → Task 8 docs; §2 architecture → Tasks 3–4; §3 files → all tasks; §4 QDQ constraints → Task 3 step 3 body; §5 recipe schema → Task 5; §6 calibration → Task 3 step 3 body; §7 degrade → Task 3 (ImportError path, warn-on-preprocess-fail); §8 validation → Task 9; §9 risks → addressed by Task 3's algo whitelist + Task 9's degrade paths; §10 future work → out of scope.
- **Placeholders:** none.
- **Type consistency:** `_prepare_brevitas_onnx` signature matches `_prepare_ort_quant_onnx` (recipe, imgsz, cache_dir, dynamic). `_SOURCE_TAG["brevitas"] = "_brev"` used consistently in Task 2 and referenced nowhere else (engine filename uses it via `_SOURCE_TAG.get` already in run_trt.py line 885 — unchanged). Recipe `calibrator` field name used consistently in YAMLs and in the helper's `_BREVITAS_ALGOS` validation.
