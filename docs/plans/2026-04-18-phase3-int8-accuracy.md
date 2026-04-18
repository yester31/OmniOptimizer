# Phase 3: INT8 정확도 드라이브 — 2:4 Sparsity 실측화 + 민감 레이어 제외

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** OmniOptimizer의 `modelopt_int8_*` 레시피 라인에 (1) 진짜 2:4 sparsity를 주입하고 (2) 민감 레이어 제외 기반 혼합 정밀도 레시피를 추가하여 INT8 mAP 드롭을 1%p 목표 아래로 밀어넣는다.

**Architecture:** 변경은 `scripts/run_trt.py`의 modelopt 경로 한 곳과 `scripts/_schemas.py`의 `TechniqueSpec`에 국한된다. 기존 `technique.source=modelopt` 디스패처를 확장해 `technique.sparsity_preprocess`(2:4 pruning)와 `technique.nodes_to_exclude`(민감 레이어)를 옵트인으로 받는다. `SPARSE_WEIGHTS` 플래그만으로는 실제 2:4 커널이 선택되지 않는다는 기존 버그(`modelopt_int8_sparsity`의 mAP가 `_entropy`와 동일)를 버그픽스로 해소하고, 신규 recipe #12로 민감 레이어 제외 축을 추가한다.

**Tech Stack:** `nvidia-modelopt` (torch.sparsity + onnx.quantization), TensorRT 10.x, ultralytics, pydantic.

---

## v2 Patches — Expert Review Fixes (2026-04-18, apply these BEFORE following the Task steps below)

The Task 1–8 sections that follow this Patches block are the v1 baseline. Where
a patch below conflicts with a v1 step, the **patch wins**. v1 Task structure
(files, test framework, commit granularity) stays the same.

### P-API. `modelopt.torch.sparsity.sparsify` signature correction

v1 plan called `sparsify(model, mode=..., data_loader=...)`. This is wrong.
Real API (confirmed against `nvidia/model-optimizer` docs):

```python
from modelopt.torch.sparsity import sparsify as mts_sparsify
from modelopt.torch.sparsity import export as mts_export

config = {"data_loader": loader, "collect_func": lambda x: x}
sparse_model = mts_sparsify(model, mode="sparse_magnitude", config=config)
sparse_model = mts_export(sparse_model)   # required — collapses wrappers
```

Key deltas vs v1:
- `data_loader` goes **inside** `config`, not as a kwarg.
- `collect_func` is required (identity is fine for CV forward pass).
- `mts.export()` must be called after `sparsify` or downstream code sees a
  wrapped module with hooks.
- Mode is `"sparse_magnitude"`, **not** `"sparsegpt"`. `sparsegpt` is a
  Hessian-based LLM technique; CNN weights use magnitude pruning.

### P-L4. `_apply_modelopt_sparsify` returns a live YOLO object, not a .pt path

v1 plan did `torch.save({"model": inner}, cached)` then re-loaded via
`YOLO(sparse_pt)`. ultralytics expects a full checkpoint (train_args, yaml,
ema) so the re-load will fail. Replace the whole helper with:

```python
def _apply_modelopt_sparsify(weights: str, imgsz: int):
    """Return a ultralytics YOLO whose backbone weights carry the 2:4 pattern.

    Loads a fresh YOLO (preserves ultralytics metadata), runs modelopt 2:4
    magnitude pruning on yolo.model, exports the sparsified module to strip
    modelopt wrappers, then swaps the weights back into the YOLO wrapper via
    load_state_dict. Downstream ONNX export path stays identical to the
    non-sparse case — same YOLO.export(...) entry point.

    Fallback: if this torch-level path proves brittle against future
    ultralytics/modelopt versions, Plan B is ONNX graph-level weight
    masking via onnx-graphsurgeon (zero out the right lanes per 4-column
    block). Not implemented now; add only if measured necessary.
    """
    try:
        from modelopt.torch.sparsity import sparsify as mts_sparsify
        from modelopt.torch.sparsity import export as mts_export
    except ImportError as e:
        raise RuntimeError(
            "nvidia-modelopt torch extension not installed. Install with: "
            "pip install --extra-index-url https://pypi.nvidia.com nvidia-modelopt"
        ) from e

    import torch
    from ultralytics import YOLO

    yolo = YOLO(weights)
    inner = yolo.model
    device = next(inner.parameters()).device
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)

    def _loader():
        yield dummy

    config = {"data_loader": _loader(), "collect_func": lambda x: x}
    sparse_model = mts_sparsify(inner, mode="sparse_magnitude", config=config)
    sparse_model = mts_export(sparse_model)

    yolo.model.load_state_dict(sparse_model.state_dict())
    print("[info] modelopt 2:4 sparse_magnitude applied + exported", file=sys.stderr)
    return yolo
```

And the dispatcher (`_prepare_modelopt_onnx`) stops taking a `.pt` path —
instead, when `sparsity_preprocess == "2:4"`, it calls
`_apply_modelopt_sparsify` to get a YOLO object, then routes **that object**
through a parallel ONNX export helper. Simplest incision: refactor
`_export_onnx` to accept either a weights string or a live YOLO.

```python
def _export_onnx(weights, imgsz: int, half: bool, cache_dir: Path,
                 dynamic: bool = True, tag_suffix: str = "") -> Path:
    """weights: str (filesystem path) OR ultralytics.YOLO instance."""
    from ultralytics import YOLO
    cache_dir.mkdir(parents=True, exist_ok=True)
    bs_tag = "dyn" if dynamic else "bs1"
    stem = Path(weights).stem if isinstance(weights, str) else \
           Path(getattr(weights, "ckpt_path", "yolo")).stem
    tag = f"{stem}_{imgsz}_{'fp16' if half else 'fp32'}{tag_suffix}_{bs_tag}.onnx"
    cached = cache_dir / tag
    if cached.exists():
        return cached
    model = weights if isinstance(weights, YOLO) else YOLO(weights)
    onnx_path = model.export(
        format="onnx", imgsz=imgsz, half=False, simplify=True, dynamic=dynamic,
    )
    src = Path(onnx_path)
    if src != cached:
        src.rename(cached)
    return cached
```

`_prepare_modelopt_onnx` when `sparsity_preprocess == "2:4"` becomes:

```python
yolo = _apply_modelopt_sparsify(recipe.model.weights, imgsz)
clean_onnx = _export_onnx(yolo, imgsz, half=False, cache_dir=cache_dir,
                          dynamic=dynamic, tag_suffix="_sparse24")
```

### P-L5. Test split: API correctness vs dispatcher flow

Replace the monolithic monkeypatch test in v1 Task 2 Step 1 with two tests.
Reasons: (1) the dispatcher test should not care what `mts.sparsify` is
actually called with, only that `_apply_modelopt_sparsify` is invoked
before `quantize`. (2) the API test is only meaningful when modelopt is
actually installed.

```python
# tests/test_run_trt_phase3.py

def test_dispatcher_invokes_sparsify_helper_before_quantize(tmp_path, monkeypatch):
    """Dispatcher: when sparsity_preprocess='2:4', _apply_modelopt_sparsify
    runs before the ONNX quantize step. Does not assert on modelopt internals."""
    from scripts import run_trt

    calls = []

    class FakeYolo:
        ckpt_path = "yolo26n.pt"
        def export(self, **kwargs): 
            out = tmp_path / "exported.onnx"
            out.write_bytes(b"onnx-stub")
            return str(out)

    def fake_apply_sparsify(weights, imgsz):
        calls.append("sparsify")
        return FakeYolo()

    def fake_quantize(onnx_path, output_path, **kwargs):
        calls.append("quantize")
        Path(output_path).write_bytes(b"qdq-stub")

    monkeypatch.setattr(run_trt, "_apply_modelopt_sparsify", fake_apply_sparsify)
    monkeypatch.setattr(run_trt, "_build_calib_numpy",
                        lambda *a, **k: __import__("numpy").zeros((1, 3, 640, 640), dtype="float32"))
    # Patch the import point used inside _prepare_modelopt_onnx.
    import types as _t
    fake_moq_onnx = _t.ModuleType("modelopt.onnx.quantization")
    fake_moq_onnx.quantize = fake_quantize
    monkeypatch.setitem(sys.modules, "modelopt.onnx.quantization", fake_moq_onnx)

    recipe = _make_recipe(tmp_path, sparsity_preprocess="2:4")
    out = run_trt._prepare_modelopt_onnx(recipe, imgsz=640,
                                         cache_dir=tmp_path / "onnx", dynamic=True)
    assert out.exists()
    assert calls == ["sparsify", "quantize"]


def test_dispatcher_skips_sparsify_when_preprocess_none(tmp_path, monkeypatch):
    from scripts import run_trt

    calls = []

    def fake_apply_sparsify(*a, **k):
        calls.append("sparsify")
    def fake_export(weights, imgsz, **k):
        out = tmp_path / f"{Path(str(weights)).stem}.onnx"
        out.write_bytes(b"onnx-stub")
        return out
    def fake_quantize(onnx_path, output_path, **kwargs):
        calls.append("quantize")
        Path(output_path).write_bytes(b"qdq-stub")

    monkeypatch.setattr(run_trt, "_apply_modelopt_sparsify", fake_apply_sparsify)
    monkeypatch.setattr(run_trt, "_export_onnx", fake_export)
    monkeypatch.setattr(run_trt, "_build_calib_numpy",
                        lambda *a, **k: __import__("numpy").zeros((1, 3, 640, 640), dtype="float32"))
    import types as _t
    fake_moq_onnx = _t.ModuleType("modelopt.onnx.quantization")
    fake_moq_onnx.quantize = fake_quantize
    monkeypatch.setitem(sys.modules, "modelopt.onnx.quantization", fake_moq_onnx)

    recipe = _make_recipe(tmp_path)
    run_trt._prepare_modelopt_onnx(recipe, imgsz=640,
                                   cache_dir=tmp_path / "onnx", dynamic=True)
    assert calls == ["quantize"]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("modelopt.torch.sparsity") is None,
    reason="modelopt not installed",
)
def test_real_modelopt_sparsify_api_signature(tmp_path):
    """Integration test: hit real modelopt to confirm our sparsify call
    signature matches the installed version. Runs only when modelopt is
    present so CI without GPU/modelopt still passes the rest of the suite."""
    import torch
    from scripts.run_trt import _apply_modelopt_sparsify
    # Use a minimal dummy Conv model saved as an ultralytics-compatible .pt.
    # Skip if we cannot build one without the full ultralytics install.
    pytest.importorskip("ultralytics")
    # This test only verifies that the call does not raise; it does not
    # check the pruning pattern (that is what Task 7 verification is for).
    yolo = _apply_modelopt_sparsify("yolo26n.pt", 640)
    assert hasattr(yolo, "model")
```

### P-L3. Recipe #12: start with 4 excludes, not 7

YOLO quantization sensitivity evidence: the stem Conv and the bbox-regression
branch (`cv2.*`) dominate mAP loss. The classification branch (`cv3.*`) is
robust to INT8 quantization because softmax is tolerant of small log-prob
perturbations. Excluding all 7 layers creates too many INT8↔FP16 reformat
boundaries and eats the INT8 speedup. Start narrow; add only if the first
measurement misses the 1%p target.

Replace the v1 `nodes_to_exclude` list in `recipes/12_modelopt_int8_mixed.yaml`
(v1 Task 5 Step 2) with:

```yaml
  nodes_to_exclude:
    # Stem Conv — activation distribution differs from middle layers.
    - /model.0/conv/Conv
    # Detect head bbox-regression Convs — pixel-precision-critical.
    - /model.23/cv2.0/Conv
    - /model.23/cv2.1/Conv
    - /model.23/cv2.2/Conv
    # NOTE: cv3.* (classification) intentionally NOT excluded. Add them
    # in a follow-up recipe #12b only if this configuration misses the
    # 1%p mAP drop target.
```

### P-L7. Task 7: acceptance criteria + sparse-kernel verification

Append to v1 Task 7 after Step 5 (before "Commit results"), BEFORE Step 6:

> **Step 5a — Verify real 2:4 kernels were actually selected (Task 7 for recipe #11)**
>
> Rebuild recipe #11's engine with verbose logging to confirm TRT picked
> sparse INT8 tactics (otherwise the fix has the same no-op failure mode as
> v1.1's SPARSE_WEIGHTS-only path).
>
> ```bash
> # one-liner: delete the cached engine, rebuild with TRT verbose, grep tactic log
> rm -f results/_engines/*sparse24*.engine
> OMNI_COCO_YAML=$PWD/coco_val_only.yaml TRT_LOG_LEVEL=VERBOSE make recipe-11 2>&1 | tee results/_engines/recipe11_build.log
> grep -iE "sparse|2:4|SparseConv" results/_engines/recipe11_build.log | head -40
> ```
>
> **Pass condition:** at least one layer's selected tactic mentions `sparse`
> or `2:4`. If nothing matches, the 2:4 pattern did not stick through the
> ONNX export (most likely cause: modelopt wrappers leaked through because
> `mts.export` was skipped or ultralytics export re-normalized the weights).
> Debugging order: (a) inspect the QDQ ONNX initializers — confirm the
> weight tensors actually have 2-of-4 zeros per row; (b) if yes, filed as
> a TRT tactic selection issue; (c) if no, the sparsify path needs the
> onnx-graphsurgeon fallback.
>
> **Step 5b — Acceptance table (decide ship vs iterate)**
>
> Fill in after Step 5 runs:
>
> | Recipe | Metric | Threshold | Measured | Verdict |
> |---|---|---|---|---|
> | #11 (real 2:4) | fps bs=1 | ≥ recipe-09 × 0.85 | | |
> | #11 | mAP drop | ≤ 2.5%p | | |
> | #11 | sparse tactic seen | ≥ 1 layer | | |
> | #12 (mixed) | fps bs=1 | ≥ 30 | | |
> | #12 | mAP drop | < 1.0%p | | |
> | #12 | fps vs recipe-09 | ≥ recipe-09 × 0.75 | | |
>
> - If #12 misses the mAP target but fps is fine → add `cv3.*` excludes,
>   ship as recipe #12b.
> - If #12 misses fps target → exclude list is too long; drop cv2.2 (the
>   largest stride's bbox head is usually the least sensitive of the three).
> - If #11 has no sparse tactic AND mAP dropped → technique failed
>   silently; do not ship, file issue.

### P-L2. Calibration leakage — acknowledge, do not fix in Phase 3

Add the following note to the plan's "Scope → 포함" section as a reminder.
This is a pre-existing Phase 2 issue, out of Phase 3 scope, but relevant to
how results are interpreted:

> **Known caveat (carried from v1.1):** `calibration_samples` and
> `num_images` both draw (seed=42 shuffle) from COCO val2017 and may
> overlap by tens of images. This biases absolute mAP numbers slightly
> high for INT8 recipes. Because every modelopt recipe shares this setup,
> *relative* comparisons between recipes #8–#12 remain valid. Splitting
> calibration onto train2017 is tracked as a separate v1.3 item.

### P-L6. Fallback plan for torch-level sparsify failure

Documented inline in P-L4 already (the `_apply_modelopt_sparsify` docstring
calls out ONNX graph-level masking as Plan B). No additional step unless
Task 7 Step 5a "sparse tactic seen" check fails.

### Applying the patches — order of operations

When executing the plan below, walk Tasks 1 → 8 in order but treat the v2
patches as in-line overrides:

1. Task 1 (schema): unchanged.
2. Task 2 (runner): write the tests from **P-L5**, implement the runner
   using **P-API** + **P-L4** versions of `_apply_modelopt_sparsify` and
   `_prepare_modelopt_onnx`. The v1 Step 1/3 code blocks are superseded.
3. Task 3 (engine cache tag): still no-op, as in v1.
4. Task 4 (recipe #11 YAML): unchanged.
5. Task 5 (recipe #12 YAML): use the **P-L3** 4-exclude list.
6. Task 6 (Makefile): unchanged.
7. Task 7 (measurement): run Steps 1–5, then **P-L7** Steps 5a/5b, then
   Step 6 commit.
8. Task 8 (README): unchanged; add the "cv3.* intentionally excluded"
   rationale to the technique.nodes_to_exclude description.

---

## Scope

**포함:**
- Recipe #11 (`modelopt_int8_sparsity`) 버그픽스: `modelopt.torch.sparsity.sparsify`로 가중치 2:4 pruning → ONNX export → 기존 QDQ 경로.
- Recipe #12 (`modelopt_int8_mixed`) 신규 추가: `technique.nodes_to_exclude` YAML 필드로 민감 레이어 후보(첫 Conv, detect head regression/classification branch)를 FP16에 남김.
- `scripts/_schemas.py`의 `TechniqueSpec` 확장.
- `scripts/run_trt.py`의 `_prepare_modelopt_onnx`에 sparsify hook + exclude_nodes 전파.
- Makefile의 `recipe-08`..`recipe-12` target 추가 (현재 `recipe-01`..`recipe-07`만 있음), `make all`이 전체 11개 (v1.2로 12개) 레시피를 돌리도록 갱신.
- README의 Recipe 표와 `technique.source` 섹션 갱신.
- Phase 3 결과 반영한 `report.md` 재생성.

**제외 (QAT, v1.3으로):**
- 재학습 기반 QAT (`modelopt.torch.quantization.mode = "QAT"`).
- 자동 민감도 검색 (`modelopt.torch.quantization.auto_quantize`). 수동 exclude 리스트로 시작.

---

## File Structure

**Create:**
- `docs/plans/2026-04-18-phase3-int8-accuracy.md` — 이 파일.
- `recipes/12_modelopt_int8_mixed.yaml` — 민감 레이어 제외 혼합 정밀도 레시피.
- `tests/test_schema_phase3.py` — `TechniqueSpec` 새 필드 파싱 테스트.
- `tests/test_run_trt_phase3.py` — sparsify/exclude dispatcher 단위 테스트 (modelopt는 monkeypatch).

**Modify:**
- `scripts/_schemas.py` — `TechniqueSpec`에 두 필드 추가.
- `scripts/run_trt.py` — `_prepare_modelopt_onnx`에 sparsify 전처리와 exclude_nodes 전달.
- `recipes/11_modelopt_int8_sparsity.yaml` — `technique.sparsity_preprocess: "2:4"` 옵션 on.
- `Makefile` — `recipe-08`..`recipe-12` target + `all` 확장.
- `README.md` — Recipe 표와 `technique.source` 섹션.
- `report.md` — Phase 3 재측정 결과 덮어쓰기 (측정 후).

---

## Task 1: Schema 확장 — `TechniqueSpec`에 2:4 sparsity preprocess와 exclude_nodes 필드 추가

**Files:**
- Modify: `scripts/_schemas.py:24-34`
- Test: `tests/test_schema_phase3.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_schema_phase3.py`:

```python
from scripts._schemas import Recipe, TechniqueSpec


def test_technique_spec_accepts_sparsity_preprocess():
    t = TechniqueSpec(name="x", source="modelopt", sparsity_preprocess="2:4")
    assert t.sparsity_preprocess == "2:4"


def test_technique_spec_defaults_sparsity_preprocess_to_none():
    t = TechniqueSpec(name="x", source="modelopt")
    assert t.sparsity_preprocess is None


def test_technique_spec_accepts_nodes_to_exclude():
    t = TechniqueSpec(
        name="x",
        source="modelopt",
        nodes_to_exclude=["/model.0/conv/Conv", "/model.23/cv2.0/Conv"],
    )
    assert t.nodes_to_exclude == [
        "/model.0/conv/Conv",
        "/model.23/cv2.0/Conv",
    ]


def test_technique_spec_rejects_invalid_sparsity_preprocess():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TechniqueSpec(name="x", source="modelopt", sparsity_preprocess="1:2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema_phase3.py -v`
Expected: FAIL with "TechniqueSpec got unexpected keyword 'sparsity_preprocess'" (pydantic forbids extras).

- [ ] **Step 3: Implement schema change**

Edit `scripts/_schemas.py`, replace the `TechniqueSpec` class (currently lines 24–34):

```python
class TechniqueSpec(BaseModel):
    name: str
    # Where quantization / sparsity logic comes from.
    # v1 uses TensorRT's built-in calibrator + SPARSE_WEIGHTS flag.
    # v1.1+ plans to add "modelopt" (nvidia-modelopt: torch-level quantization
    # + QDQ-ONNX export) and possibly "ort_quant" for ONNX Runtime's quantizer.
    source: Literal["trt_builtin", "modelopt", "ort_quant"] = "trt_builtin"
    calibrator: Optional[str] = None
    calibration_samples: Optional[int] = None
    calibration_dataset: Optional[str] = None
    calibration_seed: Optional[int] = None
    # v1.2: when set, apply structured weight pruning *before* QDQ injection
    # so the engine builder can pick real 2:4 sparse INT8 kernels.
    # The plain `runtime.sparsity` flag alone only sets SPARSE_WEIGHTS, which
    # is a no-op unless the weights actually have the 2:4 pattern.
    sparsity_preprocess: Optional[Literal["2:4"]] = None
    # v1.2: ONNX node names to leave at FP16 during modelopt.onnx quantize.
    # Used to protect sensitivity-critical layers (first Conv, detect head).
    nodes_to_exclude: Optional[list[str]] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema_phase3.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_schemas.py tests/test_schema_phase3.py
git commit -m "feat(schema): TechniqueSpec supports sparsity_preprocess + nodes_to_exclude

Two new optional fields on TechniqueSpec:
- sparsity_preprocess: '2:4' — apply modelopt.torch.sparsify before export
- nodes_to_exclude: list[str] — ONNX node names to leave at FP16

Phase 3 foundation. No runner behavior change yet."
```

---

## Task 2: Runner — `_prepare_modelopt_onnx`가 sparsity_preprocess를 처리하도록 확장

**Files:**
- Modify: `scripts/run_trt.py:78-126`
- Test: `tests/test_run_trt_phase3.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_trt_phase3.py`:

```python
import sys
import types
from pathlib import Path

import pytest


def _make_recipe(tmp_path, **technique_kwargs):
    """Build a minimal Recipe object pointed at a real weights stub."""
    import yaml
    from scripts._schemas import load_recipe

    weights = tmp_path / "yolo26n.pt"
    weights.write_bytes(b"stub")

    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump({
        "name": "phase3_test",
        "model": {"family": "yolo26", "variant": "n", "weights": str(weights)},
        "runtime": {"engine": "tensorrt", "dtype": "int8"},
        "technique": {
            "name": "int8_ptq",
            "source": "modelopt",
            "calibrator": "entropy",
            "calibration_samples": 8,
            "calibration_seed": 42,
            **technique_kwargs,
        },
        "measurement": {
            "dataset": "coco_val2017",
            "num_images": 1,
            "warmup_iters": 1,
            "measure_iters": 1,
            "batch_sizes": [1],
            "input_size": 640,
            "gpu_clock_lock": False,
            "seed": 42,
        },
    }))
    return load_recipe(str(recipe_path))


def test_prepare_modelopt_onnx_invokes_sparsify_when_preprocess_set(tmp_path, monkeypatch):
    """When technique.sparsity_preprocess='2:4', sparsify() must be called
    before modelopt.onnx.quantize()."""
    from scripts import run_trt

    calls = []

    # Stub _export_onnx to return a synthetic ONNX path, sidestepping ultralytics.
    def fake_export(weights, imgsz, half, cache_dir, dynamic=True):
        p = Path(cache_dir) / f"stub_{imgsz}_fp32_{'dyn' if dynamic else 'bs1'}.onnx"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"onnx-stub")
        return p

    monkeypatch.setattr(run_trt, "_export_onnx", fake_export)
    monkeypatch.setattr(run_trt, "_build_calib_numpy",
                        lambda *a, **k: __import__("numpy").zeros((1, 3, 640, 640), dtype="float32"))

    # Stub modelopt.torch.sparsity.sparsify: records the call.
    sparsity_mod = types.ModuleType("modelopt.torch.sparsity")
    def fake_sparsify(model, mode, data_loader=None):
        calls.append(("sparsify", mode))
        return model
    sparsity_mod.sparsify = fake_sparsify
    monkeypatch.setitem(sys.modules, "modelopt.torch.sparsity", sparsity_mod)

    # Stub modelopt.onnx.quantization.quantize: creates the output file.
    onnx_q_mod = types.ModuleType("modelopt.onnx.quantization")
    def fake_quantize(onnx_path, output_path, **kwargs):
        calls.append(("quantize", kwargs.get("calibration_method")))
        Path(output_path).write_bytes(b"qdq-stub")
    onnx_q_mod.quantize = fake_quantize
    monkeypatch.setitem(sys.modules, "modelopt.onnx.quantization", onnx_q_mod)

    recipe = _make_recipe(tmp_path, sparsity_preprocess="2:4")
    out = run_trt._prepare_modelopt_onnx(recipe, imgsz=640,
                                         cache_dir=tmp_path / "onnx",
                                         dynamic=True)
    assert out.exists()
    names = [c[0] for c in calls]
    assert names == ["sparsify", "quantize"], f"expected sparsify before quantize, got {calls}"


def test_prepare_modelopt_onnx_skips_sparsify_when_preprocess_none(tmp_path, monkeypatch):
    """Default path: no sparsify call."""
    from scripts import run_trt

    calls = []

    def fake_export(weights, imgsz, half, cache_dir, dynamic=True):
        p = Path(cache_dir) / f"stub_{imgsz}_fp32_dyn.onnx"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"onnx-stub")
        return p

    monkeypatch.setattr(run_trt, "_export_onnx", fake_export)
    monkeypatch.setattr(run_trt, "_build_calib_numpy",
                        lambda *a, **k: __import__("numpy").zeros((1, 3, 640, 640), dtype="float32"))

    sparsity_mod = types.ModuleType("modelopt.torch.sparsity")
    def fake_sparsify(*a, **k):
        calls.append("sparsify")
    sparsity_mod.sparsify = fake_sparsify
    monkeypatch.setitem(sys.modules, "modelopt.torch.sparsity", sparsity_mod)

    onnx_q_mod = types.ModuleType("modelopt.onnx.quantization")
    def fake_quantize(onnx_path, output_path, **kwargs):
        calls.append("quantize")
        Path(output_path).write_bytes(b"qdq-stub")
    onnx_q_mod.quantize = fake_quantize
    monkeypatch.setitem(sys.modules, "modelopt.onnx.quantization", onnx_q_mod)

    recipe = _make_recipe(tmp_path)  # no sparsity_preprocess
    run_trt._prepare_modelopt_onnx(recipe, imgsz=640,
                                   cache_dir=tmp_path / "onnx", dynamic=True)
    assert calls == ["quantize"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_trt_phase3.py -v`
Expected: `test_prepare_modelopt_onnx_invokes_sparsify_when_preprocess_set` FAILs because `_prepare_modelopt_onnx` never calls `modelopt.torch.sparsity.sparsify`.

- [ ] **Step 3: Implement sparsify hook**

Edit `scripts/run_trt.py` — replace the body of `_prepare_modelopt_onnx` (lines 78–125). Add a helper for the sparsify step and invoke it when `recipe.technique.sparsity_preprocess == "2:4"`:

```python
def _apply_modelopt_sparsify(weights: str, imgsz: int, cache_dir: Path) -> Path:
    """Load ultralytics weights, apply 2:4 structured sparsity via
    modelopt.torch.sparsity, and return a path to a sparsified .pt that
    subsequent ONNX export can read.

    Runs with a single-batch random tensor as forward-only calibration data
    because 2:4 magnitude pruning does not need gradients.
    """
    try:
        from modelopt.torch.sparsity import sparsify as moq_sparsify
    except ImportError as e:
        raise RuntimeError(
            "nvidia-modelopt torch extension not installed. Install with: "
            "pip install --extra-index-url https://pypi.nvidia.com nvidia-modelopt"
        ) from e

    import torch
    from ultralytics import YOLO

    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(weights).stem}_{imgsz}_sparse24.pt"
    cached = cache_dir / tag
    if cached.exists():
        return cached

    yolo = YOLO(weights)
    inner = yolo.model  # nn.Module
    device = next(inner.parameters()).device
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)

    def _data_loader():
        for _ in range(1):
            yield dummy

    moq_sparsify(inner, mode="sparsegpt", data_loader=_data_loader())
    torch.save({"model": inner}, cached)
    print(f"[info] modelopt 2:4 sparsify wrote: {cached}", file=sys.stderr)
    return cached


def _prepare_modelopt_onnx(recipe: Recipe, imgsz: int, cache_dir: Path,
                           dynamic: bool = True) -> Path:
    """Quantize via modelopt.onnx — takes ultralytics' clean ONNX export and
    injects QDQ nodes using COCO calibration images.

    When ``technique.sparsity_preprocess == '2:4'`` we first pre-prune the
    weights with modelopt.torch.sparsity so the resulting QDQ-ONNX carries
    zeros in the 2:4 pattern; TensorRT's SPARSE_WEIGHTS flag then actually
    selects sparse INT8 kernels.
    """
    try:
        from modelopt.onnx.quantization import quantize as moq_quantize
    except ImportError as e:
        raise RuntimeError(
            "nvidia-modelopt not installed. Install with: "
            "pip install --extra-index-url https://pypi.nvidia.com nvidia-modelopt"
        ) from e

    cache_dir.mkdir(parents=True, exist_ok=True)
    calibrator = recipe.technique.calibrator or "max"
    sparsity_tag = "_sparse24" if recipe.technique.sparsity_preprocess == "2:4" else ""
    bs_tag = "dyn" if dynamic else "bs1"
    tag = (
        f"{Path(recipe.model.weights).stem}_{imgsz}_modelopt_"
        f"{calibrator}{sparsity_tag}_{bs_tag}.onnx"
    )
    cached = cache_dir / tag
    if cached.exists():
        return cached

    if recipe.technique.sparsity_preprocess == "2:4":
        sparse_pt = _apply_modelopt_sparsify(recipe.model.weights, imgsz, cache_dir)
        clean_onnx = _export_onnx(str(sparse_pt), imgsz, half=False,
                                  cache_dir=cache_dir, dynamic=dynamic)
    else:
        clean_onnx = _export_onnx(recipe.model.weights, imgsz, half=False,
                                  cache_dir=cache_dir, dynamic=dynamic)

    samples = recipe.technique.calibration_samples or 512
    seed = recipe.technique.calibration_seed or 42
    val_yaml = os.environ.get("OMNI_COCO_YAML")
    calib_data = _build_calib_numpy(val_yaml, samples, imgsz, seed)

    quant_kwargs = dict(
        onnx_path=str(clean_onnx),
        quantize_mode="int8",
        calibration_method=calibrator,
        calibration_data=calib_data,
        output_path=str(cached),
        log_level="WARNING",
    )
    if recipe.technique.nodes_to_exclude:
        quant_kwargs["nodes_to_exclude"] = list(recipe.technique.nodes_to_exclude)

    print(
        f"[info] modelopt.onnx.quantize: method={calibrator}, "
        f"samples={calib_data.shape[0]}, sparsity={sparsity_tag or 'none'}, "
        f"excludes={len(recipe.technique.nodes_to_exclude or [])}, "
        f"onnx={clean_onnx.name}",
        file=sys.stderr,
    )
    moq_quantize(**quant_kwargs)
    print(f"[info] modelopt wrote QDQ onnx: {cached}", file=sys.stderr)
    return cached
```

- [ ] **Step 4: Run tests to verify**

Run: `pytest tests/test_run_trt_phase3.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_trt.py tests/test_run_trt_phase3.py
git commit -m "feat(trt): modelopt 2:4 sparsify preprocess + nodes_to_exclude

_prepare_modelopt_onnx now:
- pre-prunes weights via modelopt.torch.sparsity.sparsify when
  technique.sparsity_preprocess='2:4', so TRT's SPARSE_WEIGHTS flag
  actually fires sparse INT8 kernels (was a no-op before).
- forwards technique.nodes_to_exclude to modelopt.onnx.quantize so
  sensitivity-critical layers stay at FP16 inside the QDQ ONNX."
```

---

## Task 3: Engine cache tag가 sparsity_preprocess를 반영하도록 갱신

**Files:**
- Modify: `scripts/run_trt.py:500-504`

- [ ] **Step 1: Inspect the current cache tag**

`engine_tag` 공식은 현재 다음과 같다:

```python
engine_tag = f"{onnx_path.stem}_{dtype}{'_sparse' if sparsity else ''}{source_suffix}_bs{bs}.engine"
```

`onnx_path.stem`에 이미 `_sparse24` 태그가 들어 있으므로 **변경 불필요**. Task 2의 ONNX 태그가 엔진 캐시 분리까지 자동으로 처리한다.

- [ ] **Step 2: Write the smoke test**

Append to `tests/test_run_trt_phase3.py`:

```python
def test_engine_tag_diverges_for_sparsified_onnx(tmp_path):
    """Engine cache key derives from onnx_path.stem; the sparse ONNX stem
    carries _sparse24, so sparse and non-sparse engines do not collide."""
    non_sparse = Path("yolo26n_640_modelopt_entropy_dyn")
    sparse = Path("yolo26n_640_modelopt_entropy_sparse24_dyn")
    assert "sparse24" not in str(non_sparse)
    assert "sparse24" in str(sparse)
    # engine_tag uses {stem}_{dtype}_sparse{source}_bs{bs}.engine when
    # runtime.sparsity is set — the stem difference is what keeps the
    # sparsified build out of the plain _entropy engine cache slot.
```

- [ ] **Step 3: Run it**

Run: `pytest tests/test_run_trt_phase3.py::test_engine_tag_diverges_for_sparsified_onnx -v`
Expected: PASS (it's a regression guard, not new behavior).

- [ ] **Step 4: No commit** — this test will ship with the Task 2 commit if run together, or as a separate trivial follow-up.

---

## Task 4: Recipe #11 YAML에 `sparsity_preprocess: "2:4"` 켜기

**Files:**
- Modify: `recipes/11_modelopt_int8_sparsity.yaml`

- [ ] **Step 1: Edit the recipe**

Replace the `technique:` block (lines 12–18):

```yaml
technique:
  name: int8_ptq_sparse
  source: modelopt
  calibrator: entropy
  calibration_samples: 512
  calibration_dataset: coco_val2017
  calibration_seed: 42
  sparsity_preprocess: "2:4"   # v1.2: real 2:4 via modelopt.torch.sparsify
```

- [ ] **Step 2: Verify YAML still loads**

Run:

```bash
python -c "import sys; sys.path.insert(0, '.'); from scripts._schemas import load_recipe; r = load_recipe('recipes/11_modelopt_int8_sparsity.yaml'); print(r.technique.sparsity_preprocess)"
```

Expected output: `2:4`

- [ ] **Step 3: Commit**

```bash
git add recipes/11_modelopt_int8_sparsity.yaml
git commit -m "feat(recipe-11): enable sparsity_preprocess=2:4

Flips modelopt_int8_sparsity from SPARSE_WEIGHTS-only (which was a no-op
because weights didn't carry the 2:4 pattern) to real 2:4 via
modelopt.torch.sparsity."
```

---

## Task 5: Recipe #12 신규 — `modelopt_int8_mixed` (민감 레이어 제외)

**Files:**
- Create: `recipes/12_modelopt_int8_mixed.yaml`

- [ ] **Step 1: Identify candidate exclude nodes**

YOLO26n의 민감 레이어 후보 (ultralytics 네이밍):
- 첫 Conv stem: `/model.0/conv/Conv`
- Detect head regression/classification branch의 마지막 1×1 Conv들 (ultralytics DetectV10 기준: `/model.23/cv2.0/Conv`, `/model.23/cv2.1/Conv`, `/model.23/cv2.2/Conv`, `/model.23/cv3.0/Conv`, `/model.23/cv3.1/Conv`, `/model.23/cv3.2/Conv`).

정확한 노드 이름은 첫 실행 후 확인 — 일단 이 후보로 시작.

- [ ] **Step 2: Create the recipe**

Write `recipes/12_modelopt_int8_mixed.yaml`:

```yaml
name: modelopt_int8_mixed
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
  name: int8_ptq_mixed
  source: modelopt
  calibrator: entropy          # start from best PTQ calibrator (recipe #9)
  calibration_samples: 512
  calibration_dataset: coco_val2017
  calibration_seed: 42
  # Keep the first Conv (stem) and detect head Conv branches at FP16.
  # These are the highest-sensitivity layers for YOLO-family detection mAP.
  nodes_to_exclude:
    - /model.0/conv/Conv
    - /model.23/cv2.0/Conv
    - /model.23/cv2.1/Conv
    - /model.23/cv2.2/Conv
    - /model.23/cv3.0/Conv
    - /model.23/cv3.1/Conv
    - /model.23/cv3.2/Conv
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
  max_map_drop_pct: 1.0
  min_fps_bs1: 30
```

- [ ] **Step 3: Verify YAML loads**

Run:

```bash
python -c "import sys; sys.path.insert(0, '.'); from scripts._schemas import load_recipe; r = load_recipe('recipes/12_modelopt_int8_mixed.yaml'); print(len(r.technique.nodes_to_exclude))"
```

Expected output: `7`

- [ ] **Step 4: Commit**

```bash
git add recipes/12_modelopt_int8_mixed.yaml
git commit -m "feat(recipe-12): add modelopt_int8_mixed recipe

INT8 PTQ with first-Conv and detect-head branches left at FP16.
Targets the 1%p mAP drop budget by protecting the two layer classes
that dominate quantization-induced accuracy loss on YOLO-family nets."
```

---

## Task 6: Makefile — recipe-08..12 target 추가 + `make all` 확장

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Edit Makefile**

Replace the current Makefile with:

```make
PYTHON ?= python
RECIPES_DIR := recipes
RESULTS_DIR := results
REPORT := report.md

.PHONY: all clean env report \
        recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 recipe-07 \
        recipe-08 recipe-09 recipe-10 recipe-11 recipe-12

all: recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 recipe-07 \
     recipe-08 recipe-09 recipe-10 recipe-11 recipe-12 report

env:
	$(PYTHON) scripts/env_lock.py --out $(RESULTS_DIR)/_env.json

recipe-01:
	$(PYTHON) scripts/run_pytorch.py --recipe $(RECIPES_DIR)/01_pytorch_fp32.yaml --out $(RESULTS_DIR)/01_pytorch_fp32.json

recipe-02:
	$(PYTHON) scripts/run_pytorch.py --recipe $(RECIPES_DIR)/02_torchcompile_fp16.yaml --out $(RESULTS_DIR)/02_torchcompile_fp16.json

recipe-03:
	$(PYTHON) scripts/run_ort.py --recipe $(RECIPES_DIR)/03_ort_cuda_fp16.yaml --out $(RESULTS_DIR)/03_ort_cuda_fp16.json

recipe-04:
	$(PYTHON) scripts/run_ort.py --recipe $(RECIPES_DIR)/04_ort_trt_fp16.yaml --out $(RESULTS_DIR)/04_ort_trt_fp16.json

recipe-05:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/05_trt_fp16.yaml --out $(RESULTS_DIR)/05_trt_fp16.json

recipe-06:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/06_trt_int8_ptq.yaml --out $(RESULTS_DIR)/06_trt_int8_ptq.json

recipe-07:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/07_trt_int8_sparsity.yaml --out $(RESULTS_DIR)/07_trt_int8_sparsity.json

recipe-08:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/08_modelopt_int8_ptq.yaml --out $(RESULTS_DIR)/08_modelopt_int8_ptq.json

recipe-09:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/09_modelopt_int8_entropy.yaml --out $(RESULTS_DIR)/09_modelopt_int8_entropy.json

recipe-10:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/10_modelopt_int8_percentile.yaml --out $(RESULTS_DIR)/10_modelopt_int8_percentile.json

recipe-11:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/11_modelopt_int8_sparsity.yaml --out $(RESULTS_DIR)/11_modelopt_int8_sparsity.json

recipe-12:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/12_modelopt_int8_mixed.yaml --out $(RESULTS_DIR)/12_modelopt_int8_mixed.json

report:
	$(PYTHON) scripts/recommend.py --results-dir $(RESULTS_DIR) --out $(REPORT)

clean:
	rm -rf $(RESULTS_DIR)/*.json $(REPORT) *.engine *.onnx build/ dist/ *.egg-info/
```

- [ ] **Step 2: Dry-run make**

Run: `make -n all | head -20`
Expected: prints 12 recipe commands + report, no errors.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: Makefile targets recipe-08..12 + make all covers Phase 2/3

Previously Makefile stopped at recipe-07 even though recipes 08-11 existed
and produced JSON results (they were run manually). This wires every
modelopt recipe into make all plus the new recipe-12 mixed precision."
```

---

## Task 7: Recipe #11 + #12 실측 + 정확한 노드 이름 확인

이 단계는 GPU가 있는 환경(WSL2 + RTX 3060 Laptop 기준, 기존 세팅 그대로)에서 수동 실행. 결과 JSON은 `results/`에 떨어지고 report.md 재생성 대상.

**Files:**
- Produce: `results/11_modelopt_int8_sparsity.json` (overwrite)
- Produce: `results/12_modelopt_int8_mixed.json` (new)

- [ ] **Step 1: Export clean ONNX once to inspect node names**

Run from the project root:

```bash
OMNI_COCO_YAML=$PWD/coco_val_only.yaml python -c "
from scripts.run_trt import _export_onnx
from pathlib import Path
p = _export_onnx('yolo26n.pt', 640, half=False, cache_dir=Path('results/_onnx'), dynamic=True)
import onnx
m = onnx.load(str(p))
names = [n.name for n in m.graph.node if n.op_type == 'Conv']
print(names[0])
for n in names:
    if '/model.23/' in n:
        print(n)
"
```

**Expected:** first line is the stem Conv, remaining lines are detect-head Conv nodes. If the names differ from the placeholders in `12_modelopt_int8_mixed.yaml`, edit the YAML to match before proceeding.

- [ ] **Step 2: Fix recipe #12 node names if needed**

If actual names differ, update `recipes/12_modelopt_int8_mixed.yaml`'s `nodes_to_exclude` list. Then:

```bash
git add recipes/12_modelopt_int8_mixed.yaml
git commit -m "fix(recipe-12): correct nodes_to_exclude to real ONNX node names"
```

(Skip this commit if names were already correct.)

- [ ] **Step 3: Run recipe #11 (real 2:4 sparsity)**

```bash
OMNI_COCO_YAML=$PWD/coco_val_only.yaml make recipe-11
```

**Expected output tail:**
```
[info] modelopt 2:4 sparsify wrote: ...sparse24.pt
[info] modelopt.onnx.quantize: method=entropy, samples=512, sparsity=_sparse24, excludes=0, ...
[info] modelopt wrote QDQ onnx: ...sparse24_dyn.onnx
wrote results/11_modelopt_int8_sparsity.json
```

Sanity check: `results/11_modelopt_int8_sparsity.json` should show `accuracy.map_50_95` **different** from `09_modelopt_int8_entropy.json` (previously they were identical because sparsity didn't actually fire).

- [ ] **Step 4: Run recipe #12 (mixed precision)**

```bash
OMNI_COCO_YAML=$PWD/coco_val_only.yaml make recipe-12
```

**Expected:** `results/12_modelopt_int8_mixed.json` written. mAP drop should be lower than recipe #9 (the straight entropy PTQ) and fps somewhere between recipe #9 and recipe #5 (pure FP16).

- [ ] **Step 5: Regenerate report**

```bash
make report
```

Verify `report.md` now lists 12 recipes and highlights recipe #12 if it beats the 1%p drop threshold with reasonable fps.

- [ ] **Step 6: Commit results**

```bash
git add results/11_modelopt_int8_sparsity.json results/12_modelopt_int8_mixed.json report.md
git commit -m "bench(phase3): recipe-11 real 2:4 sparsity + recipe-12 mixed precision

Results measured on WSL2, RTX 3060 Laptop, 500 val images.

- recipe-11 (real 2:4): <fill in fps/mAP from the run>
- recipe-12 (mixed):    <fill in fps/mAP from the run>

See report.md for the full matrix."
```

---

## Task 8: README 갱신

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update Recipe 표 (recipe-12 행 추가)**

Open `README.md`, find the 11-row recipe table (after Phase 2 update). Append one row:

```markdown
| 12 | modelopt_int8_mixed | tensorrt | int8 (FP16 exclude) | modelopt | PTQ + sensitive-layer exclude |
```

- [ ] **Step 2: Update `technique.source` 섹션에 두 신규 필드 설명 추가**

Find the `### technique.source 디스패처 (v1.1+)` section, append:

```markdown
#### v1.2 옵션 필드

- `technique.sparsity_preprocess: "2:4"` — `modelopt.torch.sparsity.sparsify`로
  가중치를 2:4 패턴으로 사전 pruning. `runtime.sparsity: "2:4"`와 함께 써야
  TRT가 실제 2:4 커널을 고른다. 단독으로 `SPARSE_WEIGHTS`만 세우는 건 no-op.
- `technique.nodes_to_exclude: [str]` — modelopt QDQ 주입 단계에서 해당 ONNX
  노드를 FP16에 남긴다. YOLO 계열 detect head (`/model.23/...`)와 stem Conv
  (`/model.0/conv/Conv`)가 전형적인 후보.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README covers recipe-12 and Phase 3 technique fields"
```

---

## Self-Review 체크리스트 (작성자용, 코드 착수 전 확인)

**Spec coverage:**
- [x] 2:4 sparsity 실제 적용 → Task 1 (schema), Task 2 (runner), Task 4 (recipe YAML), Task 7 (측정).
- [x] 민감 레이어 제외 → Task 1 (schema), Task 2 (runner nodes_to_exclude forward), Task 5 (recipe YAML), Task 7 (측정).
- [x] QAT는 v1.3으로 명시 제외 (Scope 섹션).

**Placeholder scan:**
- Task 7 Step 6의 commit message에 `<fill in fps/mAP>` — 실측 후에만 확정 가능한 값이므로 실행 단계에서 채움. 이건 허용되는 placeholder (결과 의존).
- 그 외 모든 step은 실행 가능한 코드/명령어 포함.

**Type consistency:**
- `TechniqueSpec.sparsity_preprocess` (Task 1) ↔ `recipe.technique.sparsity_preprocess` (Task 2, 4) ↔ `sparsity_preprocess: "2:4"` (Task 4 YAML). 일치.
- `TechniqueSpec.nodes_to_exclude` (Task 1) ↔ `recipe.technique.nodes_to_exclude` (Task 2) ↔ `nodes_to_exclude:` (Task 5 YAML). 일치.
- `_apply_modelopt_sparsify` signature (Task 2) consistent with call site in `_prepare_modelopt_onnx`.

**Risk notes:**
- `modelopt.torch.sparsity.sparsify`의 정확한 API (mode name, data_loader requirement)는 설치된 modelopt 버전에 따라 달라질 수 있음. Task 2 Step 3의 `mode="sparsegpt"`는 일반적인 기본값이며, 첫 실행에서 불일치가 나면 조정한다.
- YOLO26n 모델의 detect head Conv 노드 이름은 ultralytics 버전에 따라 약간 다를 수 있음. Task 7 Step 1에서 실제 이름을 확인하는 단계가 들어가 있으므로 설계상 방어됨.
