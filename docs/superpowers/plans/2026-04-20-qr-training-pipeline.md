# QR Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `best_qr.pt` 기반 QAT/Sparsity 학습 파이프라인과 3개 레시피 (#07 prune_24, #11 modelopt_sparsify, #17 modelopt_qat) 활성화.

**Architecture:** `scripts/train.py` 단독 entry로 YAML recipe의 `technique.training`을 dispatch. 3개 modifier 각자 `_modifiers/*.py`가 ultralytics YOLO wrapper를 in-place 수정 (prune/sparsify/quantize fake) → `model.train()` 래핑 → in-memory yolo.model 직렬화. modelopt는 `mto.save/restore`, prune_24는 plain state_dict. `run_trt.py`는 `trained_weights/{recipe}.pt` 자동 조회.

**Tech Stack:** Python 3.13, ultralytics 8.4.27, nvidia-modelopt 0.43.0, torch 2.x, TensorRT 10.16, pydantic 2, pytest

**Spec reference:** `docs/superpowers/specs/2026-04-20-qr-training-pipeline-design.md`

---

## Task Dependency Graph

```
Task 1 (TrainingSpec)
  └── Task 2 (_resolve_weights 준비) ─── Task 8 (run_trt 통합)
  └── Task 3 (_modifiers base)
        ├── Task 4 (prune_24) ──┐
        ├── Task 5 (modelopt_sparsify) ─┤
        └── Task 6 (modelopt_qat) ─────┤
                                       └── Task 7 (_train_core + train.py)
                                             └── Task 9 (recipes)
                                                   └── Task 10 (batch/configs)
                                                         └── Task 11 (smoke)
                                                               └── Task 12-15 (실제 학습/평가/문서)
```

---

## Task 1: TrainingSpec schema 추가

**Files:**
- Modify: `scripts/_schemas.py`
- Create: `tests/test_training_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_schema.py`:
```python
"""Schema tests for TrainingSpec (spec §4)."""
import pytest
from pydantic import ValidationError
from scripts._schemas import TrainingSpec, TechniqueSpec


def test_training_spec_minimal_fields():
    spec = TrainingSpec(
        base_checkpoint="best_qr.pt",
        epochs=30,
        modifier="modelopt_qat",
    )
    assert spec.base_checkpoint == "best_qr.pt"
    assert spec.epochs == 30
    assert spec.modifier == "modelopt_qat"
    assert spec.lr0 == 0.001  # default
    assert spec.batch == 8


def test_training_spec_requires_modifier():
    with pytest.raises(ValidationError):
        TrainingSpec(base_checkpoint="best_qr.pt", epochs=30)  # missing modifier


def test_training_spec_rejects_unknown_modifier():
    with pytest.raises(ValidationError):
        TrainingSpec(
            base_checkpoint="best_qr.pt",
            epochs=30,
            modifier="unknown_algo",
        )


def test_training_spec_accepts_all_three_modifiers():
    for m in ("prune_24", "modelopt_sparsify", "modelopt_qat"):
        spec = TrainingSpec(
            base_checkpoint="best_qr.pt",
            epochs=10,
            modifier=m,
        )
        assert spec.modifier == m


def test_technique_spec_training_is_optional():
    t = TechniqueSpec(name="int8_ptq", source="trt_builtin")
    assert t.training is None


def test_technique_spec_with_training():
    t = TechniqueSpec(
        name="int8_qat",
        source="modelopt",
        training=TrainingSpec(
            base_checkpoint="best_qr.pt",
            epochs=30,
            modifier="modelopt_qat",
        ),
    )
    assert t.training.modifier == "modelopt_qat"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_training_schema.py -v
```
Expected: FAIL — `ImportError: cannot import name 'TrainingSpec' from 'scripts._schemas'`

- [ ] **Step 3: Add `TrainingSpec` class + `TechniqueSpec.training` field**

Edit `scripts/_schemas.py`. Add after `class TechniqueSpec` definition (around line 45, before `class HardwareSpec`):

```python
class TrainingSpec(BaseModel):
    """Fine-tuning recipe for QAT / sparsity modifiers.

    Appears under ``TechniqueSpec.training`` and activates the
    ``scripts/train.py`` entry point. Absent for non-training recipes.
    """
    base_checkpoint: str
    epochs: int
    batch: int = 8
    workers: int = 4
    imgsz: int = 640
    lr0: float = 0.001
    optimizer: str = "AdamW"
    seed: int = 42
    data_yaml: Optional[str] = None
    modifier: Literal["prune_24", "modelopt_sparsify", "modelopt_qat"]
    prune_amount: Optional[float] = None
    quant_config: Optional[str] = "int8_default"
```

Then modify `TechniqueSpec` to add the new field (insert at end of its fields):
```python
class TechniqueSpec(BaseModel):
    # ... existing fields ...
    nodes_to_exclude: Optional[list[str]] = None
    # v1.3: fine-tune before quantize (QAT / sparsity recovery). None for
    # PTQ-only recipes. Drives scripts/train.py; see TrainingSpec.
    training: Optional["TrainingSpec"] = None
```

Also update the `__all__`-style imports if any. At bottom of file, ensure forward reference resolves:
```python
TechniqueSpec.model_rebuild()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_training_schema.py -v
```
Expected: PASS, 6 tests

- [ ] **Step 5: Run full test suite to catch regressions**

```bash
pytest tests/ -q
```
Expected: all previous tests still pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/_schemas.py tests/test_training_schema.py
git commit -m "feat(schemas): TrainingSpec + TechniqueSpec.training field

spec §4. 3개 modifier literal 제약 (prune_24 / modelopt_sparsify /
modelopt_qat). epochs/batch/lr0 기본값 포함. 기존 PTQ 레시피는 training=None
유지.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `_resolve_weights(recipe)` 헬퍼 (prune_24 케이스만)

**Files:**
- Modify: `scripts/run_trt.py` (상단 유틸 섹션)
- Create: `tests/test_run_trt_trained_weights.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_trt_trained_weights.py`:
```python
"""Tests for _resolve_weights() in run_trt.py (spec §6)."""
import pytest
from pathlib import Path
from unittest.mock import patch
from scripts._schemas import (
    Recipe, ModelSpec, RuntimeSpec, TechniqueSpec, MeasurementSpec,
    TrainingSpec,
)


def _make_recipe(training: TrainingSpec | None = None, weights: str = "yolo26n.pt") -> Recipe:
    return Recipe(
        name="test_recipe",
        model=ModelSpec(family="yolo26", variant="n", weights=weights),
        runtime=RuntimeSpec(engine="tensorrt", dtype="int8"),
        technique=TechniqueSpec(name="int8_test", training=training),
        measurement=MeasurementSpec(
            dataset="coco_val2017", num_images=10,
            warmup_iters=1, measure_iters=1, batch_sizes=[1],
        ),
    )


def test_resolve_weights_no_training_returns_original():
    from scripts.run_trt import _resolve_weights
    recipe = _make_recipe(training=None, weights="yolo26n.pt")
    assert _resolve_weights(recipe) == "yolo26n.pt"


def test_resolve_weights_missing_trained_raises(tmp_path, monkeypatch):
    from scripts import run_trt
    monkeypatch.setattr(run_trt, "ROOT", tmp_path)
    recipe = _make_recipe(training=TrainingSpec(
        base_checkpoint="best_qr.pt", epochs=5, modifier="prune_24",
    ))
    with pytest.raises(RuntimeError, match="requires training"):
        run_trt._resolve_weights(recipe)


def test_resolve_weights_prune_24_returns_trained_path(tmp_path, monkeypatch):
    from scripts import run_trt
    monkeypatch.setattr(run_trt, "ROOT", tmp_path)
    tw = tmp_path / "trained_weights"
    tw.mkdir()
    trained = tw / "test_recipe.pt"
    trained.write_bytes(b"fake")
    recipe = _make_recipe(training=TrainingSpec(
        base_checkpoint="best_qr.pt", epochs=5, modifier="prune_24",
    ))
    assert _resolve_weights_result := run_trt._resolve_weights(recipe)
    assert Path(_resolve_weights_result) == trained
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_run_trt_trained_weights.py -v
```
Expected: FAIL — `AttributeError: module 'scripts.run_trt' has no attribute '_resolve_weights'`

- [ ] **Step 3: Add `_resolve_weights()` helper in `scripts/run_trt.py`**

Insert after the imports block (around line 44, after `from scripts import _split`):

```python
def _resolve_weights(recipe: Recipe) -> str:
    """Return the weights path for runner consumption.

    If the recipe has ``technique.training``, prefer ``trained_weights/{name}.pt``
    produced by ``scripts/train.py``. Currently returns a plain path for
    ``prune_24`` modifier (state_dict is ultralytics-compatible). Modelopt
    modifiers (sparsify/qat) need additional restore — see Task 8.
    """
    if recipe.technique.training is None:
        return recipe.model.weights
    trained = ROOT / "trained_weights" / f"{recipe.name}.pt"
    if not trained.exists():
        raise RuntimeError(
            f"Recipe {recipe.name!r} requires training but {trained} is "
            f"missing. Run: python scripts/train.py --recipe "
            f"recipes/{recipe.name}.yaml"
        )
    modifier = recipe.technique.training.modifier
    if modifier == "prune_24":
        return str(trained)
    # modelopt_sparsify / modelopt_qat handled in Task 8
    raise NotImplementedError(
        f"_resolve_weights for modifier={modifier!r} not wired yet"
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_run_trt_trained_weights.py -v
```
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/run_trt.py tests/test_run_trt_trained_weights.py
git commit -m "feat(run_trt): _resolve_weights helper for trained pt (prune_24)

spec §6. training 있는 recipe에서 trained_weights/{name}.pt 자동 조회.
없으면 명확한 에러. modelopt modifier는 NotImplementedError (Task 8에서 구현).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `_modifiers` 패키지 스캐폴드

**Files:**
- Create: `scripts/_modifiers/__init__.py`
- Create: `scripts/_modifiers/_base.py` (공통 인터페이스 docstring)

- [ ] **Step 1: Create package marker**

Create `scripts/_modifiers/__init__.py`:
```python
"""Training modifier plugins for scripts/train.py.

Each module exposes two functions:

    def apply(yolo: "YOLO", spec: "TrainingSpec") -> None:
        '''Mutate the YOLO wrapper (pruning masks / fake-quant modules /
        sparsity state). Called before ultralytics model.train().'''

    def finalize(yolo: "YOLO", spec: "TrainingSpec", out_pt: "Path") -> None:
        '''Serialize the trained model to out_pt. Called after training.
        prune_24 writes a plain state_dict; modelopt_* use mto.save.'''

Spec reference: docs/superpowers/specs/2026-04-20-qr-training-pipeline-design.md
"""
```

- [ ] **Step 2: Verify import**

```bash
python -c "from scripts._modifiers import *; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/_modifiers/__init__.py
git commit -m "feat(_modifiers): package scaffold for training modifier plugins

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `prune_24` modifier 구현

**Files:**
- Create: `scripts/_modifiers/prune_24.py`
- Create: `tests/test_modifiers_prune_24.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_modifiers_prune_24.py`:
```python
"""prune_24 modifier tests (spec §5.4-5.5)."""
import pytest
import torch
import torch.nn as nn


def _has_2_4_pattern(w: torch.Tensor) -> bool:
    """Every group of 4 consecutive elements has at most 2 non-zeros."""
    flat = w.detach().reshape(-1)
    # Pad to multiple of 4 if needed
    pad = (4 - flat.numel() % 4) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
    groups = flat.view(-1, 4)
    return bool(((groups != 0).sum(dim=-1) <= 2).all().item())


def test_verify_2_4_helper_positive():
    from scripts._modifiers.prune_24 import _verify_2_4_pattern
    w = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0])
    assert _verify_2_4_pattern(w)


def test_verify_2_4_helper_negative():
    from scripts._modifiers.prune_24 import _verify_2_4_pattern
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])  # all nonzero → violates
    assert not _verify_2_4_pattern(w)


def test_apply_prunes_conv_weight():
    from scripts._modifiers.prune_24 import _apply_2_4_mask_to_module
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 16, 3)
    _apply_2_4_mask_to_module(conv)
    # After apply, forward gives masked weight; check weight_orig has weight_mask
    assert hasattr(conv, "weight_orig")
    assert hasattr(conv, "weight_mask")
    # The effective weight (conv.weight) should satisfy 2:4
    assert _has_2_4_pattern(conv.weight)


def test_finalize_removes_pruning_parametrization():
    from scripts._modifiers.prune_24 import _apply_2_4_mask_to_module, _finalize_module
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 16, 3)
    _apply_2_4_mask_to_module(conv)
    _finalize_module(conv)
    # After finalize: weight is a plain Parameter (no weight_orig), still 2:4
    assert not hasattr(conv, "weight_orig")
    assert _has_2_4_pattern(conv.weight)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_modifiers_prune_24.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts._modifiers.prune_24'`

- [ ] **Step 3: Implement `prune_24.py`**

Create `scripts/_modifiers/prune_24.py`:
```python
"""2:4 structured magnitude pruning + SAT (Sparsity-Aware Training).

Flow (spec §5.4 / §5.5):
1. ``apply(yolo, spec)``: for each eligible Conv/Linear in the model, compute
   a mask that keeps top-2 magnitude weights out of every consecutive group
   of 4, then register ``torch.nn.utils.prune.custom_from_mask``. The forward
   pre-hook re-applies the mask each call so ``optimizer.step`` updates
   ``weight_orig`` freely but the effective forward weight always satisfies
   the 2:4 pattern.
2. Training runs normally (60 epochs per recipe).
3. ``finalize(yolo, spec, out_pt)``: call ``prune.remove()`` to bake masks
   into the weight tensor permanently, then ``torch.save`` a plain state
   dict that ``ultralytics.YOLO`` can load via its normal path.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

if TYPE_CHECKING:
    from ultralytics import YOLO
    from scripts._schemas import TrainingSpec


def _compute_2_4_mask(weight: torch.Tensor) -> torch.Tensor:
    """For a weight tensor, return a mask with the 2:4 structured pattern.

    The mask keeps the two largest-magnitude elements in every block of 4
    along the flattened dim and zeros the other two. If the total element
    count is not a multiple of 4, the trailing remainder keeps its
    magnitude-top half (equivalent to 2:4 applied to the padded view).
    """
    flat = weight.detach().reshape(-1)
    n = flat.numel()
    pad = (4 - n % 4) % 4
    if pad:
        flat_padded = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
    else:
        flat_padded = flat
    groups = flat_padded.abs().view(-1, 4)
    # Top-2 magnitude per group
    _, topk_idx = groups.topk(2, dim=-1)
    mask_groups = torch.zeros_like(groups)
    mask_groups.scatter_(1, topk_idx, 1.0)
    mask_flat = mask_groups.view(-1)
    if pad:
        mask_flat = mask_flat[:n]
    return mask_flat.view_as(weight)


def _verify_2_4_pattern(weight: torch.Tensor) -> bool:
    """Check that every 4-element group has <= 2 non-zeros."""
    flat = weight.detach().reshape(-1)
    n = flat.numel()
    pad = (4 - n % 4) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
    groups = flat.view(-1, 4)
    return bool(((groups != 0).sum(dim=-1) <= 2).all().item())


def _is_eligible_module(m: nn.Module) -> bool:
    """Conv2d and Linear layers are eligible for 2:4 pruning.

    Excludes 1x1 pointwise convs with < 4 input channels (too small),
    depthwise convs (groups == in_channels), and the final detect head
    (heuristic: skip if ``_omni_skip_prune`` attribute is set).
    """
    if getattr(m, "_omni_skip_prune", False):
        return False
    if isinstance(m, nn.Conv2d):
        if m.groups == m.in_channels and m.groups > 1:  # depthwise
            return False
        if m.weight.numel() < 16:  # too small to benefit
            return False
        return True
    if isinstance(m, nn.Linear):
        return m.weight.numel() >= 16
    return False


def _apply_2_4_mask_to_module(module: nn.Module) -> None:
    """Attach a 2:4 mask via ``custom_from_mask``.

    Registers ``weight_orig`` (learnable) and ``weight_mask`` (buffer), plus
    a forward pre-hook that computes ``weight = weight_orig * weight_mask``.
    """
    mask = _compute_2_4_mask(module.weight)
    prune.custom_from_mask(module, name="weight", mask=mask)


def _finalize_module(module: nn.Module) -> None:
    """Bake the mask permanently into ``weight`` and remove hook/parametrization."""
    prune.remove(module, "weight")


def apply(yolo: "YOLO", spec: "TrainingSpec") -> None:
    """Mutate ``yolo.model`` in-place to add 2:4 mask parametrizations."""
    applied = 0
    for name, m in yolo.model.named_modules():
        if _is_eligible_module(m):
            _apply_2_4_mask_to_module(m)
            applied += 1
    print(f"[prune_24] applied 2:4 mask to {applied} modules")


def finalize(yolo: "YOLO", spec: "TrainingSpec", out_pt: Path) -> None:
    """Bake masks, verify, and save a plain ultralytics-compatible checkpoint."""
    removed = 0
    for name, m in yolo.model.named_modules():
        if prune.is_pruned(m):
            _finalize_module(m)
            removed += 1
            if not _verify_2_4_pattern(m.weight):
                raise RuntimeError(
                    f"prune_24.finalize: module {name!r} weight does not "
                    f"satisfy 2:4 pattern after prune.remove()"
                )
    print(f"[prune_24] finalized {removed} modules, 2:4 pattern verified")
    # Save in ultralytics checkpoint format (train.py layer handles this path)
    torch.save({"model": yolo.model}, str(out_pt))
    print(f"[prune_24] saved → {out_pt}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_modifiers_prune_24.py -v
```
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/_modifiers/prune_24.py tests/test_modifiers_prune_24.py
git commit -m "feat(_modifiers): prune_24 magnitude 2:4 + SAT

spec §5.4-5.5. custom_from_mask hook으로 optimizer.step()과 2:4 공존.
finalize에서 prune.remove() + _verify_2_4_pattern assert.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `modelopt_sparsify` modifier 구현

**Files:**
- Create: `scripts/_modifiers/modelopt_sparsify.py`
- Create: `tests/test_modifiers_modelopt_sparsify.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_modifiers_modelopt_sparsify.py`:
```python
"""modelopt_sparsify modifier tests (spec §5)."""
import pytest
import torch
import torch.nn as nn
from pathlib import Path

mto = pytest.importorskip("modelopt.torch.opt")
ms = pytest.importorskip("modelopt.torch.sparsity")


class _FakeYolo:
    """Minimal YOLO-like stub (has .model attribute)."""
    def __init__(self, model):
        self.model = model


def _tiny_net() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3),
    )


def test_apply_returns_none_and_wraps_model():
    from scripts._modifiers import modelopt_sparsify
    from scripts._schemas import TrainingSpec
    yolo = _FakeYolo(_tiny_net())
    spec = TrainingSpec(
        base_checkpoint="x.pt", epochs=1,
        modifier="modelopt_sparsify",
    )
    modelopt_sparsify.apply(yolo, spec)
    # modelopt.torch.sparsity.sparsify mutates model in-place.
    # Verify some modelopt metadata was added.
    state = mto.modelopt_state(yolo.model)
    assert state is not None


def test_finalize_saves_via_mto(tmp_path):
    from scripts._modifiers import modelopt_sparsify
    from scripts._schemas import TrainingSpec
    yolo = _FakeYolo(_tiny_net())
    spec = TrainingSpec(
        base_checkpoint="x.pt", epochs=1,
        modifier="modelopt_sparsify",
    )
    modelopt_sparsify.apply(yolo, spec)
    out = tmp_path / "test.pt"
    modelopt_sparsify.finalize(yolo, spec, out)
    assert out.exists()
    # Restoring should reconstruct modelopt state
    yolo2 = _FakeYolo(_tiny_net())
    mto.restore(yolo2.model, str(out))
    assert mto.modelopt_state(yolo2.model) is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_modifiers_modelopt_sparsify.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts._modifiers.modelopt_sparsify'`

- [ ] **Step 3: Implement `modelopt_sparsify.py`**

Create `scripts/_modifiers/modelopt_sparsify.py`:
```python
"""2:4 sparsity via nvidia-modelopt (torch-level).

Wraps ``modelopt.torch.sparsity.sparsify`` with mode ``sparse_magnitude``.
Default search config already uses ``pattern='2:4 sparsity'``, so no extra
config dict is needed.  Finalize uses ``modelopt.torch.opt.save`` to preserve
modelopt metadata alongside the state dict; runner restores via
``modelopt.torch.opt.restore``.
See spec §6.

NOTE: The ``config`` param of ``ms.sparsify`` is for *search* options
(verbose, forward_loop, etc.), not sparsity type.  The 2:4 pattern is the
default for ``sparse_magnitude`` mode (``pattern='2:4 sparsity'``).
Passing ``config={"sparsity": {"sparsity_type": "2:4"}}`` raises
``AssertionError: Unexpected config keys: {'sparsity'}`` on modelopt 0.43.0.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO
    from scripts._schemas import TrainingSpec


def apply(yolo: "YOLO", spec: "TrainingSpec") -> None:
    import modelopt.torch.sparsity as ms

    ms.sparsify(yolo.model, mode="sparse_magnitude")

    # Verify at least one layer actually got a non-trivial mask. modelopt
    # silently skips layers that don't meet shape constraints (e.g.,
    # in_channels % 16 != 0); if every layer is skipped, _weight_mask ends
    # up all-ones and TRT SPARSE_WEIGHTS becomes a no-op.
    sparsified = 0
    for _, m in yolo.model.named_modules():
        mask = getattr(m, "_weight_mask", None)
        if mask is not None and not bool(mask.all().item()):
            sparsified += 1
    if sparsified == 0:
        raise RuntimeError(
            "modelopt_sparsify.apply: no layers were actually sparsified. "
            "All eligible layers may have been skipped by modelopt due to "
            "shape constraints (e.g., in_channels % 16 != 0). Check "
            "model architecture."
        )
    print(f"[modelopt_sparsify] sparsify(sparse_magnitude, 2:4) applied — "
          f"{sparsified} layers sparsified")


def finalize(yolo: "YOLO", spec: "TrainingSpec", out_pt: Path) -> None:
    import modelopt.torch.opt as mto

    mto.save(yolo.model, str(out_pt))
    print(f"[modelopt_sparsify] saved → {out_pt}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_modifiers_modelopt_sparsify.py -v
```
Expected: PASS, 2 tests (or SKIP if modelopt not installed)

- [ ] **Step 5: Commit**

```bash
git add scripts/_modifiers/modelopt_sparsify.py tests/test_modifiers_modelopt_sparsify.py
git commit -m "feat(_modifiers): modelopt_sparsify (2:4 via nvidia-modelopt)

spec §6. ms.sparsify(sparse_magnitude, 2:4) + mto.save/restore.
pytest.importorskip로 modelopt 미설치 환경 skip.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `modelopt_qat` modifier 구현

**Files:**
- Create: `scripts/_modifiers/modelopt_qat.py`
- Create: `tests/test_modifiers_modelopt_qat.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_modifiers_modelopt_qat.py`:
```python
"""modelopt_qat modifier tests."""
import pytest
import torch
import torch.nn as nn
from pathlib import Path

mto = pytest.importorskip("modelopt.torch.opt")
mtq = pytest.importorskip("modelopt.torch.quantization")


class _FakeYolo:
    def __init__(self, model):
        self.model = model


def _tiny_net() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3),
    )


def test_apply_inserts_fake_quant():
    from scripts._modifiers import modelopt_qat
    from scripts._schemas import TrainingSpec
    yolo = _FakeYolo(_tiny_net())
    spec = TrainingSpec(
        base_checkpoint="x.pt", epochs=1,
        modifier="modelopt_qat",
    )
    modelopt_qat.apply(yolo, spec)
    # After quantize, modelopt metadata should be attached
    state = mto.modelopt_state(yolo.model)
    assert state is not None


def test_finalize_roundtrip(tmp_path):
    from scripts._modifiers import modelopt_qat
    from scripts._schemas import TrainingSpec
    yolo = _FakeYolo(_tiny_net())
    spec = TrainingSpec(
        base_checkpoint="x.pt", epochs=1,
        modifier="modelopt_qat",
    )
    modelopt_qat.apply(yolo, spec)
    out = tmp_path / "qat.pt"
    modelopt_qat.finalize(yolo, spec, out)
    assert out.exists()
    # Round-trip restore
    yolo2 = _FakeYolo(_tiny_net())
    mto.restore(yolo2.model, str(out))
    # Forward still works
    x = torch.randn(1, 3, 32, 32)
    y = yolo2.model(x)
    assert y.shape[1] == 16


def test_unknown_quant_config_raises():
    from scripts._modifiers import modelopt_qat
    from scripts._schemas import TrainingSpec
    yolo = _FakeYolo(_tiny_net())
    spec = TrainingSpec(
        base_checkpoint="x.pt", epochs=1,
        modifier="modelopt_qat",
        quant_config="nonexistent_cfg_name",
    )
    with pytest.raises(ValueError, match="Unknown quant_config"):
        modelopt_qat.apply(yolo, spec)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_modifiers_modelopt_qat.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Implement `modelopt_qat.py`**

Create `scripts/_modifiers/modelopt_qat.py`:
```python
"""INT8 QAT (Quantization-Aware Training) via nvidia-modelopt.

Inserts fake-quant modules before training; STE (straight-through
estimator) lets gradients flow through quant/dequant ops so scales are
learned. See spec §5.1-5.2 (lr0=1e-4, amp=False mandated at train layer).

Known limitation (spec §13): no forward_loop calibration — scales init
at default and train from there. Future iteration can add 512-image
COCO val calibration loop before training.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO
    from scripts._schemas import TrainingSpec


_QUANT_CONFIG_PRESETS = {
    "int8_default": "INT8_DEFAULT_CFG",
    # Future: "int8_smoothquant": "INT8_SMOOTHQUANT_CFG",
}


def _resolve_config(quant_config: str | None):
    import modelopt.torch.quantization as mtq

    key = quant_config or "int8_default"
    attr_name = _QUANT_CONFIG_PRESETS.get(key)
    if attr_name is None:
        raise ValueError(
            f"Unknown quant_config {key!r}. Known: "
            f"{list(_QUANT_CONFIG_PRESETS)}"
        )
    return getattr(mtq, attr_name)


def apply(yolo: "YOLO", spec: "TrainingSpec") -> None:
    import modelopt.torch.quantization as mtq

    cfg = _resolve_config(spec.quant_config)
    mtq.quantize(yolo.model, cfg)
    print(f"[modelopt_qat] quantize({spec.quant_config or 'int8_default'}) applied")


def finalize(yolo: "YOLO", spec: "TrainingSpec", out_pt: Path) -> None:
    import modelopt.torch.opt as mto

    mto.save(yolo.model, str(out_pt))
    print(f"[modelopt_qat] saved → {out_pt}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_modifiers_modelopt_qat.py -v
```
Expected: PASS, 3 tests (or SKIP if modelopt unavailable)

- [ ] **Step 5: Commit**

```bash
git add scripts/_modifiers/modelopt_qat.py tests/test_modifiers_modelopt_qat.py
git commit -m "feat(_modifiers): modelopt_qat INT8 QAT

spec §5.1-5.2, §6. mtq.quantize(INT8_DEFAULT_CFG) fake-quant 삽입,
mto.save/restore round-trip. forward_loop calibration은 MVP 생략
(spec §13 Known limitation).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `_train_core` + `scripts/train.py` entry

**Files:**
- Create: `scripts/_train_core.py`
- Create: `scripts/train.py`
- Create: `tests/test_train_dispatch.py`
- Create: `tests/test_train_skip_logic.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_train_dispatch.py`:
```python
"""train.py modifier dispatch tests (spec §12 data flow)."""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock


def _write_recipe(tmp_path, modifier: str, name: str = "test_r") -> Path:
    yaml_text = f"""
name: {name}
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
runtime:
  engine: tensorrt
  dtype: int8
technique:
  name: int8_test
  training:
    base_checkpoint: fake_base.pt
    epochs: 1
    modifier: {modifier}
    data_yaml: fake_data.yaml
measurement:
  dataset: coco
  num_images: 10
  warmup_iters: 1
  measure_iters: 1
  batch_sizes: [1]
"""
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml_text)
    return p


def test_dispatch_prune_24(tmp_path, monkeypatch):
    """train.py should route prune_24 recipes to the prune_24 modifier."""
    from scripts import train, _train_core
    recipe_path = _write_recipe(tmp_path, "prune_24")
    called = {}

    def fake_train(yolo, spec, run_name):
        called["trained"] = True
        return tmp_path / "dummy_runs" / run_name / "weights" / "last.pt"

    def fake_load_yolo(path):
        m = MagicMock()
        m.model = nn.Linear(4, 4)
        return m

    monkeypatch.setattr(_train_core, "_run_ultralytics_train", fake_train)
    monkeypatch.setattr(_train_core, "_load_yolo", fake_load_yolo)
    monkeypatch.setattr(_train_core, "ROOT", tmp_path)
    out = tmp_path / "trained_weights"
    out.mkdir()

    import scripts._modifiers.prune_24 as mod
    applied = []
    monkeypatch.setattr(mod, "apply", lambda y, s: applied.append("applied"))
    monkeypatch.setattr(mod, "finalize", lambda y, s, p: p.write_bytes(b"fake"))

    train.main([f"--recipe={recipe_path}", "--force"])
    assert applied == ["applied"]
    assert (out / "test_r.pt").exists()
    assert called.get("trained") is True


def test_dispatch_modelopt_qat(tmp_path, monkeypatch):
    from scripts import train, _train_core
    recipe_path = _write_recipe(tmp_path, "modelopt_qat", name="test_qat")

    def fake_train(yolo, spec, run_name):
        return tmp_path / "dummy" / "last.pt"

    def fake_load_yolo(path):
        m = MagicMock()
        m.model = nn.Linear(4, 4)
        return m

    monkeypatch.setattr(_train_core, "_run_ultralytics_train", fake_train)
    monkeypatch.setattr(_train_core, "_load_yolo", fake_load_yolo)
    monkeypatch.setattr(_train_core, "ROOT", tmp_path)
    (tmp_path / "trained_weights").mkdir()

    import scripts._modifiers.modelopt_qat as mod
    monkeypatch.setattr(mod, "apply", lambda y, s: None)
    monkeypatch.setattr(mod, "finalize", lambda y, s, p: p.write_bytes(b"fake"))

    train.main([f"--recipe={recipe_path}", "--force"])
    assert (tmp_path / "trained_weights" / "test_qat.pt").exists()


def test_rejects_recipe_without_training(tmp_path):
    from scripts import train
    recipe_path = tmp_path / "no_train.yaml"
    recipe_path.write_text("""
name: no_train
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
runtime:
  engine: tensorrt
  dtype: int8
technique:
  name: int8_test
measurement:
  dataset: coco
  num_images: 10
  warmup_iters: 1
  measure_iters: 1
  batch_sizes: [1]
""")
    with pytest.raises(SystemExit):
        train.main([f"--recipe={recipe_path}"])
```

Create `tests/test_train_skip_logic.py`:
```python
"""train.py skip / --force tests."""
import pytest
from pathlib import Path


def _write_recipe(tmp_path, name: str = "skip_r") -> Path:
    yaml_text = f"""
name: {name}
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
runtime:
  engine: tensorrt
  dtype: int8
technique:
  name: int8_test
  training:
    base_checkpoint: fake_base.pt
    epochs: 1
    modifier: prune_24
measurement:
  dataset: coco
  num_images: 10
  warmup_iters: 1
  measure_iters: 1
  batch_sizes: [1]
"""
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml_text)
    return p


def test_skip_when_output_exists(tmp_path, monkeypatch, capsys):
    from scripts import train, _train_core
    monkeypatch.setattr(_train_core, "ROOT", tmp_path)
    out_dir = tmp_path / "trained_weights"
    out_dir.mkdir()
    (out_dir / "skip_r.pt").write_bytes(b"existing")

    called = {}
    monkeypatch.setattr(_train_core, "train_with_modifier",
                        lambda r: called.setdefault("ran", True))
    recipe_path = _write_recipe(tmp_path)
    train.main([f"--recipe={recipe_path}"])
    captured = capsys.readouterr()
    assert "skip" in captured.out.lower()
    assert "ran" not in called


def test_force_overrides_skip(tmp_path, monkeypatch):
    from scripts import train, _train_core
    monkeypatch.setattr(_train_core, "ROOT", tmp_path)
    out_dir = tmp_path / "trained_weights"
    out_dir.mkdir()
    (out_dir / "skip_r.pt").write_bytes(b"existing")

    called = {}
    def fake_train(recipe):
        called["ran"] = True
        return out_dir / f"{recipe.name}.pt"
    monkeypatch.setattr(_train_core, "train_with_modifier", fake_train)

    recipe_path = _write_recipe(tmp_path)
    train.main([f"--recipe={recipe_path}", "--force"])
    assert called.get("ran") is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train_dispatch.py tests/test_train_skip_logic.py -v
```
Expected: FAIL — scripts.train module missing

- [ ] **Step 3: Implement `scripts/_train_core.py`**

Create `scripts/_train_core.py`:
```python
"""Common training loop shared by all modifier plugins.

Flow:
1. Load base YOLO from ``spec.base_checkpoint``.
2. Call modifier.apply(yolo, spec).
3. Run ``ultralytics YOLO.train(...)`` with recipe params (amp toggled
   per modifier to avoid modelopt fake-quant / AMP corruption).
4. Call modifier.finalize(yolo, spec, out_pt) to serialize. The in-memory
   ``yolo.model`` is used directly — we do NOT load ``last.pt`` so
   ultralytics' EMA-based best selection cannot leak validation data.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]

if TYPE_CHECKING:
    from ultralytics import YOLO
    from scripts._schemas import Recipe, TrainingSpec


# Modifiers that wrap model in modelopt fake-quant / sparsity modules.
# AMP (fp16 mixed precision) breaks modelopt scale correctness, so disable.
_MODELOPT_MODIFIERS = {"modelopt_sparsify", "modelopt_qat"}


def _load_yolo(path: str) -> "YOLO":
    from ultralytics import YOLO
    return YOLO(path)


def _resolve_base_checkpoint(spec: "TrainingSpec") -> Path:
    p = Path(spec.base_checkpoint)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"base_checkpoint not found: {p}. See README for best_qr.pt "
            f"placement."
        )
    return p


def _resolve_data_yaml(spec: "TrainingSpec") -> str:
    if spec.data_yaml:
        p = Path(spec.data_yaml)
        if not p.is_absolute():
            p = ROOT / p
        return str(p)
    env = os.environ.get("OMNI_TRAIN_YAML") or os.environ.get("OMNI_COCO_YAML")
    if env:
        return env
    return str(ROOT / "qr_barcode.yaml")


def _run_ultralytics_train(yolo: "YOLO", spec: "TrainingSpec",
                           run_name: str) -> Path:
    """Call ultralytics model.train(). Returns path to last.pt (unused for
    modelopt; kept for potential debugging). See spec §5.3 for why we
    don't consume this file."""
    smoke = os.environ.get("OMNI_TRAIN_SMOKE") == "1"
    kwargs = dict(
        data=_resolve_data_yaml(spec),
        epochs=1 if smoke else spec.epochs,
        batch=spec.batch,
        workers=int(os.environ.get("OMNI_TRAIN_WORKERS", spec.workers)),
        imgsz=spec.imgsz,
        lr0=spec.lr0,
        optimizer=spec.optimizer,
        seed=spec.seed,
        device=os.environ.get("OMNI_TRAIN_DEVICE", "0"),
        name=run_name,
        exist_ok=True,
        amp=spec.modifier not in _MODELOPT_MODIFIERS,
        verbose=True,
    )
    if smoke:
        kwargs["fraction"] = 0.1
    yolo.train(**kwargs)
    # ultralytics writes to runs/train/{run_name}/weights/{last,best}.pt
    return ROOT / "runs" / "train" / run_name / "weights" / "last.pt"


def _load_modifier(name: str):
    return importlib.import_module(f"scripts._modifiers.{name}")


def train_with_modifier(recipe: "Recipe") -> Path:
    spec = recipe.technique.training
    if spec is None:
        raise ValueError(f"recipe {recipe.name} has no training section")
    modifier = _load_modifier(spec.modifier)

    base = _resolve_base_checkpoint(spec)
    print(f"[train] loading base: {base}")
    yolo = _load_yolo(str(base))

    print(f"[train] applying modifier: {spec.modifier}")
    modifier.apply(yolo, spec)

    run_name = recipe.name
    print(f"[train] ultralytics model.train(epochs={spec.epochs}, "
          f"amp={spec.modifier not in _MODELOPT_MODIFIERS}) → runs/train/{run_name}")
    started = time.time()
    _run_ultralytics_train(yolo, spec, run_name)
    duration = time.time() - started

    out_dir = ROOT / "trained_weights"
    out_dir.mkdir(exist_ok=True)
    out_pt = out_dir / f"{recipe.name}.pt"

    print(f"[train] calling {spec.modifier}.finalize(out_pt={out_pt})")
    modifier.finalize(yolo, spec, out_pt)

    _write_train_json(out_dir / f"{recipe.name}.train.json", recipe, duration)
    return out_pt


def _write_train_json(path: Path, recipe: "Recipe", duration_s: float) -> None:
    spec = recipe.technique.training
    assert spec is not None
    path.write_text(json.dumps({
        "recipe": recipe.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(duration_s, 1),
        "base_checkpoint": spec.base_checkpoint,
        "epochs": spec.epochs,
        "modifier": spec.modifier,
        "lr0": spec.lr0,
        "amp": spec.modifier not in _MODELOPT_MODIFIERS,
        "notes": "val mAP during training is EMA-based and unreliable for "
                 "modelopt modifiers; use run_trt.py for authoritative eval",
    }, indent=2))
```

- [ ] **Step 4: Implement `scripts/train.py`**

Create `scripts/train.py`:
```python
"""CLI entry for QAT / sparsity fine-tuning.

Usage:
    python scripts/train.py --recipe recipes/17_modelopt_int8_qat.yaml
    python scripts/train.py --recipe recipes/07_trt_int8_sparsity.yaml --force

Produces ``trained_weights/{recipe.name}.pt`` + ``.train.json``.
Skips if output already exists (use ``--force`` to retrain).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._schemas import load_recipe  # noqa: E402
from scripts import _train_core  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--force", action="store_true",
                        help="retrain even if trained_weights/{name}.pt exists")
    args = parser.parse_args(argv)

    recipe = load_recipe(args.recipe)
    if recipe.technique.training is None:
        print(f"error: recipe {recipe.name!r} has no technique.training "
              f"section — nothing to train.", file=sys.stderr)
        sys.exit(1)

    out_dir = _train_core.ROOT / "trained_weights"
    out_dir.mkdir(exist_ok=True)
    out_pt = out_dir / f"{recipe.name}.pt"
    if out_pt.exists() and not args.force:
        print(f"[skip] {out_pt} already exists (use --force to retrain)")
        return 0

    trained = _train_core.train_with_modifier(recipe)
    print(f"✔ trained weights → {trained}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_train_dispatch.py tests/test_train_skip_logic.py -v
```
Expected: PASS, 5 tests

- [ ] **Step 6: Commit**

```bash
git add scripts/_train_core.py scripts/train.py tests/test_train_dispatch.py tests/test_train_skip_logic.py
git commit -m "feat(train): train.py CLI + _train_core dispatch loop

spec §2, §5, §7. modifier.apply → ultralytics model.train(amp per-modifier)
→ modifier.finalize. OMNI_TRAIN_SMOKE=1 (epochs=1, fraction=0.1) for dry-run.
Skip logic + --force.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `run_trt._resolve_weights` modelopt 경로 + in-memory YOLO 전달

**Files:**
- Modify: `scripts/run_trt.py` (여러 위치)
- Modify: `tests/test_run_trt_trained_weights.py` (추가 테스트)

- [ ] **Step 1: Extend failing test for modelopt path**

Append to `tests/test_run_trt_trained_weights.py`:
```python
def test_resolve_weights_modelopt_returns_yolo_like(tmp_path, monkeypatch):
    """For modelopt_* modifier, _resolve_weights returns a YOLO instance
    (not a string path) so downstream _export_onnx can consume it."""
    from scripts import run_trt
    monkeypatch.setattr(run_trt, "ROOT", tmp_path)
    tw = tmp_path / "trained_weights"
    tw.mkdir()
    trained = tw / "test_recipe.pt"

    # Create a minimal modelopt-savable checkpoint
    import torch.nn as nn
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    m = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())
    mtq.quantize(m, mtq.INT8_DEFAULT_CFG)
    mto.save(m, str(trained))

    recipe = _make_recipe(
        training=TrainingSpec(
            base_checkpoint="yolo26n.pt",
            epochs=5,
            modifier="modelopt_qat",
        ),
        weights="yolo26n.pt",
    )

    class _FakeYolo:
        def __init__(self, path):
            self.model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())
            self.ckpt_path = path

    def fake_yolo(path):
        return _FakeYolo(path)

    monkeypatch.setattr(run_trt, "_load_yolo_for_restore", fake_yolo)
    out = run_trt._resolve_weights(recipe)
    assert hasattr(out, "model")  # YOLO-like
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
pytest tests/test_run_trt_trained_weights.py::test_resolve_weights_modelopt_returns_yolo_like -v
```
Expected: FAIL — `NotImplementedError` raised in `_resolve_weights` for modelopt modifier

- [ ] **Step 3: Update `_resolve_weights()` + add helper in `scripts/run_trt.py`**

Replace the `_resolve_weights` function body (from Task 2) with:

```python
def _load_yolo_for_restore(base_path: str):
    """Load a plain YOLO instance to serve as the architecture skeleton
    for mto.restore()."""
    from ultralytics import YOLO
    return YOLO(base_path)


def _resolve_weights(recipe: Recipe):
    """Return runner input: either a path string, or a YOLO-like object
    whose ``.model`` has modelopt modules restored.

    - No training: str path (recipe.model.weights).
    - prune_24 trained: str path to trained_weights/{name}.pt (plain
      ultralytics checkpoint after prune.remove()).
    - modelopt_sparsify / modelopt_qat: YOLO instance with mto.restore()
      applied; downstream _export_onnx accepts YOLO objects directly
      (see the ``is_path`` branch near line 70).
    """
    if recipe.technique.training is None:
        return recipe.model.weights
    trained = ROOT / "trained_weights" / f"{recipe.name}.pt"
    if not trained.exists():
        raise RuntimeError(
            f"Recipe {recipe.name!r} requires training but {trained} is "
            f"missing. Run: python scripts/train.py --recipe "
            f"recipes/{recipe.name}.yaml"
        )
    modifier = recipe.technique.training.modifier
    if modifier == "prune_24":
        return str(trained)
    if modifier in ("modelopt_sparsify", "modelopt_qat"):
        import modelopt.torch.opt as mto
        yolo = _load_yolo_for_restore(recipe.model.weights)
        mto.restore(yolo.model, str(trained))
        return yolo
    raise RuntimeError(f"unexpected modifier: {modifier!r}")
```

- [ ] **Step 4: Wire `_resolve_weights()` into `main()` of run_trt.py**

Near the top of `main()` (after `recipe = load_recipe(args.recipe)`), add:

```python
    # spec §6 + Task 8: resolve weights (may return YOLO instance for
    # modelopt trained recipes).
    resolved = _resolve_weights(recipe)
    if not isinstance(resolved, (str, Path)):
        # modelopt trained: downstream code that expects a string path
        # needs to branch. _export_onnx already accepts YOLO; other call
        # sites fall back to ckpt_path for filename stems.
        recipe._resolved_yolo = resolved  # type: ignore[attr-defined]
    else:
        recipe.model.weights = str(resolved)
```

Then update export call sites to prefer the YOLO instance if present. Grep for `recipe.model.weights` and add the pattern:

```python
weights_in = getattr(recipe, "_resolved_yolo", None) or recipe.model.weights
```

Specifically update these lines in `run_trt.py` (use `Edit`):
- Line ~173: `yolo = _apply_modelopt_sparsify(recipe.model.weights, imgsz)` → guard with `if getattr(recipe, "_resolved_yolo", None): yolo = recipe._resolved_yolo; else: yolo = _apply_modelopt_sparsify(recipe.model.weights, imgsz)` (preserve existing untrained path)
- Line ~178, 284, 688: `_export_onnx(recipe.model.weights, ...)` → `_export_onnx(getattr(recipe, "_resolved_yolo", None) or recipe.model.weights, ...)`
- Line ~459: `yolo = YOLO(recipe.model.weights)` → `yolo = getattr(recipe, "_resolved_yolo", None) or YOLO(recipe.model.weights)`

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -q
```
Expected: all pass (new test + existing tests)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_trt.py tests/test_run_trt_trained_weights.py
git commit -m "feat(run_trt): _resolve_weights modelopt path (mto.restore)

spec §6. modelopt_sparsify/qat 모드에서 YOLO 인스턴스로 restore된 모델 전달.
_export_onnx 등 call site는 YOLO or str path 이미 수용. prune_24는 str 경로.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Recipe YAML — #17 신규 + #07 #11 패치

**Files:**
- Create: `recipes/17_modelopt_int8_qat.yaml`
- Modify: `recipes/07_trt_int8_sparsity.yaml`
- Modify: `recipes/11_modelopt_int8_sparsity.yaml`

- [ ] **Step 1: Create `recipes/17_modelopt_int8_qat.yaml`**

```yaml
name: modelopt_int8_qat
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
  name: int8_qat
  source: modelopt
  calibrator: null
  calibration_samples: null
  calibration_dataset: null
  training:
    base_checkpoint: best_qr.pt
    epochs: 30
    batch: 8
    workers: 4
    imgsz: 640
    lr0: 0.0001
    optimizer: AdamW
    seed: 42
    data_yaml: qr_barcode.yaml
    modifier: modelopt_qat
    quant_config: int8_default
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

- [ ] **Step 2: Patch `recipes/07_trt_int8_sparsity.yaml`**

Insert at the end of `technique:` block (after `calibration_seed: 42`):

```yaml
  training:
    base_checkpoint: best_qr.pt
    epochs: 60
    batch: 8
    workers: 4
    imgsz: 640
    lr0: 0.001
    optimizer: AdamW
    seed: 42
    data_yaml: qr_barcode.yaml
    modifier: prune_24
    prune_amount: 0.5
```

- [ ] **Step 3: Patch `recipes/11_modelopt_int8_sparsity.yaml`**

Insert at the end of `technique:` block (after the v1.2 sparsity_preprocess comment):

```yaml
  training:
    base_checkpoint: best_qr.pt
    epochs: 60
    batch: 8
    workers: 4
    imgsz: 640
    lr0: 0.001
    optimizer: AdamW
    seed: 42
    data_yaml: qr_barcode.yaml
    modifier: modelopt_sparsify
```

- [ ] **Step 4: Verify recipes load**

```bash
python -c "
from scripts._schemas import load_recipe
for r in ['07_trt_int8_sparsity', '11_modelopt_int8_sparsity', '17_modelopt_int8_qat']:
    recipe = load_recipe(f'recipes/{r}.yaml')
    assert recipe.technique.training is not None, f'{r}: training missing'
    print(f'{r}: modifier={recipe.technique.training.modifier}, epochs={recipe.technique.training.epochs}')
"
```
Expected:
```
07_trt_int8_sparsity: modifier=prune_24, epochs=60
11_modelopt_int8_sparsity: modifier=modelopt_sparsify, epochs=60
17_modelopt_int8_qat: modifier=modelopt_qat, epochs=30
```

- [ ] **Step 5: Commit**

```bash
git add recipes/07_trt_int8_sparsity.yaml recipes/11_modelopt_int8_sparsity.yaml recipes/17_modelopt_int8_qat.yaml
git commit -m "feat(recipes): #17 modelopt_int8_qat + training section for #07 #11

spec §8. #17 QAT 30ep lr0=1e-4, sparsity 2개 60ep lr0=1e-3. best_qr.pt
기반 recovery. data_yaml=qr_barcode.yaml 명시.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Batch scripts + Makefile + configs

**Files:**
- Create: `scripts/run_qr_train_batch.sh`
- Modify: `scripts/run_qr_batch.sh`
- Modify: `Makefile`
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Create `scripts/run_qr_train_batch.sh`**

```bash
#!/usr/bin/env bash
# Train QAT / sparsity recipes with the QR/Barcode fine-tuned checkpoint.
# Produces trained_weights/*.pt, skips existing.
set -u

export OMNI_COCO_YAML="$PWD/qr_barcode.yaml"
export OMNI_WEIGHTS_OVERRIDE="$PWD/best_qr.pt"

mkdir -p trained_weights

TRAINING_RECIPES=(
    07_trt_int8_sparsity
    11_modelopt_int8_sparsity
    17_modelopt_int8_qat
)

for r in "${TRAINING_RECIPES[@]}"; do
    out="trained_weights/${r}.pt"
    if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        continue
    fi
    echo "=== [$(date +%H:%M:%S)] training $r ==="
    python scripts/train.py --recipe "recipes/${r}.yaml"
    ec=$?
    echo "--- exit=$ec ---"
done

echo "all training done."
```

Mark executable:
```bash
chmod +x scripts/run_qr_train_batch.sh
```

- [ ] **Step 2: Patch `scripts/run_qr_batch.sh` — add #07, #11, #17 to ORDER and RECIPES**

In `scripts/run_qr_batch.sh`, find the `declare -A RECIPES=(` block and add entries:
```bash
    [07_trt_int8_sparsity]="run_trt"
    [11_modelopt_int8_sparsity]="run_trt"
    [17_modelopt_int8_qat]="run_trt"
```

Then in `ORDER=(` block, replace the existing list with one that includes #07 after #06, #11 after #10, and #17 after #16:
```bash
ORDER=(
    00_trt_fp32 00_trt_fp32_tf32
    01_pytorch_fp32 02_torchcompile_fp16
    03_ort_cuda_fp16 04_ort_trt_fp16
    05_trt_fp16 06_trt_int8_ptq
    07_trt_int8_sparsity
    08_modelopt_int8_ptq 09_modelopt_int8_entropy
    10_modelopt_int8_percentile
    11_modelopt_int8_sparsity
    12_modelopt_int8_mixed
    13_ort_int8_minmax 14_ort_int8_entropy
    15_ort_int8_percentile 16_ort_int8_distribution
    17_modelopt_int8_qat
    20_brevitas_int8_percentile 21_brevitas_int8_mse
)
```

- [ ] **Step 3: Update `Makefile`**

Open `Makefile`. Find the `PARKED` variable definition. Current value (after Brevitas work):
```makefile
PARKED := 22_brevitas_int8_entropy 19_inc_int8_qat 18_inc_int8_mixed 17_inc_int8_qat_sparsity
```
(verify exact current form via `grep PARKED Makefile` first)

Replace with:
```makefile
PARKED := 22_brevitas_int8_entropy
```

Add after the last target, before any `.PHONY` aggregation:
```makefile
train-qr:
	bash scripts/run_qr_train_batch.sh

.PHONY: train-qr
```

- [ ] **Step 4: Update `.gitignore`**

Append to `.gitignore`:
```gitignore
# Training artifacts (spec 2026-04-20 §3)
trained_weights/
runs/
best_qr.pt
```

- [ ] **Step 5: Update `pyproject.toml`**

Find the `[project.optional-dependencies]` block and add:
```toml
modelopt = ["nvidia-modelopt>=0.15"]
```

Also update the `all` extras line to include modelopt:
```toml
all = [
    "omnioptimizer[torch,onnx,trt,brevitas,modelopt]",
]
```

- [ ] **Step 6: Update `README.md` — add fine-tuned checkpoint section**

Find the top-level reproduction section (likely near "Quick start" or end of document). Append or insert:
```markdown
## QR/Barcode fine-tuned checkpoint

Recipes `#07`, `#11`, and `#17` fine-tune on a 2-class (barcode, qrcode) YOLO26n
checkpoint. The file is gitignored; copy it locally before running training /
QR-specific evaluation:

\`\`\`bash
cp "C:/Users/yeste/OneDrive/Desktop/QR_Barcode/QR_Barcode_detection/yolo26n_qrcode_barcode_bg/weights/best.pt" ./best_qr.pt
\`\`\`

External users can substitute any 2-class (nc=2) ultralytics checkpoint.
Training:
\`\`\`bash
bash scripts/run_qr_train_batch.sh
\`\`\`
produces `trained_weights/{recipe}.pt` (~2 hours total on RTX 3060 Laptop).

Smoke dry-run (1 epoch, 10% data, ~3 minutes):
\`\`\`bash
OMNI_TRAIN_SMOKE=1 bash scripts/run_qr_train_batch.sh
\`\`\`
```

- [ ] **Step 7: Verify recipes still load + tests still pass**

```bash
pytest tests/ -q
```
Expected: all pass (no regressions).

- [ ] **Step 8: Commit**

```bash
git add scripts/run_qr_train_batch.sh scripts/run_qr_batch.sh Makefile .gitignore pyproject.toml README.md
git commit -m "chore(infra): train batch + Makefile + gitignore + pyproject modelopt

spec §12, §15. run_qr_train_batch.sh 신규. run_qr_batch.sh ORDER/RECIPES에
#07 #11 #17 추가. Makefile PARKED=#22만, train-qr 타겟. .gitignore +=
trained_weights/ runs/ best_qr.pt. pyproject modelopt extras.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Smoke dry-run 검증 (실제 학습 전 파이프라인 검증)

**Files:** (변경 없음, 검증만)

- [ ] **Step 1: Verify `best_qr.pt` is in place**

```bash
ls -la best_qr.pt || echo "MISSING — copy per README"
```
Expected: file exists, ~6MB.

- [ ] **Step 2: Smoke run for `#17` (fastest: 1 epoch, 10% data, modelopt QAT)**

```bash
OMNI_TRAIN_SMOKE=1 python scripts/train.py --recipe recipes/17_modelopt_int8_qat.yaml --force
```
Expected: completes in ~2-3 minutes. Prints `[modelopt_qat] quantize(int8_default) applied` and `[modelopt_qat] saved → trained_weights/17_modelopt_int8_qat.pt`. Returns 0.

- [ ] **Step 3: Verify output files**

```bash
ls -la trained_weights/17_modelopt_int8_qat.pt trained_weights/17_modelopt_int8_qat.train.json
cat trained_weights/17_modelopt_int8_qat.train.json
```
Expected: `.pt` exists (~6MB), `.train.json` has modifier/epochs/duration_s fields.

- [ ] **Step 4: Smoke eval — run TRT on trained weights**

```bash
OMNI_COCO_YAML="$PWD/qr_barcode.yaml" \
OMNI_WEIGHTS_OVERRIDE="$PWD/best_qr.pt" \
python scripts/run_trt.py --recipe recipes/17_modelopt_int8_qat.yaml --out /tmp/smoke_17.json
```
Expected: completes without crash. Produces a JSON with `meets_constraints` populated (likely `False` because smoke-trained model has terrible mAP). `_resolve_weights` code path exercised.

- [ ] **Step 5: Smoke run for `#07` (prune_24)**

```bash
rm -f trained_weights/07_trt_int8_sparsity.pt
OMNI_TRAIN_SMOKE=1 python scripts/train.py --recipe recipes/07_trt_int8_sparsity.yaml --force
```
Expected: completes, prints `[prune_24] applied 2:4 mask to N modules` and `[prune_24] finalized N modules, 2:4 pattern verified`.

- [ ] **Step 6: Smoke run for `#11` (modelopt_sparsify)**

```bash
rm -f trained_weights/11_modelopt_int8_sparsity.pt
OMNI_TRAIN_SMOKE=1 python scripts/train.py --recipe recipes/11_modelopt_int8_sparsity.yaml --force
```
Expected: completes, prints `[modelopt_sparsify] sparsify(sparse_magnitude, 2:4) applied`.

- [ ] **Step 7: Clean up smoke artifacts**

```bash
rm -f trained_weights/*.pt trained_weights/*.train.json /tmp/smoke_17.json
rm -rf runs/train/
```

- [ ] **Step 8: Commit (checkpoint marker, no code changes)**

```bash
git commit --allow-empty -m "chore: smoke verification pass (Task 11)

3개 modifier 모두 OMNI_TRAIN_SMOKE=1로 1 epoch 파이프라인 통과.
run_trt._resolve_weights modelopt 경로 exercise 확인.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: 실제 학습 실행 (~2시간, 수동 실행)

**Files:** (학습 산출물, git 추적 안 함)

- [ ] **Step 1: Ensure `best_qr.pt` in place and environment healthy**

```bash
ls -la best_qr.pt
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```
Expected: file exists, CUDA available.

- [ ] **Step 2: Run training batch**

```bash
bash scripts/run_qr_train_batch.sh 2>&1 | tee /tmp/train_batch.log
```
Expected: 3 recipes train sequentially. Total ~2 hours on RTX 3060 Laptop:
- #17 modelopt_int8_qat: ~25 min
- #07 trt_int8_sparsity: ~45 min
- #11 modelopt_int8_sparsity: ~45 min

Each produces `trained_weights/{name}.pt` + `.train.json`.

- [ ] **Step 3: Verify artifacts**

```bash
ls -la trained_weights/
```
Expected: 6 files (3 `.pt` + 3 `.train.json`).

- [ ] **Step 4: Commit marker (no tracked changes)**

```bash
git commit --allow-empty -m "chore: QR training complete (Task 12)

trained_weights/{07,11,17}.pt generated, not tracked.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: 평가 배치 실행 (~1시간, 수동 실행)

**Files:** `results_qr/*.json`

- [ ] **Step 1: Run evaluation batch**

```bash
bash scripts/run_qr_batch.sh 2>&1 | tee /tmp/eval_batch.log
```
Expected: existing 18 results skip (already in `results_qr/`), only 3 new ones run (#07, #11, #17). Each takes 3-10 min.

- [ ] **Step 2: Verify new results**

```bash
ls -la results_qr/07_*.json results_qr/11_*.json results_qr/17_*.json
python -c "
import json
for f in ['07_trt_int8_sparsity', '11_modelopt_int8_sparsity', '17_modelopt_int8_qat']:
    r = json.load(open(f'results_qr/{f}.json'))
    print(f'{f}: mAP@50-95={r[\"accuracy\"][\"map_50_95\"]:.4f}, bs1={r[\"throughput_fps\"][\"bs1\"]:.1f}, meets={r[\"meets_constraints\"]}')
"
```
Expected: 3 JSONs exist. Print summary (mAP + fps + meets_constraints).

- [ ] **Step 3: Regenerate `results_qr/_summary.json`**

```bash
python scripts/recommend.py --results-dir results_qr --out /tmp/qr_report.md --exclude 22_brevitas_int8_entropy
```
Expected: generates summary + report. Winner may change depending on QAT / sparsity mAP.

- [ ] **Step 4: Commit results**

```bash
git add results_qr/07_trt_int8_sparsity.json results_qr/11_modelopt_int8_sparsity.json results_qr/17_modelopt_int8_qat.json results_qr/_summary.json
git commit -m "feat(results_qr): +#07 +#11 +#17 trained recipe measurements

3 trained-recipe results + regenerated summary.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: `docs/qr_barcode_eval_v2.md` Training-based recipes 섹션 append

**Files:** `docs/qr_barcode_eval_v2.md`

- [ ] **Step 1: Append new section**

Find the `## Follow-ups` section. Insert a new `## Training-based recipes` section *before* it (after "새 Recommendation"):

```markdown
## Training-based recipes (신규, 2026-04-20)

baseline: `00 trt_fp32` mAP@50=0.9893, mAP@50-95=0.9328. best_qr.pt 기반
QAT/sparsity recovery fine-tune.

| # | Recipe | base | epochs | p50 ms | bs1 fps | mAP@50 | ΔmAP@50-95 (%p) | 비고 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 07 | trt_int8_sparsity | best_qr.pt | 60 | {fill} | {fill} | {fill} | {fill} | prune_24 + SAT |
| 11 | modelopt_int8_sparsity | best_qr.pt | 60 | {fill} | {fill} | {fill} | {fill} | modelopt sparsify 2:4 |
| 17 | modelopt_int8_qat | best_qr.pt | 30 | {fill} | {fill} | {fill} | {fill} | modelopt QAT, lr0=1e-4 |

**관찰** (실제 값으로 교체):
- `#17 modelopt_int8_qat`: QAT가 PTQ 대비 mAP@50-95 손실 {fill}%p — calibration forward_loop 생략에도 불구하고 recovery.
- `#11 modelopt_int8_sparsity`: modelopt sparsify 2:4 가 실제로 TRT SPARSE_WEIGHTS kernel 사용 {fill}.
- `#07 trt_int8_sparsity`: magnitude pruning + SAT 가 modelopt 대비 {fill}.

설명:
- training 시 in-memory `yolo.model` 직접 직렬화 (best.pt 경유 X) → val leak 차단.
- modelopt modifier는 `amp=False` 강제 (fake quant scale 무결성).
- 2:4 패턴은 `_verify_2_4_pattern()` 로 finalize 단계에서 assert.
```

Replace `{fill}` placeholders with actual values from `results_qr/{07,11,17}_*.json`. For each: `p50_ms = r["latency_ms"]["p50"]`, `bs1_fps = r["throughput_fps"]["bs1"]`, `map_50 = r["accuracy"]["map_50"]`, delta = `r["accuracy"]["map_50_95"] - 0.9328`.

Shell helper:
```bash
python -c "
import json
for n in ['07_trt_int8_sparsity', '11_modelopt_int8_sparsity', '17_modelopt_int8_qat']:
    r = json.load(open(f'results_qr/{n}.json'))
    p50 = r['latency_ms']['p50']
    bs1 = r['throughput_fps']['bs1']
    m50 = r['accuracy']['map_50']
    m5095 = r['accuracy']['map_50_95']
    delta = (m5095 - 0.9328) * 100
    print(f'{n}: p50={p50:.2f}, bs1={bs1:.1f}, mAP50={m50:.4f}, delta={delta:+.2f}')
"
```
Copy the printed values into the table, replacing `{fill}` cells.

Also update the `## Follow-ups` section to remove the entry about parked sparsity recipes (since they're now active):
- Remove line: "파킹된 sparsity 레시피 (#07/#11) 는 학습 파이프라인 붙인 후 재평가 예정."

- [ ] **Step 2: Commit doc update**

```bash
git add docs/qr_barcode_eval_v2.md
git commit -m "docs(qr_barcode_eval_v2): Training-based recipes 섹션 (Task 14)

#07, #11, #17 실측 결과 + 관찰. Follow-up 중 'sparsity 파킹' 항목 제거.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: `docs/architecture.md` 갱신 + 최종 검증

**Files:** `docs/architecture.md`

- [ ] **Step 1: Grep current recipe counts**

```bash
grep -n "recipe\|active\|parked\|Wave" docs/architecture.md | head -40
```
Note current statements like "18 active recipes", Wave 3/4 sections.

- [ ] **Step 2: Update recipe counts + Wave 5 section**

- Replace any "18 active" / "20 recipes" phrasing to reflect the new count: **20 active + 1 parked** (#22 brevitas_int8_entropy). Use `Edit` with exact current string.
- Before the `## Windows-specific gotchas` section (or a similar trailing section), add:

```markdown
## Wave 5 — Training pipeline (2026-04-20)

QAT/sparsity 레시피는 recipe YAML의 ``technique.training`` 섹션으로 학습
파라미터를 기술하고, ``scripts/train.py`` 가 modifier별 전후훅을 실행.

- ``#07 trt_int8_sparsity``: ``prune_24`` modifier — magnitude 2:4 pruning
  + SAT 60 epochs.
- ``#11 modelopt_int8_sparsity``: ``modelopt_sparsify`` modifier —
  ``modelopt.torch.sparsity.sparsify(sparse_magnitude, 2:4)`` 60 epochs.
- ``#17 modelopt_int8_qat``: ``modelopt_qat`` modifier — ``mtq.quantize(INT8_DEFAULT_CFG)``
  fake quant 삽입 + 30 epochs QAT at ``lr0=1e-4``.

설계 원칙:
- Modifier 플러그인 (`scripts/_modifiers/{prune_24,modelopt_sparsify,modelopt_qat}.py`)
  각자 ``apply/finalize`` 노출.
- In-memory ``yolo.model`` 직접 직렬화 (best.pt 경유 X) → validation leak 방지.
- modelopt modifier는 AMP 비활성 + ``mto.save/restore``로 wrapped state 보존.
- 출력: ``trained_weights/{recipe.name}.pt`` (gitignored).

재현은 ``bash scripts/run_qr_train_batch.sh`` → ``bash scripts/run_qr_batch.sh``.
설계 스펙: ``docs/superpowers/specs/2026-04-20-qr-training-pipeline-design.md``.
```

- [ ] **Step 3: Run acceptance-criteria checks**

Per spec §19 checklist:
```bash
# All tests pass
pytest tests/ -q

# Smoke ran (implicit from Task 11 — just verify runner accepts --help)
python scripts/train.py --help

# Trained artifacts exist
ls trained_weights/07_*.pt trained_weights/11_*.pt trained_weights/17_*.pt

# 3 new JSONs produced
ls results_qr/07_*.json results_qr/11_*.json results_qr/17_*.json

# Docs updated
grep "Training-based recipes" docs/qr_barcode_eval_v2.md
grep "Wave 5" docs/architecture.md

# At least one new recipe meets constraints
python -c "
import json
any_met = False
for n in ['07_trt_int8_sparsity', '11_modelopt_int8_sparsity', '17_modelopt_int8_qat']:
    r = json.load(open(f'results_qr/{n}.json'))
    if r['meets_constraints']:
        any_met = True
        print(f'{n}: meets_constraints=True')
assert any_met, 'spec §19: at least one new recipe must meet_constraints'
print('acceptance: PASS')
"

# make all still completes (degrade-don't-crash)
make all 2>&1 | tail -5
```
Expected: all checks pass.

- [ ] **Step 4: Final commit**

```bash
git add docs/architecture.md
git commit -m "docs(architecture): Wave 5 training pipeline section

spec §14. recipe 개수 갱신 (20 active + 1 parked), Wave 5 설계 요약,
modifier 플러그인 구조 + mto.save/restore 직렬화 원칙 명시.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Acceptance Summary

Upon completion of all 15 tasks:

- [ ] `pytest tests/ -q` — 전체 pass (신규 ~20 개 테스트 포함, 기존 23 개 유지)
- [ ] `trained_weights/{07,11,17}*.pt` + `.train.json` 존재
- [ ] `results_qr/{07,11,17}*.json` 생성됨
- [ ] `docs/qr_barcode_eval_v2.md` "Training-based recipes" 섹션 실측값 기술
- [ ] `docs/architecture.md` Wave 5 섹션 존재
- [ ] 최소 1개 신규 recipe `meets_constraints=True`
- [ ] `make all` 완주

**Estimated effort:**
- Tasks 1–10 (코드/설정): 4–6 시간 (TDD, 테스트 포함)
- Task 11 (smoke): ~15분
- Task 12 (실제 학습): ~2시간 (사용자 수동)
- Task 13 (평가): ~45분 (사용자 수동)
- Tasks 14–15 (문서): 30분

총 ~8–10시간 실개발 + ~3시간 무인 학습/평가.
