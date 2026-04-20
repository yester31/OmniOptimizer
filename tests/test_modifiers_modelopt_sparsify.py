"""modelopt_sparsify modifier tests (spec §5, §6)."""
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
    yolo2 = _FakeYolo(_tiny_net())
    mto.restore(yolo2.model, str(out))
    assert mto.modelopt_state(yolo2.model) is not None
