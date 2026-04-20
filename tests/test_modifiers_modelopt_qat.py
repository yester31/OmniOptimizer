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
    yolo2 = _FakeYolo(_tiny_net())
    mto.restore(yolo2.model, str(out))
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


def test_apply_raises_when_no_modules_wrapped():
    """Defensive guard: quantize on a no-Conv no-Linear model should not
    produce a silently un-quantized checkpoint."""
    from scripts._modifiers import modelopt_qat
    from scripts._schemas import TrainingSpec

    class _NoConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.ReLU()  # no conv, no linear
        def forward(self, x):
            return self.a(x)

    yolo = _FakeYolo(_NoConvNet())
    spec = TrainingSpec(
        base_checkpoint="x.pt", epochs=1,
        modifier="modelopt_qat",
    )
    # Allow either our guard or modelopt to raise
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        modelopt_qat.apply(yolo, spec)
