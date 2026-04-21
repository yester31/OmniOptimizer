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
    # _resolve_weights lives in scripts._weights_io (Wave 6 refactor).
    # run_trt.py re-imports it, so patches must target the source module.
    from scripts import _weights_io, run_trt
    monkeypatch.setattr(_weights_io, "ROOT", tmp_path)
    recipe = _make_recipe(training=TrainingSpec(
        base_checkpoint="best_qr.pt", epochs=5, modifier="prune_24",
    ))
    with pytest.raises(RuntimeError, match="requires training"):
        run_trt._resolve_weights(recipe)


def test_resolve_weights_prune_24_returns_trained_path(tmp_path, monkeypatch):
    from scripts import _weights_io, run_trt
    monkeypatch.setattr(_weights_io, "ROOT", tmp_path)
    tw = tmp_path / "trained_weights"
    tw.mkdir()
    trained = tw / "test_recipe.pt"
    trained.write_bytes(b"fake")
    recipe = _make_recipe(training=TrainingSpec(
        base_checkpoint="best_qr.pt", epochs=5, modifier="prune_24",
    ))
    result = run_trt._resolve_weights(recipe)
    assert Path(result) == trained


def test_resolve_weights_modelopt_returns_yolo_like(tmp_path, monkeypatch):
    """For modelopt_* modifier, _resolve_weights returns a YOLO instance
    (not a string path) so downstream _export_onnx can consume it."""
    import torch.nn as nn

    modelopt_torch_opt = pytest.importorskip("modelopt.torch.opt")
    modelopt_torch_quantization = pytest.importorskip("modelopt.torch.quantization")

    from scripts import _weights_io, run_trt
    monkeypatch.setattr(_weights_io, "ROOT", tmp_path)
    tw = tmp_path / "trained_weights"
    tw.mkdir()
    trained = tw / "test_recipe.pt"

    # Create a minimal modelopt-savable checkpoint
    m = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())
    modelopt_torch_quantization.quantize(m, modelopt_torch_quantization.INT8_DEFAULT_CFG)
    modelopt_torch_opt.save(m, str(trained))

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

    # Patch the source module's _load_yolo_for_restore so the inner call site
    # picks up the fake. Patching run_trt's re-export wouldn't reach it.
    monkeypatch.setattr(_weights_io, "_load_yolo_for_restore", fake_yolo)
    out = run_trt._resolve_weights(recipe)
    assert hasattr(out, "model")  # YOLO-like
