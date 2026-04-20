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
    result = run_trt._resolve_weights(recipe)
    assert Path(result) == trained
