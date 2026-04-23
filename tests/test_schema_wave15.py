"""Wave 15 D4.1 — MeasurementSpec.build_ceiling_s schema tests.

Covers field bounds + legacy Result/Recipe JSON compatibility (pre-Wave 15
JSON files must still parse clean).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripts._schemas import MeasurementSpec, Recipe


def _minimal_recipe(**measurement_overrides) -> dict:
    return {
        "name": "test_wave15",
        "model": {"family": "yolo26", "variant": "n", "weights": "yolo26n.pt"},
        "runtime": {"engine": "tensorrt", "dtype": "fp16"},
        "technique": {"name": "none"},
        "measurement": {
            "dataset": "coco_val2017",
            "num_images": 500,
            "warmup_iters": 100,
            "measure_iters": 100,
            "batch_sizes": [1, 8],
            **measurement_overrides,
        },
    }


def test_build_ceiling_s_defaults_to_none():
    m = MeasurementSpec(
        dataset="coco_val2017",
        num_images=1,
        warmup_iters=1,
        measure_iters=1,
        batch_sizes=[1],
    )
    assert m.build_ceiling_s is None


def test_build_ceiling_s_accepts_positive_int():
    for v in (1, 300, 600, 900, 1200, 3600):
        m = MeasurementSpec(
            dataset="d",
            num_images=1,
            warmup_iters=1,
            measure_iters=1,
            batch_sizes=[1],
            build_ceiling_s=v,
        )
        assert m.build_ceiling_s == v


def test_build_ceiling_s_rejects_zero():
    with pytest.raises(ValidationError):
        MeasurementSpec(
            dataset="d",
            num_images=1,
            warmup_iters=1,
            measure_iters=1,
            batch_sizes=[1],
            build_ceiling_s=0,
        )


def test_build_ceiling_s_rejects_negative():
    with pytest.raises(ValidationError):
        MeasurementSpec(
            dataset="d",
            num_images=1,
            warmup_iters=1,
            measure_iters=1,
            batch_sizes=[1],
            build_ceiling_s=-1,
        )


def test_recipe_with_ceiling_900_parses():
    r = Recipe.model_validate(_minimal_recipe(build_ceiling_s=900))
    assert r.measurement.build_ceiling_s == 900


def test_recipe_without_ceiling_still_valid():
    r = Recipe.model_validate(_minimal_recipe())
    assert r.measurement.build_ceiling_s is None


def test_legacy_measurement_json_without_build_ceiling_parses():
    """Pre-Wave 15 recipe YAML / JSON must parse clean without the new field."""
    legacy = {
        "dataset": "coco_val2017",
        "num_images": 500,
        "warmup_iters": 100,
        "measure_iters": 100,
        "batch_sizes": [1, 8],
        "input_size": 640,
        "gpu_clock_lock": True,
        "seed": 42,
    }
    m = MeasurementSpec.model_validate(legacy)
    assert m.build_ceiling_s is None
    assert m.thread_count is None  # Wave 6 field
    assert m.iter_cooldown_ms is None  # Wave 6 field


def test_wave14_builder_opt_and_wave15_ceiling_compose():
    """Recipe carrying both Wave 14's builder_optimization_level and Wave 15's
    build_ceiling_s must validate — this is the target combo for INT8 opt-in
    recipes where opt_level=5 pushes build_time_s above the legacy 600s mark.
    """
    d = _minimal_recipe(build_ceiling_s=1200)
    d["runtime"]["builder_optimization_level"] = 5
    r = Recipe.model_validate(d)
    assert r.runtime.builder_optimization_level == 5
    assert r.measurement.build_ceiling_s == 1200
