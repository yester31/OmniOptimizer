"""Wave 14 schema tests: RuntimeSpec.builder_optimization_level + Result.build_time_s."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripts._schemas import AccuracyStats, LatencyStats, Recipe, Result, RuntimeSpec


def _minimal_recipe(**runtime_overrides) -> dict:
    return {
        "name": "test",
        "model": {"family": "yolo26", "variant": "n", "weights": "yolo26n.pt"},
        "runtime": {"engine": "tensorrt", "dtype": "fp16", **runtime_overrides},
        "technique": {"name": "none"},
        "measurement": {
            "dataset": "coco_val2017",
            "num_images": 500,
            "warmup_iters": 100,
            "measure_iters": 100,
            "batch_sizes": [1, 8],
        },
    }


def test_builder_optimization_level_defaults_to_none():
    rt = RuntimeSpec(engine="tensorrt", dtype="fp16")
    assert rt.builder_optimization_level is None


def test_builder_optimization_level_accepts_0_through_5():
    for lvl in (0, 1, 2, 3, 4, 5):
        rt = RuntimeSpec(engine="tensorrt", dtype="fp16", builder_optimization_level=lvl)
        assert rt.builder_optimization_level == lvl


def test_builder_optimization_level_rejects_6():
    with pytest.raises(ValidationError):
        RuntimeSpec(engine="tensorrt", dtype="fp16", builder_optimization_level=6)


def test_builder_optimization_level_rejects_negative():
    with pytest.raises(ValidationError):
        RuntimeSpec(engine="tensorrt", dtype="fp16", builder_optimization_level=-1)


def test_recipe_with_opt_level_5_parses():
    r = Recipe.model_validate(_minimal_recipe(builder_optimization_level=5))
    assert r.runtime.builder_optimization_level == 5


def test_recipe_without_opt_level_still_valid():
    r = Recipe.model_validate(_minimal_recipe())
    assert r.runtime.builder_optimization_level is None


def test_result_build_time_s_defaults_to_none():
    r = Result(
        recipe="t",
        started_at="2026-04-23T00:00:00+00:00",
        finished_at="2026-04-23T00:01:00+00:00",
        env={},
        latency_ms=LatencyStats(),
        accuracy=AccuracyStats(),
    )
    assert r.build_time_s is None


def test_result_build_time_s_accepts_float():
    r = Result(
        recipe="t",
        started_at="2026-04-23T00:00:00+00:00",
        finished_at="2026-04-23T00:01:00+00:00",
        env={},
        latency_ms=LatencyStats(),
        accuracy=AccuracyStats(),
        build_time_s=123.45,
    )
    assert r.build_time_s == 123.45


def test_legacy_result_json_without_build_time_s_parses():
    """Historical JSONs (pre-Wave 14) have no build_time_s field. Must parse clean."""
    legacy = {
        "recipe": "legacy_recipe",
        "started_at": "2025-01-01T00:00:00+00:00",
        "finished_at": "2025-01-01T00:00:10+00:00",
        "env": {},
        "latency_ms": {"p50": 5.0},
        "accuracy": {},
    }
    r = Result.model_validate(legacy)
    assert r.recipe == "legacy_recipe"
    assert r.build_time_s is None


def test_bf16_dtype_valid_on_tensorrt_runtime():
    """Wave 14 A2: dtype=bf16 is a valid RuntimeSpec value (shared with Wave 6 CPU)."""
    r = Recipe.model_validate(_minimal_recipe(dtype="bf16"))
    assert r.runtime.dtype == "bf16"
