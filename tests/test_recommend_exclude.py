"""recommend.py --exclude regression tests (Wave 11 Task 6).

Guards the parked-recipe hide mechanism. Used by Makefile's ``PARKED`` variable
to keep a Result JSON on disk for history while removing it from the ranking
table. Matching is by ``Result.recipe`` (the recipe's ``name:`` field), not by
filename.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts._schemas import AccuracyStats, EnvInfo, LatencyStats, Result, ThroughputStats
from scripts.recommend import load_results


def _write_result(dir_: Path, filename: str, recipe_name: str) -> None:
    r = Result(
        recipe=recipe_name,
        started_at="2026-04-23T00:00:00+00:00",
        finished_at="2026-04-23T00:01:00+00:00",
        env=EnvInfo(gpu="test"),
        latency_ms=LatencyStats(p50=5.0),
        throughput_fps=ThroughputStats(bs1=200.0),
        accuracy=AccuracyStats(map_50=0.95),
    )
    (dir_ / filename).write_text(r.model_dump_json(), encoding="utf-8")


def test_exclude_drops_single_recipe(tmp_path: Path):
    _write_result(tmp_path, "a.json", "active_recipe")
    _write_result(tmp_path, "b.json", "parked_recipe")

    results = load_results(tmp_path, exclude={"parked_recipe"})
    names = [r.recipe for r in results]
    assert names == ["active_recipe"]


def test_exclude_multiple_recipes(tmp_path: Path):
    _write_result(tmp_path, "a.json", "alpha")
    _write_result(tmp_path, "b.json", "beta")
    _write_result(tmp_path, "c.json", "gamma")

    results = load_results(tmp_path, exclude={"alpha", "gamma"})
    names = [r.recipe for r in results]
    assert names == ["beta"]


def test_exclude_empty_set_keeps_all(tmp_path: Path):
    _write_result(tmp_path, "a.json", "alpha")
    _write_result(tmp_path, "b.json", "beta")

    results = load_results(tmp_path, exclude=set())
    names = sorted(r.recipe for r in results)
    assert names == ["alpha", "beta"]


def test_exclude_matches_recipe_name_not_filename(tmp_path: Path):
    """filename is '00_trt_fp32.json' but Result.recipe='trt_fp32'.
    Excluding by filename stem should NOT match — we filter on recipe.name.
    """
    _write_result(tmp_path, "00_trt_fp32.json", "trt_fp32")

    # Exclude by filename stem: no effect.
    results = load_results(tmp_path, exclude={"00_trt_fp32"})
    assert [r.recipe for r in results] == ["trt_fp32"]

    # Exclude by recipe name: match.
    results = load_results(tmp_path, exclude={"trt_fp32"})
    assert results == []


def test_exclude_unknown_recipe_is_noop(tmp_path: Path):
    """Excluding a non-existent name doesn't raise or remove anything."""
    _write_result(tmp_path, "a.json", "alpha")

    results = load_results(tmp_path, exclude={"does_not_exist"})
    assert [r.recipe for r in results] == ["alpha"]
