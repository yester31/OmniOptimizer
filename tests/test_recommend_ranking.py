"""recommend.py ranking + constraint filter regression tests (Wave 11 Task 6).

Guards:
- Wave 10 reopen bug: passing ``--max-map-drop-pct 5.0`` was required for
  FastNAS recipes; default 1.0 marked them ✘. No test caught the CLI
  propagation path. These tests cover rank() directly.
- Schema backward compat: historical Result JSONs had many Optional fields
  unset. Parsing must not break.
- Ordering: fps descending, meets-constraints first.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts._schemas import AccuracyStats, EnvInfo, LatencyStats, Result, ThroughputStats
from scripts.recommend import load_results, rank


def _make_result(
    name: str,
    fps_bs1: float,
    map_50: float | None = None,
    map_50_95: float | None = None,
    notes: str | None = None,
) -> Result:
    return Result(
        recipe=name,
        started_at="2026-04-23T00:00:00+00:00",
        finished_at="2026-04-23T00:01:00+00:00",
        env=EnvInfo(gpu="test"),
        latency_ms=LatencyStats(p50=1.0 / fps_bs1 * 1000.0, p95=1.1, p99=1.2)
        if fps_bs1 > 0
        else LatencyStats(),
        throughput_fps=ThroughputStats(bs1=fps_bs1 if fps_bs1 > 0 else None),
        accuracy=AccuracyStats(map_50=map_50, map_50_95=map_50_95),
        notes=notes,
    )


def test_max_map_drop_pct_threshold_gates_recipe():
    """baseline mAP 0.988, candidate mAP 0.950 (drop 3.8%p).
    - max_map_drop_pct=1.5 → ✘ (3.8 > 1.5)
    - max_map_drop_pct=5.0 → ✔ (3.8 <= 5.0)
    """
    baseline = _make_result("pytorch_fp32", fps_bs1=50.0, map_50=0.988)
    candidate = _make_result("fastnas_int8", fps_bs1=700.0, map_50=0.950)

    # Strict threshold: candidate drops out.
    rows, _ = rank([baseline, candidate], max_map_drop_pct=1.5)
    cand_row = next(r for r in rows if r["recipe"] == "fastnas_int8")
    assert not cand_row["meets"]
    assert any("mAP drop" in reason for reason in cand_row["reasons"])

    # Relaxed threshold: candidate passes.
    rows, _ = rank([baseline, candidate], max_map_drop_pct=5.0)
    cand_row = next(r for r in rows if r["recipe"] == "fastnas_int8")
    assert cand_row["meets"], f"reasons: {cand_row['reasons']}"


def test_min_fps_bs1_threshold():
    """min_fps_bs1=100 filters out a 50-fps recipe but keeps a 700-fps one."""
    slow = _make_result("pytorch_fp32", fps_bs1=50.0, map_50=0.988)
    fast = _make_result("trt_int8", fps_bs1=700.0, map_50=0.985)

    rows, _ = rank([slow, fast], max_map_drop_pct=1.5, min_fps_bs1=100.0)
    slow_row = next(r for r in rows if r["recipe"] == "pytorch_fp32")
    fast_row = next(r for r in rows if r["recipe"] == "trt_int8")
    assert not slow_row["meets"]
    assert any("fps 50" in r for r in slow_row["reasons"])
    assert fast_row["meets"]


def test_ranking_order_meets_first_then_fps_desc():
    """Meeting-constraint recipes rank first; within each group, higher fps wins."""
    base = _make_result("pytorch_fp32", fps_bs1=50.0, map_50=0.988)
    broken = _make_result("broken_recipe", fps_bs1=1.0, map_50=0.0)  # mAP=0
    mid = _make_result("ort_fp16", fps_bs1=200.0, map_50=0.985)
    top = _make_result("trt_int8", fps_bs1=800.0, map_50=0.987)

    rows, _ = rank([base, broken, mid, top], max_map_drop_pct=2.0)
    order = [r["recipe"] for r in rows]
    # Meets cluster (by fps desc): trt_int8 > ort_fp16 > pytorch_fp32.
    # Failing cluster: broken_recipe last.
    assert order[0] == "trt_int8"
    assert order[1] == "ort_fp16"
    assert order[2] == "pytorch_fp32"
    assert order[3] == "broken_recipe"


def test_missing_measurements_fail_gracefully():
    """Result with fps_bs1=None (session build failed) ranks last with 'missing measurements'."""
    base = _make_result("pytorch_fp32", fps_bs1=50.0, map_50=0.988)
    failed = _make_result("trt_broken", fps_bs1=0.0, map_50=None, notes="engine build failed")

    rows, _ = rank([base, failed])
    failed_row = next(r for r in rows if r["recipe"] == "trt_broken")
    assert not failed_row["meets"]
    assert any("missing" in r for r in failed_row["reasons"])


def test_all_optional_fields_none_parses_cleanly(tmp_path: Path):
    """A minimal historical-style Result JSON (all Optionals None) must parse.
    Covers the backward-compat surface we promised when adding Wave 6 CPU fields
    and Wave 10 FastNAS fields.
    """
    minimal = {
        "recipe": "legacy_recipe",
        "started_at": "2025-01-01T00:00:00+00:00",
        "finished_at": "2025-01-01T00:00:10+00:00",
        "env": {},
        "latency_ms": {},
        "throughput_fps": {},
        "accuracy": {},
    }
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(minimal), encoding="utf-8")

    results = load_results(tmp_path)
    assert len(results) == 1
    assert results[0].recipe == "legacy_recipe"
    # rank() must not raise.
    rows, _ = rank(results)
    assert rows[0]["recipe"] == "legacy_recipe"


def test_load_results_skips_underscored_meta_files(tmp_path: Path):
    """_env.json and other underscore-prefixed files are meta, not results."""
    (tmp_path / "_env.json").write_text('{"gpu": "test"}', encoding="utf-8")
    good = _make_result("real_recipe", fps_bs1=100.0, map_50=0.9)
    (tmp_path / "real.json").write_text(good.model_dump_json(), encoding="utf-8")

    results = load_results(tmp_path)
    assert len(results) == 1
    assert results[0].recipe == "real_recipe"


def test_load_results_skips_malformed_json(tmp_path: Path):
    """Non-schema-conforming JSON produces [warn] but does not crash load_results."""
    (tmp_path / "broken.json").write_text('{"garbage": true}', encoding="utf-8")
    good = _make_result("real_recipe", fps_bs1=100.0, map_50=0.9)
    (tmp_path / "real.json").write_text(good.model_dump_json(), encoding="utf-8")

    results = load_results(tmp_path)
    assert [r.recipe for r in results] == ["real_recipe"]
