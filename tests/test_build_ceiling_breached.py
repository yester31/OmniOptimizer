"""Tests for Wave 16 D1 — build_ceiling_breached round-trip + recommend.py surfacing.

Covers:
- Schema: new field defaults to None (backward-compat for historical JSONs)
- Schema: accepts explicit True / False
- Schema: JSON round-trip preserves the value
- recommend.py: ## Build-Time Ceiling Breaches section appears when any recipe
  has the flag, hidden otherwise
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _minimal_result_payload(**overrides) -> dict:
    """Produce the smallest dict that validates as a Result (historical shape)."""
    payload = {
        "recipe": "test",
        "started_at": "2026-04-23T00:00:00+00:00",
        "finished_at": "2026-04-23T00:00:05+00:00",
        "env": {"os": "Windows", "python": "3.13"},
        "latency_ms": {},  # LatencyStats is a required BaseModel — empty dict validates
    }
    payload.update(overrides)
    return payload


def test_schema_defaults_to_none_for_historical_json():
    """Historical result JSONs have no build_ceiling_breached key — must not break."""
    from scripts._schemas import Result

    r = Result.model_validate(_minimal_result_payload())
    assert r.build_ceiling_breached is None


def test_schema_accepts_true():
    from scripts._schemas import Result

    r = Result.model_validate(
        _minimal_result_payload(build_ceiling_breached=True)
    )
    assert r.build_ceiling_breached is True


def test_schema_accepts_false():
    from scripts._schemas import Result

    r = Result.model_validate(
        _minimal_result_payload(build_ceiling_breached=False)
    )
    assert r.build_ceiling_breached is False


def test_schema_json_round_trip_preserves_breach_flag():
    """Dump → load → same value. Key invariant for MLPerf-style reproducibility."""
    from scripts._schemas import Result

    r = Result.model_validate(
        _minimal_result_payload(build_ceiling_breached=True, build_time_s=850.5)
    )
    dumped = json.loads(r.model_dump_json())
    assert dumped["build_ceiling_breached"] is True
    assert dumped["build_time_s"] == 850.5

    restored = Result.model_validate(dumped)
    assert restored.build_ceiling_breached is True
    assert restored.build_time_s == 850.5


def test_recommend_report_hides_breach_section_when_none():
    """No breach → no ## Build-Time Ceiling Breaches section."""
    from scripts.recommend import format_report

    rows = [
        {
            "recipe": "clean_recipe",
            "fps_bs1": 500.0, "fps_bs8": 800.0,
            "p50_ms": 2.0, "p50_gpu_ms": 1.9,
            "map_50": 0.95, "map_50_95": 0.72,
            "drop_pp": 0.0, "peak_mem_mb": 300,
            "meets": True, "reasons": [], "notes": None,
            "build_time_s": 120.0,
            "build_ceiling_breached": False,
        },
    ]
    report = format_report(rows, baseline=None)
    assert "Build-Time Ceiling Breaches" not in report


def test_recommend_report_shows_breach_section_when_present():
    """At least one breached recipe → section appears listing recipe name + build time."""
    from scripts.recommend import format_report

    rows = [
        {
            "recipe": "clean_recipe",
            "fps_bs1": 500.0, "fps_bs8": 800.0,
            "p50_ms": 2.0, "p50_gpu_ms": 1.9,
            "map_50": 0.95, "map_50_95": 0.72,
            "drop_pp": 0.0, "peak_mem_mb": 300,
            "meets": True, "reasons": [], "notes": None,
            "build_time_s": 120.0,
            "build_ceiling_breached": False,
        },
        {
            "recipe": "slow_recipe",
            "fps_bs1": 600.0, "fps_bs8": 900.0,
            "p50_ms": 1.7, "p50_gpu_ms": 1.6,
            "map_50": 0.94, "map_50_95": 0.71,
            "drop_pp": -0.5, "peak_mem_mb": 350,
            "meets": True, "reasons": [], "notes": None,
            "build_time_s": 850.0,
            "build_ceiling_breached": True,
        },
    ]
    report = format_report(rows, baseline=None)
    assert "## Build-Time Ceiling Breaches" in report
    assert "slow_recipe" in report
    assert "850" in report  # build_time_s is surfaced
    # clean_recipe must NOT appear inside the breaches section header line,
    # but it's still in the main ranking table — check order to prove scoping.
    breach_idx = report.index("## Build-Time Ceiling Breaches")
    clean_in_breach_block = "clean_recipe" in report[breach_idx : breach_idx + 500]
    # The breach block lists only breached recipes, so clean_recipe
    # shouldn't appear in the first 500 chars after the section header.
    assert not clean_in_breach_block, (
        "clean_recipe leaked into the ceiling-breach section"
    )


def test_advance_ceiling_tracker_none_build_time_keeps_prev():
    """Cached-engine path returns build_time_s=None — tracker must not crash.

    Regression for the Wave 16 D1 end-to-end validation bug: the merged code
    tried to compare None > ceiling, raising TypeError. No build happened ⇒
    no signal to record.
    """
    from scripts.run_trt import _advance_ceiling_tracker

    assert _advance_ceiling_tracker(None, None, 600) is None
    assert _advance_ceiling_tracker(True, None, 600) is True
    assert _advance_ceiling_tracker(False, None, 600) is False


def test_advance_ceiling_tracker_breach_sets_true():
    from scripts.run_trt import _advance_ceiling_tracker

    assert _advance_ceiling_tracker(None, 1200.0, 600) is True
    # Sticky: an earlier False cannot block a later True.
    assert _advance_ceiling_tracker(False, 1200.0, 600) is True
    # Sticky: earlier True stays True.
    assert _advance_ceiling_tracker(True, 1200.0, 600) is True


def test_advance_ceiling_tracker_under_ceiling_promotes_none_to_false():
    from scripts.run_trt import _advance_ceiling_tracker

    assert _advance_ceiling_tracker(None, 120.0, 600) is False


def test_advance_ceiling_tracker_under_ceiling_never_demotes_true():
    """Sticky-True invariant: a later under-ceiling build cannot mask an
    earlier breach across the bs loop."""
    from scripts.run_trt import _advance_ceiling_tracker

    # Scenario: bs=1 breached (True), bs=8 builds fast (< ceiling).
    after_bs1 = _advance_ceiling_tracker(None, 1200.0, 600)
    assert after_bs1 is True
    after_bs8 = _advance_ceiling_tracker(after_bs1, 50.0, 600)
    assert after_bs8 is True  # must NOT flip back to False


def test_recommend_report_tolerates_historical_rows_without_breach_key():
    """Rows produced from historical JSONs (via .get()) → no KeyError, no section."""
    from scripts.recommend import format_report

    rows = [
        {
            "recipe": "legacy_recipe",
            "fps_bs1": 500.0, "fps_bs8": 800.0,
            "p50_ms": 2.0, "p50_gpu_ms": 1.9,
            "map_50": 0.95, "map_50_95": 0.72,
            "drop_pp": 0.0, "peak_mem_mb": 300,
            "meets": True, "reasons": [], "notes": None,
            # No build_time_s, no build_ceiling_breached — pre-Wave-16 row shape.
        },
    ]
    report = format_report(rows, baseline=None)
    assert "Build-Time Ceiling Breaches" not in report
    assert "legacy_recipe" in report  # still ranked in the main table
