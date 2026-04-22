"""Tests for Wave 6 Task 6 — measurement hygiene extensions.

Two additions on top of existing measure.py:
1. `LatencyStats.stddev_ms` — wall-clock stddev published alongside the
   p50/p95/p99 percentiles. Optional, defaults None so historical result
   JSONs stay valid and GPU paths get it for free on re-run.
2. `MeasurementSpec.iter_cooldown_ms` + `measure_latency(iter_cooldown_ms=...)`
   — per-iteration sleep between forward passes. Off by default; CPU
   recipes may opt in to absorb thermal throttle between iters.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_latency_stats_accepts_stddev_ms_field():
    """Schema: stddev_ms must be an Optional[float] field on LatencyStats."""
    from scripts._schemas import LatencyStats

    stats = LatencyStats(p50=10.0, p95=12.0, p99=15.0, stddev_ms=1.2)
    assert stats.stddev_ms == 1.2
    # Default None — historical JSONs don't carry the field
    stats2 = LatencyStats(p50=10.0, p95=12.0, p99=15.0)
    assert stats2.stddev_ms is None


def test_percentiles_helper_returns_stddev_ms():
    """measure.percentiles() must include stddev_ms so every runner
    (GPU + CPU) picks it up through its usual LatencyStats filter."""
    from scripts.measure import percentiles

    samples = [10.0, 11.0, 10.5, 9.5, 10.2]
    result = percentiles(samples)
    assert "stddev_ms" in result
    # population stddev for this sample is ~0.51
    assert 0.3 < result["stddev_ms"] < 0.8


def test_percentiles_stddev_zero_for_constant_samples():
    from scripts.measure import percentiles

    result = percentiles([5.0, 5.0, 5.0, 5.0])
    assert result["stddev_ms"] == 0.0


def test_measure_latency_emits_stddev_ms_in_output():
    """measure_latency() wraps percentiles() so stddev_ms propagates."""
    from scripts.measure import measure_latency

    tick = [0.0]

    def fake_forward():
        # Not strictly needed — perf_counter advances across iterations,
        # stddev is whatever the loop overhead happens to be.
        tick[0] += 1

    stats = measure_latency(fake_forward, warmup_iters=2, measure_iters=5)
    assert "stddev_ms" in stats
    assert stats["stddev_ms"] is not None
    assert stats["stddev_ms"] >= 0.0


def test_measurement_spec_has_iter_cooldown_ms_field():
    """MeasurementSpec exposes iter_cooldown_ms (None = no cooldown).
    Recipe YAML can set a non-null float for thermal relief between iters."""
    from scripts._schemas import MeasurementSpec

    spec = MeasurementSpec(
        dataset="coco_val2017", num_images=10,
        warmup_iters=2, measure_iters=2, batch_sizes=[1],
        iter_cooldown_ms=10.0,
    )
    assert spec.iter_cooldown_ms == 10.0
    # Default stays None (GPU recipes unaffected)
    spec2 = MeasurementSpec(
        dataset="coco_val2017", num_images=10,
        warmup_iters=2, measure_iters=2, batch_sizes=[1],
    )
    assert spec2.iter_cooldown_ms is None


def test_measure_latency_respects_iter_cooldown_ms(monkeypatch):
    """When iter_cooldown_ms is provided, measure_latency sleeps that
    many ms between iterations. We monkey-patch time.sleep to count calls
    rather than actually wait."""
    from scripts import measure

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(measure.time, "sleep", fake_sleep)

    def fake_forward():
        pass

    measure.measure_latency(
        fake_forward, warmup_iters=1, measure_iters=4, iter_cooldown_ms=5.0,
    )
    # One sleep per measure iter (warmup is untimed, no cooldown needed)
    assert len(sleeps) == 4
    assert all(abs(s - 0.005) < 1e-9 for s in sleeps), \
        f"expected 4×0.005s sleeps, got {sleeps!r}"


def test_measure_latency_no_cooldown_calls_no_sleep(monkeypatch):
    """iter_cooldown_ms=None (default) means no time.sleep() is called
    inside the measurement loop — preserves GPU hot-path timing."""
    from scripts import measure

    sleeps: list[float] = []
    monkeypatch.setattr(measure.time, "sleep", lambda s: sleeps.append(s))

    def fake_forward():
        pass

    measure.measure_latency(fake_forward, warmup_iters=1, measure_iters=4)
    assert sleeps == [], \
        f"unexpected sleep calls without iter_cooldown_ms: {sleeps!r}"
