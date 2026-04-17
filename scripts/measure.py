"""Latency / throughput / memory measurement utilities.

The core pattern: run ``forward_fn`` ``warmup_iters`` times to absorb JIT /
kernel-cache warm-up, then ``measure_iters`` times to record per-iteration
wall-clock latency, then compute percentiles.
"""
from __future__ import annotations

import gc
import time
from typing import Callable, Optional

import numpy as np


def _cuda_sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _reset_peak_mem() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _read_peak_mem_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    # Fallback via pynvml used memory (less precise)
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        pynvml.nvmlShutdown()
        return info.used / (1024 ** 2)
    except Exception:
        return None


def percentiles(samples_ms: list[float]) -> dict:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def measure_latency(
    forward_fn: Callable[[], object],
    warmup_iters: int,
    measure_iters: int,
) -> dict:
    """Run ``forward_fn`` in a tight loop and return latency percentiles (ms)."""
    _reset_peak_mem()

    # Warm-up
    for _ in range(warmup_iters):
        forward_fn()
    _cuda_sync()

    # Measure
    samples: list[float] = []
    for _ in range(measure_iters):
        _cuda_sync()
        t0 = time.perf_counter()
        forward_fn()
        _cuda_sync()
        samples.append((time.perf_counter() - t0) * 1000.0)

    stats = percentiles(samples)
    stats["peak_gpu_mem_mb"] = _read_peak_mem_mb()
    return stats


def measure_cold_start(load_fn: Callable[[], object]) -> tuple[object, float]:
    """Time a cold-start: delete caches, call ``load_fn``, measure wall-clock.

    Returns (loaded_object, cold_start_ms).
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    t0 = time.perf_counter()
    obj = load_fn()
    _cuda_sync()
    cold_ms = (time.perf_counter() - t0) * 1000.0
    return obj, cold_ms


def throughput_from_latency(latency_ms_p50: float, batch_size: int) -> float:
    """Convert a steady-state p50 latency to throughput (fps)."""
    if latency_ms_p50 <= 0:
        return 0.0
    return (1000.0 / latency_ms_p50) * batch_size
