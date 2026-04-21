"""Latency / throughput / memory measurement utilities.

The core pattern: run ``forward_fn`` ``warmup_iters`` times to absorb JIT /
kernel-cache warm-up, then ``measure_iters`` times to record per-iteration
wall-clock latency, then compute percentiles.

Wall-clock (``time.perf_counter`` around ``_cuda_sync``) is the primary metric
— it captures the end-to-end user-visible latency including Python + TRT
launch overhead. CUDA events (``p50_gpu`` etc.) are reported as a secondary
metric isolating on-GPU execution time, useful when debugging launch-overhead
dominated nano models.

Peak GPU memory is likewise reported twice: ``peak_gpu_mem_mb`` uses torch's
caching allocator (misses TRT's own ``cudaMalloc`` calls), and
``peak_gpu_mem_mb_nvml_delta`` uses NVML process-memory delta (baseline →
peak), which captures TRT allocator bytes. Both degrade to ``None`` when
their backing API is unavailable.
"""
from __future__ import annotations

import gc
import time
from typing import Callable, Optional, TypeVar

_T = TypeVar("_T")

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
    """Torch caching-allocator peak since last reset. None if torch/CUDA absent.

    This is the "backbone" peak: correct for PyTorch/ORT-CUDA runners, an
    underestimate for TRT runners (TRT uses its own allocator). Pair with
    ``_read_nvml_used_bytes`` for a TRT-inclusive second source.
    """
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


class _NvmlDeltaProbe:
    """Track NVML device memory baseline → peak during a measurement window.

    Graceful: if pynvml is missing or nvmlInit fails, ``delta_mb()`` returns
    ``None`` and all ``sample()`` calls are no-ops. This lets measure_latency
    always call it unconditionally.
    """

    def __init__(self, device_index: int = 0):
        self._ok = False
        self._handle = None
        self._baseline: Optional[int] = None
        self._peak: Optional[int] = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._baseline = int(info.used)
            self._peak = int(info.used)
            self._ok = True
        except Exception:
            self._ok = False

    def sample(self) -> None:
        if not self._ok:
            return
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            used = int(info.used)
            if self._peak is None or used > self._peak:
                self._peak = used
        except Exception:
            # Don't break the measurement loop on a flaky NVML call.
            pass

    def delta_mb(self) -> Optional[float]:
        if not self._ok or self._baseline is None or self._peak is None:
            return None
        return max(0.0, (self._peak - self._baseline) / (1024 ** 2))

    def close(self) -> None:
        if not self._ok:
            return
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass


def percentiles(samples_ms: list[float]) -> dict:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        # Wave 6 Task 6: population stddev over the measurement window.
        # Useful for CPU recipes where thermal / scheduler jitter is the
        # dominant noise source; on GPU it mostly reflects launch-overhead
        # variance and is much smaller.
        "stddev_ms": float(np.std(arr)),
    }


def _gpu_percentiles(samples_ms: list[float]) -> dict:
    """Same percentile bins as ``percentiles`` but keyed for GPU-time fields."""
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50_gpu": float(np.percentile(arr, 50)),
        "p95_gpu": float(np.percentile(arr, 95)),
        "p99_gpu": float(np.percentile(arr, 99)),
    }


def _maybe_make_cuda_events() -> Optional[tuple]:
    """Return a (start, end) pair of torch.cuda.Event if CUDA is usable, else None.

    ``enable_timing=True`` is required to query ``elapsed_time``.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        return (start, end)
    except Exception:
        return None


def measure_latency(
    forward_fn: Callable[[], object],
    warmup_iters: int,
    measure_iters: int,
    iter_cooldown_ms: Optional[float] = None,
) -> dict:
    """Run ``forward_fn`` in a tight loop and return latency percentiles (ms).

    Returned dict (keys; ``None`` entries are kept so callers can pass the
    whole dict through to ``LatencyStats``/``Result`` without filtering):

    - ``p50`` / ``p95`` / ``p99``: wall-clock percentiles in ms (primary)
    - ``stddev_ms``: population stddev of wall-clock samples (Wave 6 Task 6)
    - ``p50_gpu`` / ``p95_gpu`` / ``p99_gpu``: CUDA-event percentiles in ms,
      or ``None`` when CUDA/torch is not available
    - ``peak_gpu_mem_mb``: torch caching-allocator peak (``None`` if torch
      absent; NVML used-bytes fallback if NVML alone is present)
    - ``peak_gpu_mem_mb_nvml_delta``: NVML process-memory delta during the
      measurement window (``None`` if pynvml is unavailable)

    ``iter_cooldown_ms`` (Wave 6 Task 6): when set, sleep this many ms
    *after* each measured iteration. Off by default. CPU recipes may opt
    in via ``MeasurementSpec.iter_cooldown_ms`` to blunt thermal throttle
    between iters; GPU recipes should leave it off so wall-clock timing
    isn't polluted by the sleep scheduling granularity.

    Runners that only forward the subset of keys they know about (legacy
    behaviour) still work — the new Optional fields simply stay ``None`` in
    the resulting ``LatencyStats`` / ``Result``.
    """
    _reset_peak_mem()

    # Warm-up (no timing, but let the GPU / caches settle)
    for _ in range(warmup_iters):
        forward_fn()
    _cuda_sync()

    # Baseline NVML snapshot taken *after* warm-up so steady-state buffers
    # are already resident — the delta then reflects measurement-window
    # allocations (e.g. new TRT IO tensors), not one-time runtime init.
    nvml_probe = _NvmlDeltaProbe()

    # Set up CUDA events once; if CUDA is unavailable we simply won't record
    # per-iteration GPU time and p*_gpu stays None.
    events = _maybe_make_cuda_events()

    # Measure
    wall_samples: list[float] = []
    gpu_samples: list[float] = []
    for _ in range(measure_iters):
        _cuda_sync()
        if events is not None:
            start_ev, end_ev = events
            t0 = time.perf_counter()
            start_ev.record()
            forward_fn()
            end_ev.record()
            # synchronize the *event* (not the whole device) for accurate
            # GPU time, then ensure CPU-side timing also reflects completion.
            end_ev.synchronize()
            gpu_samples.append(float(start_ev.elapsed_time(end_ev)))
            _cuda_sync()
            wall_samples.append((time.perf_counter() - t0) * 1000.0)
        else:
            t0 = time.perf_counter()
            forward_fn()
            _cuda_sync()
            wall_samples.append((time.perf_counter() - t0) * 1000.0)
        nvml_probe.sample()
        if iter_cooldown_ms:
            # Outside the timing window — intentional: we want to charge
            # this sleep against real time, not record it as latency.
            time.sleep(iter_cooldown_ms / 1000.0)

    stats: dict = percentiles(wall_samples)
    # Always include GPU keys so downstream consumers can rely on the shape;
    # value is None when CUDA events weren't available.
    if gpu_samples:
        stats.update(_gpu_percentiles(gpu_samples))
    else:
        stats.update({"p50_gpu": None, "p95_gpu": None, "p99_gpu": None})

    stats["peak_gpu_mem_mb"] = _read_peak_mem_mb()
    stats["peak_gpu_mem_mb_nvml_delta"] = nvml_probe.delta_mb()
    nvml_probe.close()
    return stats


def measure_cold_start(load_fn: Callable[[], _T]) -> tuple[_T, float]:
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
