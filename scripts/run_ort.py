"""Runner for ONNX Runtime recipes (#3 CUDA EP FP16, #4 TensorRT EP FP16).

Strategy: export the ultralytics model to ONNX once (cached under
``results/_onnx/<name>.onnx``), then create an ``InferenceSession`` with the
requested execution provider and measure latency on a synthetic input.

Accuracy in v1 is delegated back to ultralytics' ``model.val(..., format='onnx')``
path so numbers remain directly comparable to the PyTorch runner.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


def _add_tensorrt_dll_dir() -> None:
    """Register TRT wheel's DLL directory before ORT imports the TRT provider.

    Wave 11 Task 0 finding (2026-04-22): pip-installed tensorrt 10.x drops
    ``nvinfer_10.dll`` under ``site-packages/tensorrt_libs/`` but does not
    add that directory to the Windows DLL search path. ``ort.get_available_
    providers()`` still lists ``TensorrtExecutionProvider`` (stub check), so
    the failure is silent — session creation falls back to CPU EP and fps
    measurements become meaningless. See
    ``docs/improvements/2026-04-22-wave11-task0-findings.md``.
    """
    try:
        import tensorrt  # noqa: F401 — locate the wheel
    except Exception:
        return
    site_parent = Path(tensorrt.__file__).resolve().parent.parent
    libs_dir = site_parent / "tensorrt_libs"
    if libs_dir.is_dir() and hasattr(os, "add_dll_directory"):
        # Handle-scoped: does not mutate PATH or interfere with other wheels'
        # CUDA DLL search (torch/lib/cublas64_12.dll).
        os.add_dll_directory(str(libs_dir))


_add_tensorrt_dll_dir()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._schemas import (  # noqa: E402
    AccuracyStats,
    LatencyStats,
    Recipe,
    Result,
    ThroughputStats,
    load_recipe,
)
from scripts.env_lock import collect_env, lock_gpu_clock  # noqa: E402
from scripts.measure import (  # noqa: E402
    measure_cold_start,
    measure_latency,
    throughput_from_latency,
)
from scripts import _split  # noqa: E402


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _export_onnx(weights: str, imgsz: int, half: bool, cache_dir: Path,
                 dynamic: bool = True) -> Path:
    """Export ultralytics weights to ONNX. Dynamic batch by default so ORT
    sessions can run both bs=1 and bs>1 without a second export."""
    from ultralytics import YOLO

    cache_dir.mkdir(parents=True, exist_ok=True)
    bs_tag = "dyn" if dynamic else "bs1"
    tag = f"{Path(weights).stem}_{imgsz}_{'fp16' if half else 'fp32'}_{bs_tag}.onnx"
    cached = cache_dir / tag
    if cached.exists():
        return cached

    model = YOLO(weights)
    onnx_path = model.export(
        format="onnx", imgsz=imgsz, half=half, simplify=True, dynamic=dynamic,
    )
    src = Path(onnx_path)
    if src != cached:
        src.rename(cached)
    return cached


def _make_session(onnx_path: Path, execution_provider: str, dtype: str = "fp32"):
    import onnxruntime as ort

    available = ort.get_available_providers()
    if execution_provider not in available:
        raise RuntimeError(
            f"{execution_provider} not available. onnxruntime providers: {available}"
        )

    # Wave 11 Task 3 (B3) — TRT EP requires explicit cache + fp16 opt-in.
    # get_available_providers() listing alone is unreliable (Task 0 finding).
    #
    # Wave 15 D1.2 — match Wave 14 native TRT defaults on the ORT EP side.
    # builder_optimization_level=5 runs exhaustive tactic autotune (Wave 14
    # #40 measured +48% fps on FP16 with this single knob). timing cache
    # amortizes tactic timings across recipe rebuilds, cutting cold_start_ms
    # on subsequent runs. Older ORT that doesn't recognize these keys would
    # raise on session init; the fallback below strips them and retries so
    # the runner stays compatible.
    if execution_provider == "TensorrtExecutionProvider":
        trt_cache = ROOT / "results" / "_trt_cache"
        trt_cache.mkdir(parents=True, exist_ok=True)
        trt_opts = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(trt_cache),
            "trt_fp16_enable": dtype == "fp16",
            "trt_builder_optimization_level": 5,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(trt_cache),
            "trt_detailed_build_log": True,
        }
        # Fall-through chain: TRT → CUDA → CPU so ops TRT cannot compile still run on GPU.
        providers: list = [(execution_provider, trt_opts), "CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        trt_opts = None
        providers = [execution_provider, "CPUExecutionProvider"]

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
    except (ValueError, RuntimeError) as e:
        # Wave 15 D1.2 backward-compat: strip new TRT EP keys if this ORT
        # build rejects them and retry once. Only applies when we actually
        # sent the Wave 15 additions; other failures propagate.
        # Narrowed from bare Exception so OSError / MemoryError / auth errors
        # surface rather than being misrouted into the retry path.
        wave15_keys = (
            "trt_builder_optimization_level",
            "trt_timing_cache_enable",
            "trt_timing_cache_path",
            "trt_detailed_build_log",
        )
        if trt_opts is not None and any(k in str(e) for k in wave15_keys):
            for k in wave15_keys:
                trt_opts.pop(k, None)
            providers = [(execution_provider, trt_opts), "CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"[warn] ORT rejected Wave 15 TRT EP keys ({e}); retrying with legacy options",
                  file=sys.stderr)
            session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
        else:
            raise
    # Guard: make sure the requested EP actually bound. Silent CPU fallback masks
    # regressions (Wave 11 Task 0 finding).
    active = session.get_providers()
    if active[0] != execution_provider:
        raise RuntimeError(
            f"{execution_provider} requested but primary is {active[0]!r}. "
            f"Active chain: {active}. Check TRT / CUDA DLL deps."
        )
    return session


def _make_forward(session, input_name: str, input_shape, dtype_np):
    import numpy as np

    x = np.random.randn(*input_shape).astype(dtype_np)

    def fwd():
        return session.run(None, {input_name: x})

    return fwd


def run(recipe_path: str, out_path: str) -> int:
    import numpy as np

    recipe: Recipe = load_recipe(recipe_path)
    _seed_all(recipe.measurement.seed)

    env = collect_env()
    clock_note = lock_gpu_clock(recipe.measurement.gpu_clock_lock)
    if clock_note:
        env["clock_lock_note"] = clock_note

    ep = recipe.runtime.execution_provider
    if not ep:
        raise ValueError("runtime.execution_provider is required for ORT recipes")

    half = recipe.runtime.dtype == "fp16"
    dtype_np = np.float16 if half else np.float32
    imgsz = recipe.measurement.input_size

    cache_dir = Path("results/_onnx")
    # Keep one ONNX reference around for the mAP eval path; per-bs loop
    # rebuilds dedicated sessions from static/dynamic ONNX to keep CUDA EP
    # fast at bs=1 while still allowing bs>1.
    onnx_path_default = _export_onnx(recipe.model.weights, imgsz, half,
                                     cache_dir, dynamic=False)

    started = datetime.now(timezone.utc).isoformat()

    ort_type_to_np = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
    }

    per_bs: dict[int, dict] = {}
    cold_start_ms = None
    for bs in recipe.measurement.batch_sizes:
        onnx_for_bs = _export_onnx(recipe.model.weights, imgsz, half,
                                   cache_dir, dynamic=(bs > 1))
        try:
            def _load(p=onnx_for_bs):
                return _make_session(p, ep, dtype=recipe.runtime.dtype)

            session, this_cold_ms = measure_cold_start(_load)
            if cold_start_ms is None:
                cold_start_ms = this_cold_ms

            input_meta = session.get_inputs()[0]
            input_name = input_meta.name
            actual_dtype_np = ort_type_to_np.get(input_meta.type, dtype_np)
            shape_with_bs = (bs, 3, imgsz, imgsz)
            fwd = _make_forward(session, input_name, shape_with_bs, actual_dtype_np)
            stats = measure_latency(
                fwd,
                warmup_iters=recipe.measurement.warmup_iters,
                measure_iters=recipe.measurement.measure_iters,
            )
            per_bs[bs] = stats
        except Exception as e:
            print(f"[warn] bs={bs} skipped: {e}", file=sys.stderr)

    if not per_bs:
        raise RuntimeError("no batch size measurements succeeded")

    lat = per_bs.get(1) or next(iter(per_bs.values()))
    throughput = ThroughputStats(
        bs1=throughput_from_latency(per_bs[1]["p50"], 1) if 1 in per_bs else None,
        bs8=throughput_from_latency(per_bs[8]["p50"], 8) if 8 in per_bs else None,
    )
    peak_mem = max((v.get("peak_gpu_mem_mb") or 0.0) for v in per_bs.values()) or None

    # Accuracy via ultralytics' ONNX val path
    acc = AccuracyStats()
    if os.environ.get("OMNI_SKIP_ACCURACY"):
        print("[info] OMNI_SKIP_ACCURACY set — skipping mAP eval", file=sys.stderr)
    else:
        try:
            from ultralytics import YOLO

            m = YOLO(str(onnx_path_default))
            data_yaml = _split.eval_yaml(
                os.environ.get("OMNI_COCO_YAML", "coco.yaml"),
                calib_yaml_path=_split.calib_yaml(),
                calib_seed=recipe.technique.calibration_seed or 42,
                calib_n=recipe.technique.calibration_samples or 512,
            )
            metrics = m.val(data=data_yaml, imgsz=imgsz, batch=1, half=half,
                            device=0, plots=False, verbose=False)
            acc = AccuracyStats(
                map_50_95=float(metrics.box.map),
                map_50=float(metrics.box.map50),
            )
        except Exception as e:
            print(f"[warn] accuracy eval failed for {ep}: {e}", file=sys.stderr)

    try:
        model_size_mb = onnx_path_default.stat().st_size / (1024 ** 2)
    except Exception:
        model_size_mb = None

    finished = datetime.now(timezone.utc).isoformat()

    meets = None
    c = recipe.constraints
    if c.min_fps_bs1 is not None and throughput.bs1 is not None:
        meets = throughput.bs1 >= c.min_fps_bs1

    result = Result(
        recipe=recipe.name,
        started_at=started,
        finished_at=finished,
        env=env,  # type: ignore[arg-type]
        model_size_mb=model_size_mb,
        latency_ms=LatencyStats(**{k: v for k, v in lat.items() if k in {"p50", "p95", "p99", "stddev_ms"}}),
        throughput_fps=throughput,
        peak_gpu_mem_mb=peak_mem,
        cold_start_ms=cold_start_ms,
        accuracy=acc,
        meets_constraints=meets,
        notes=f"execution_provider={ep}, onnx={onnx_path_default.name}",
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json.loads(result.model_dump_json()), f, indent=2)
    print(f"wrote {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    return run(args.recipe, args.out)


if __name__ == "__main__":
    sys.exit(main())
