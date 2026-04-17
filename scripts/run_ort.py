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


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _export_onnx(weights: str, imgsz: int, half: bool, cache_dir: Path) -> Path:
    """Export ultralytics weights to ONNX if not already cached."""
    from ultralytics import YOLO

    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(weights).stem}_{imgsz}_{'fp16' if half else 'fp32'}.onnx"
    cached = cache_dir / tag
    if cached.exists():
        return cached

    model = YOLO(weights)
    onnx_path = model.export(format="onnx", imgsz=imgsz, half=half, simplify=True, dynamic=False)
    src = Path(onnx_path)
    if src != cached:
        src.rename(cached)
    return cached


def _make_session(onnx_path: Path, execution_provider: str):
    import onnxruntime as ort

    available = ort.get_available_providers()
    if execution_provider not in available:
        raise RuntimeError(
            f"{execution_provider} not available. onnxruntime providers: {available}"
        )
    # Provider-specific tuning can be added here. Keep defaults for v1.
    providers = [execution_provider, "CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)


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
    onnx_path = _export_onnx(recipe.model.weights, imgsz, half, cache_dir)

    started = datetime.now(timezone.utc).isoformat()

    def _load():
        return _make_session(onnx_path, ep)

    session, cold_start_ms = measure_cold_start(_load)

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    # Use actual input dtype from the session (ultralytics sometimes exports FP32
    # inputs even when half=True was requested).
    ort_type_to_np = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
    }
    actual_dtype_np = ort_type_to_np.get(input_meta.type, dtype_np)
    if actual_dtype_np != dtype_np:
        print(
            f"[info] session input dtype is {input_meta.type}, using {actual_dtype_np} for feed",
            file=sys.stderr,
        )

    per_bs: dict[int, dict] = {}
    for bs in recipe.measurement.batch_sizes:
        # If the exported model has a fixed batch dim, only bs=1 is valid.
        shape_with_bs = (bs, 3, imgsz, imgsz)
        try:
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

            m = YOLO(str(onnx_path))
            data_yaml = os.environ.get("OMNI_COCO_YAML", "coco.yaml")
            metrics = m.val(data=data_yaml, imgsz=imgsz, batch=1, half=half, plots=False, verbose=False)
            acc = AccuracyStats(
                map_50_95=float(metrics.box.map),
                map_50=float(metrics.box.map50),
            )
        except Exception as e:
            print(f"[warn] accuracy eval failed for {ep}: {e}", file=sys.stderr)

    try:
        model_size_mb = onnx_path.stat().st_size / (1024 ** 2)
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
        latency_ms=LatencyStats(**{k: v for k, v in lat.items() if k in {"p50", "p95", "p99"}}),
        throughput_fps=throughput,
        peak_gpu_mem_mb=peak_mem,
        cold_start_ms=cold_start_ms,
        accuracy=acc,
        meets_constraints=meets,
        notes=f"execution_provider={ep}, onnx={onnx_path.name}",
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
