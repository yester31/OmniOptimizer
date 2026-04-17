"""Runner for PyTorch recipes (#1 eager FP32, #2 torch.compile FP16).

Loads a YOLO26n model via ultralytics, wraps its forward in a callable suitable
for ``measure_latency``, and writes a result JSON that matches ``Result``
schema.
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
from scripts.eval_coco import evaluate_via_ultralytics  # noqa: E402


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _make_forward(model, input_shape, device, dtype):
    import torch

    x = torch.randn(*input_shape, device=device, dtype=dtype)

    def fwd():
        with torch.inference_mode():
            return model(x)

    return fwd


def run(recipe_path: str, out_path: str) -> int:
    import torch

    recipe: Recipe = load_recipe(recipe_path)
    _seed_all(recipe.measurement.seed)

    env = collect_env()
    clock_note = lock_gpu_clock(recipe.measurement.gpu_clock_lock)
    if clock_note:
        env["clock_lock_note"] = clock_note

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16}[recipe.runtime.dtype]

    started = datetime.now(timezone.utc).isoformat()

    # Cold-start: load weights
    def _load():
        from ultralytics import YOLO

        m = YOLO(recipe.model.weights)
        inner = m.model
        inner.eval()
        inner.to(device=device, dtype=torch_dtype)
        return m, inner

    (ult_model, inner_model), cold_start_ms = measure_cold_start(_load)

    # Optional torch.compile (with fallback to eager on failure — e.g. Triton missing on Windows)
    imgsz = recipe.measurement.input_size
    compile_note: str | None = None
    if recipe.runtime.mode == "compile":
        try:
            compiled = torch.compile(inner_model, mode="reduce-overhead")
            probe = torch.randn(1, 3, imgsz, imgsz, device=device, dtype=torch_dtype)
            with torch.inference_mode():
                _ = compiled(probe)
            inner_model = compiled
        except Exception as e:
            compile_note = f"torch.compile unavailable, fell back to eager: {type(e).__name__}: {e}"
            print(f"[warn] {compile_note}", file=sys.stderr)

    # Measure latency at each requested batch size
    per_bs: dict[int, dict] = {}
    for bs in recipe.measurement.batch_sizes:
        fwd = _make_forward(inner_model, (bs, 3, imgsz, imgsz), device, torch_dtype)
        stats = measure_latency(
            fwd,
            warmup_iters=recipe.measurement.warmup_iters,
            measure_iters=recipe.measurement.measure_iters,
        )
        per_bs[bs] = stats

    # Canonical latency = bs1
    lat = per_bs.get(1) or next(iter(per_bs.values()))
    throughput = ThroughputStats(
        bs1=throughput_from_latency(per_bs[1]["p50"], 1) if 1 in per_bs else None,
        bs8=throughput_from_latency(per_bs[8]["p50"], 8) if 8 in per_bs else None,
    )
    peak_mem = max((v.get("peak_gpu_mem_mb") or 0.0) for v in per_bs.values()) or None

    # Accuracy
    if os.environ.get("OMNI_SKIP_ACCURACY"):
        print("[info] OMNI_SKIP_ACCURACY set — skipping mAP eval", file=sys.stderr)
        acc = AccuracyStats()
    else:
        try:
            acc = evaluate_via_ultralytics(
                weights=recipe.model.weights,
                imgsz=imgsz,
                batch=1,
                device=0 if device == "cuda" else "cpu",
                half=(recipe.runtime.dtype == "fp16"),
            )
        except Exception as e:
            print(f"[warn] accuracy eval failed: {e}", file=sys.stderr)
            acc = AccuracyStats()

    # Model size
    try:
        weights_path = Path(recipe.model.weights)
        model_size_mb = weights_path.stat().st_size / (1024 ** 2) if weights_path.exists() else None
    except Exception:
        model_size_mb = None

    finished = datetime.now(timezone.utc).isoformat()

    # Constraints check
    meets = None
    c = recipe.constraints
    if c.min_fps_bs1 is not None and throughput.bs1 is not None:
        meets = throughput.bs1 >= c.min_fps_bs1
    # mAP drop check is relative to baseline; handled by recommend.py

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
        notes=compile_note,
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
