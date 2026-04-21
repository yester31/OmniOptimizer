"""Runner for Wave 6 CPU recipes (#30 fp32, #31 bf16, #32-33 int8,
#34-35 openvino).

Pipeline:
1. Resolve weights (reuses _weights_io._resolve_weights — handles
   trained_weights/ for Wave 5 modelopt / prune_24 recipes even though
   Wave 6 baselines are PTQ-only).
2. Export ultralytics weights to ONNX (cached, dynamic batch).
3. Dispatch on (technique.source, runtime.dtype) to the appropriate
   session-preparation function.
4. Measure latency via measure_latency (CPU-only path: no CUDA events,
   peak_gpu_mem stays None, NVML probe gracefully no-ops).
5. Accuracy falls back to ultralytics.YOLO(onnx_path).val(data=...).

Module invariants (enforced by test_run_cpu_imports_without_tensorrt):
- No top-level import of tensorrt, pycuda, or scripts.run_trt.
- onnxruntime and openvino imports live inside their dispatcher branches
  so the module loads cleanly even in environments missing one of them
  (e.g., openvino not installed on ARM).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._schemas import (  # noqa: E402
    AccuracyStats,
    EnvInfo,
    LatencyStats,
    Recipe,
    Result,
    ThroughputStats,
    load_recipe,
)
from scripts._weights_io import (  # noqa: E402 — TRT-free import
    _build_calib_numpy,
    _export_onnx,
    _iter_calib_samples,
    _resolve_weights,
)
from scripts.env_lock import collect_env  # noqa: E402
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


def _resolve_thread_count(recipe: Recipe) -> int:
    """Return the intra-op thread count to use for this recipe.

    - Explicit value in recipe.measurement.thread_count: use verbatim.
    - None: auto-detect physical cores.
      1st try psutil (cross-platform), then Linux /proc/cpuinfo,
      then fallback os.cpu_count() // 2 (hyperthreading assumption).

    Never returns 0 — pydantic validator forbids it but we also guard here
    so future refactors don't regress.
    """
    explicit = recipe.measurement.thread_count
    if explicit is not None:
        return int(explicit)

    # Try psutil
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n and n > 0:
            return int(n)
    except Exception:
        pass

    # Linux /proc fallback
    try:
        with open("/proc/cpuinfo", "r") as f:
            cores = set()
            current_phys: Optional[str] = None
            current_core: Optional[str] = None
            for line in f:
                if line.startswith("physical id"):
                    current_phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    current_core = line.split(":")[1].strip()
                elif line.strip() == "" and current_phys and current_core:
                    cores.add((current_phys, current_core))
                    current_phys = current_core = None
            if cores:
                return len(cores)
    except Exception:
        pass

    # Last resort: logical cores / 2 (hyperthread guess)
    logical = os.cpu_count() or 2
    return max(1, logical // 2)


# ---------------------------------------------------------------------------
# Dispatchers (Task 3 = fp32 only; Task 4 extends int8/bf16; Task 5 openvino)
# ---------------------------------------------------------------------------

def _build_ort_session_options(recipe: Recipe):
    """Common ORT SessionOptions: EXTENDED graph opts, explicit thread count."""
    import onnxruntime as ort  # lazy — allows test_imports_without_tensorrt to pass

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.intra_op_num_threads = _resolve_thread_count(recipe)
    so.inter_op_num_threads = 1  # single-graph inference, no parallel branches
    return so


def _prepare_ort_cpu_fp32(recipe: Recipe, session_options_factory: Callable):
    """FP32 path: plain ONNX + CPU EP.

    Returns a tuple of (session, input_name, output_names, cold_ms) so the
    caller can construct a forward_fn closure.
    """
    import onnxruntime as ort

    weights = _resolve_weights(recipe)
    imgsz = recipe.measurement.input_size
    onnx_cache = Path("results_cpu/_onnx")
    onnx_path = _export_onnx(weights, imgsz, half=False, cache_dir=onnx_cache,
                             dynamic=True)

    so = session_options_factory(recipe)

    def _load():
        return ort.InferenceSession(str(onnx_path), sess_options=so,
                                    providers=["CPUExecutionProvider"])

    session, cold_ms = measure_cold_start(_load)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    return session, input_name, output_names, cold_ms, onnx_path


# ---------------------------------------------------------------------------
# Calibration data builder (Recipe-aware wrapper over _weights_io helper)
# ---------------------------------------------------------------------------

def _build_calib_numpy_array(recipe: Recipe):
    """Recipe-aware wrapper around _weights_io._build_calib_numpy.

    Calibration YAML resolution order (Wave 5 pattern carried forward):
        1. OMNI_CALIB_YAML env var — overrides everything, used when eval
           and calibration datasets diverge (Wave 5 QR: eval on QR val,
           calibrate on COCO because QR val has only 133 images).
        2. OMNI_COCO_YAML env var — default COCO yaml path.
        3. recipe.technique.calibration_dataset — recipe-declared dataset
           name (e.g., "coco_val2017").

    Returns (N, 3, imgsz, imgsz) float32 numpy array, N = recipe.technique.
    calibration_samples or 512.
    """
    val_yaml = (
        os.environ.get("OMNI_CALIB_YAML")
        or os.environ.get("OMNI_COCO_YAML")
        or recipe.technique.calibration_dataset
    )
    n_samples = recipe.technique.calibration_samples or 512
    imgsz = recipe.measurement.input_size
    seed = recipe.technique.calibration_seed or recipe.measurement.seed or 42
    return _build_calib_numpy(val_yaml, n_samples, imgsz, seed)


# ---------------------------------------------------------------------------
# INT8 paths (Task 4)
# ---------------------------------------------------------------------------

class _NumpyReader:
    """onnxruntime CalibrationDataReader adapter.

    Accepts either a pre-built numpy array (N,3,H,W) OR an iterable / factory
    of (1,3,H,W) samples. The iterable path is the memory-lean option for
    static quant on memory-constrained hosts (avoids np.stack 2× peak).
    """

    def __init__(self, samples, input_name: str):
        self._input_name = input_name
        self._cached: list = []
        # Normalize: if samples is a numpy array, wrap as simple iterator.
        # If it's already an iterable/generator, use it directly. A factory
        # (callable returning iterable) lets us re-create on rewind() without
        # exhausting the generator.
        import numpy as np

        if callable(samples):
            self._factory = samples
            self._iter = iter(samples())
        elif isinstance(samples, np.ndarray):
            arr = samples
            self._factory = lambda a=arr: iter(a[i:i + 1] for i in range(len(a)))
            self._iter = self._factory()
        else:
            # Assume it's already an iterator/iterable; single pass only.
            self._factory = None
            self._iter = iter(samples)

    def get_next(self):
        try:
            sample = next(self._iter)
        except StopIteration:
            return None
        return {self._input_name: sample}

    def rewind(self):
        if self._factory is not None:
            self._iter = iter(self._factory())
        # else: single-pass iterator — rewind is a no-op (ORT calls rewind()
        # only for multi-pass calibrators; entropy + percentile are single-pass).


def _prepare_ort_cpu_int8_dynamic(recipe: Recipe, session_options_factory: Callable):
    """Dynamic INT8: weight-only quantization + activation scales computed
    at runtime per batch.

    Trade-off vs static:
    - Setup simpler (no calibration data)
    - Runtime slower (~10-20% per ORT benchmarks — activation quantization
      on every forward)
    - Accuracy identical for most conv-heavy models (weights dominate)
    """
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    weights = _resolve_weights(recipe)
    imgsz = recipe.measurement.input_size
    onnx_cache = Path("results_cpu/_onnx")
    onnx_cache.mkdir(parents=True, exist_ok=True)

    fp32_onnx = _export_onnx(weights, imgsz, half=False, cache_dir=onnx_cache,
                             dynamic=True)
    int8_onnx = onnx_cache / f"{recipe.name}_int8_dyn.onnx"

    if not int8_onnx.exists():
        # CPU EP gotcha: quantize_dynamic + per_channel=True + QInt8 produces
        # ConvInteger(10) which MLAS does not implement for Conv. The
        # documented-working combo for dynamic is per_channel=False + QUInt8,
        # which uses DynamicQuantizeLinear + QLinearConv path. Per-channel
        # accuracy gain is negligible for dynamic anyway (no calibration).
        quantize_dynamic(
            model_input=str(fp32_onnx),
            model_output=str(int8_onnx),
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=False,
        )

    so = session_options_factory(recipe)

    def _load():
        return ort.InferenceSession(str(int8_onnx), sess_options=so,
                                    providers=["CPUExecutionProvider"])

    session, cold_ms = measure_cold_start(_load)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    return session, input_name, output_names, cold_ms, int8_onnx


def _prepare_ort_cpu_int8_static(recipe: Recipe, session_options_factory: Callable):
    """Static INT8 with QDQ format: activation scales computed offline from
    calibration data. VNNI-optimized settings.

    Key flags (from Wave 3 P0-A patch — symmetry is non-negotiable on VNNI):
    - ActivationSymmetric=True, WeightSymmetric=True — required for MLAS
      INT8 dot-product kernel to avoid zero-point overhead
    - per_channel=True — per-output-channel weight scales on Conv
    - AddQDQPairToWeight=True — matches modelopt.onnx quantize convention
    - DedicatedQDQPair=False — CPU MLAS benefits from fused QDQ pairs
      (TRT explicit quantization wants True, inverse trade-off).
      Empirical decision deferred to Task 4 Step 3b (measure both).
    - reduce_range=False — VNNI full-range OK; pre-VNNI CPUs may need True
    """
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    weights = _resolve_weights(recipe)
    imgsz = recipe.measurement.input_size
    onnx_cache = Path("results_cpu/_onnx")
    onnx_cache.mkdir(parents=True, exist_ok=True)

    fp32_onnx = _export_onnx(weights, imgsz, half=False, cache_dir=onnx_cache,
                             dynamic=True)

    calibrator = recipe.technique.calibrator or "entropy"
    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    if calibrator not in method_map:
        raise ValueError(
            f"recipe {recipe.name!r} calibrator={calibrator!r} not supported "
            f"for ort_cpu int8 static; use one of {list(method_map)}"
        )

    int8_onnx = onnx_cache / f"{recipe.name}_int8_{calibrator}.onnx"

    if not int8_onnx.exists():
        # Pre-process (symbolic + ONNX shape inference, optimization) — required
        # by quantize_static to resolve activation shapes before inserting QDQ
        # nodes. Without this, ORT 1.22 writes a sibling `-inferred.onnx` it
        # then fails to re-open. Same pattern as run_trt ort_quant path; use
        # auto_merge=True for YOLO26n attention blocks (dynamic reshapes).
        preproc_onnx = onnx_cache / (fp32_onnx.stem + ".ortpp.onnx")
        if not preproc_onnx.exists():
            try:
                from onnxruntime.quantization.shape_inference import quant_pre_process
                quant_pre_process(
                    input_model_path=str(fp32_onnx),
                    output_model_path=str(preproc_onnx),
                    skip_optimization=False,
                    skip_onnx_shape=False,
                    skip_symbolic_shape=False,
                    auto_merge=True,
                    verbose=0,
                )
            except Exception as e:
                print(f"[warn] ort_cpu static preprocess failed ({e}); using raw ONNX",
                      file=sys.stderr)
                preproc_onnx = fp32_onnx

        # Discover ONNX input name from the preprocessed model so the reader
        # tensor key matches.
        tmp_sess = ort.InferenceSession(
            str(preproc_onnx), providers=["CPUExecutionProvider"],
        )
        input_name = tmp_sess.get_inputs()[0].name
        del tmp_sess

        # Streaming calibration: factory re-creates iterator on each pass
        # (entropy/percentile need 2 passes — first for histogram, second
        # for threshold). No bulk np.stack allocation, so 128-sample calib
        # fits on memory-constrained hosts (~5 MiB peak vs ~1.2 GiB bulk).
        val_yaml = (
            os.environ.get("OMNI_CALIB_YAML")
            or os.environ.get("OMNI_COCO_YAML")
            or recipe.technique.calibration_dataset
        )
        n_samples = recipe.technique.calibration_samples or 128
        seed = recipe.technique.calibration_seed or recipe.measurement.seed or 42

        def _calib_factory():
            return _iter_calib_samples(val_yaml, n_samples, imgsz, seed)

        reader = _NumpyReader(_calib_factory, input_name)
        quantize_static(
            model_input=str(preproc_onnx),
            model_output=str(int8_onnx),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            calibrate_method=method_map[calibrator],
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "AddQDQPairToWeight": True,
                # NOTE: Task 4 Step 3b is the empirical comparison for this
                # flag. Current default False per CPU MLAS fusion guidance;
                # flip to True if smoke shows latency regression.
                "DedicatedQDQPair": False,
            },
        )

    so = session_options_factory(recipe)

    def _load():
        return ort.InferenceSession(str(int8_onnx), sess_options=so,
                                    providers=["CPUExecutionProvider"])

    session, cold_ms = measure_cold_start(_load)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    return session, input_name, output_names, cold_ms, int8_onnx


def _prepare_cpu_session(recipe: Recipe):
    """Dispatcher on (source, dtype). Returns session object; structure
    depends on backend."""
    source = recipe.technique.source
    dtype = recipe.runtime.dtype

    if source == "ort_cpu" and dtype == "fp32":
        return _prepare_ort_cpu_fp32(recipe, _build_ort_session_options)

    if source == "ort_cpu" and dtype == "int8":
        # calibrator=None → dynamic (runtime activation scales).
        # calibrator in {minmax,entropy,percentile} → static QDQ.
        if recipe.technique.calibrator is None:
            return _prepare_ort_cpu_int8_dynamic(recipe, _build_ort_session_options)
        return _prepare_ort_cpu_int8_static(recipe, _build_ort_session_options)
    if source == "ort_cpu" and dtype == "bf16":
        raise NotImplementedError(
            "ort_cpu + bf16: implemented by Task 4 Step 6 "
            "(requires AMX or AVX-512 BF16 hardware)"
        )
    if source == "openvino":
        raise NotImplementedError(
            "openvino source: implemented by Task 5 "
            "(OpenVINO runtime + NNCF INT8 PTQ)"
        )

    raise RuntimeError(
        f"run_cpu does not support source={source!r} dtype={dtype!r}. "
        f"GPU recipes (trt_builtin/modelopt/ort_quant/brevitas) must use run_trt.py."
    )


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(recipe_path: str, out_path: str) -> int:
    recipe: Recipe = load_recipe(recipe_path)
    _seed_all(recipe.measurement.seed)

    env = collect_env()

    imgsz = recipe.measurement.input_size
    started = datetime.now(timezone.utc).isoformat()
    note_parts: list[str] = []

    per_bs: dict[int, dict] = {}
    cold_start_ms: Optional[float] = None

    try:
        session, input_name, output_names, cold_ms, onnx_path = _prepare_cpu_session(recipe)
        cold_start_ms = cold_ms
    except NotImplementedError as e:
        note_parts.append(str(e))
        session = None
    except Exception as e:
        note_parts.append(f"session prep failed: {e}")
        session = None

    if session is not None:
        import numpy as np

        for bs in recipe.measurement.batch_sizes:
            try:
                # Build a steady-state input tensor once, outside the timing
                # loop. This ensures we measure forward + NMS only, not the
                # preprocessing that would otherwise happen per-iter.
                x = np.random.randn(bs, 3, imgsz, imgsz).astype(np.float32)

                def _forward(s=session, n=input_name, inp=x, outs=output_names):
                    return s.run(outs, {n: inp})

                stats = measure_latency(
                    _forward,
                    warmup_iters=max(recipe.measurement.warmup_iters, 200),
                    measure_iters=max(recipe.measurement.measure_iters, 300),
                )
                per_bs[bs] = stats
            except Exception as e:
                note_parts.append(f"bs={bs}: run failed ({e})")

    if not per_bs:
        finished = datetime.now(timezone.utc).isoformat()
        result = Result(
            recipe=recipe.name,
            started_at=started,
            finished_at=finished,
            env=EnvInfo(**{k: v for k, v in env.items() if k in EnvInfo.model_fields}),
            latency_ms=LatencyStats(p50=float("nan"), p95=float("nan"), p99=float("nan")),
            throughput_fps=ThroughputStats(),
            peak_gpu_mem_mb=None,
            cold_start_ms=cold_start_ms,
            accuracy=AccuracyStats(),
            meets_constraints=False,
            notes="; ".join(note_parts) or "all batch sizes failed",
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json.loads(result.model_dump_json()), f, indent=2)
        print(f"wrote {out_path} (FAILED)")
        return 1

    lat = per_bs.get(1) or next(iter(per_bs.values()))
    throughput = ThroughputStats(
        bs1=throughput_from_latency(per_bs[1]["p50"], 1) if 1 in per_bs else None,
        bs8=throughput_from_latency(per_bs[8]["p50"], 8) if 8 in per_bs else None,
    )

    # Accuracy: reuse ultralytics val path on the ONNX file.
    acc = AccuracyStats()
    if os.environ.get("OMNI_SKIP_ACCURACY"):
        print("[info] OMNI_SKIP_ACCURACY set — skipping mAP eval", file=sys.stderr)
    else:
        try:
            from ultralytics import YOLO

            m = YOLO(str(onnx_path))
            data_yaml = _split.eval_yaml(
                os.environ.get("OMNI_COCO_YAML", "coco.yaml"),
                calib_yaml_path=_split.calib_yaml(),
                calib_seed=recipe.technique.calibration_seed or 42,
                calib_n=recipe.technique.calibration_samples or 512,
            )
            metrics = m.val(data=data_yaml, imgsz=imgsz, batch=1, device="cpu",
                            plots=False, verbose=False)
            acc = AccuracyStats(
                map_50_95=float(metrics.box.map),
                map_50=float(metrics.box.map50),
            )
        except Exception as e:
            note_parts.append(f"accuracy eval failed: {e}")

    finished = datetime.now(timezone.utc).isoformat()

    meets = None
    c = recipe.constraints
    if c.min_fps_bs1 is not None and throughput.bs1 is not None:
        meets = throughput.bs1 >= c.min_fps_bs1

    result = Result(
        recipe=recipe.name,
        started_at=started,
        finished_at=finished,
        env=EnvInfo(**{k: v for k, v in env.items() if k in EnvInfo.model_fields}),
        model_size_mb=None,
        latency_ms=LatencyStats(**{k: v for k, v in lat.items() if k in {"p50", "p95", "p99"}}),
        throughput_fps=throughput,
        peak_gpu_mem_mb=None,  # CPU runner: no GPU memory to report
        cold_start_ms=cold_start_ms,
        accuracy=acc,
        meets_constraints=meets,
        notes="; ".join(note_parts) or None,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json.loads(result.model_dump_json()), f, indent=2)
    print(f"wrote {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="OmniOptimizer CPU inference runner (Wave 6)")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    return run(args.recipe, args.out)


if __name__ == "__main__":
    sys.exit(main())
