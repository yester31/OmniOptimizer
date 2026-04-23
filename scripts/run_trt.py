"""Runner for native TensorRT recipes (#5 FP16, #6 INT8 PTQ, #7 INT8 + 2:4 Sparsity).

Pipeline:
1. Export ultralytics weights to ONNX (cached).
2. Build a TensorRT engine with the requested dtype / sparsity flags. INT8
   recipes run an entropy calibrator over ``calibration_samples`` random
   images from COCO val (seeded).
3. Deserialize the engine and measure latency via ``execute_async_v3``.
4. Accuracy falls back to ultralytics' engine-aware val path.

This file is the most intricate of the three runners. Where we hit edge cases
(e.g. TensorRT build errors, sparsity unsupported on non-Ampere hardware) we
record the problem in ``notes`` rather than crashing the whole ``make all``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
from scripts._weights_io import (  # noqa: E402
    _build_calib_numpy,
    _export_onnx,
    _letterbox,
    _load_yolo_for_restore,
    _resolve_weights,
)

# Module-level override set by run() when _resolve_weights returns a YOLO
# instance (modelopt_sparsify / modelopt_qat). Reset to None after run()
# completes to prevent state leaking between unit-test invocations.
_MAIN_TRAINED_YOLO = None  # type: ignore[var-annotated]


def _get_weights_or_yolo(recipe: Recipe):
    """Return _MAIN_TRAINED_YOLO if set (modelopt trained), else recipe.model.weights.

    Call sites that accept both str paths and YOLO instances (e.g. _export_onnx)
    should use this helper to transparently pick up a restored modelopt model.
    """
    return _MAIN_TRAINED_YOLO if _MAIN_TRAINED_YOLO is not None else recipe.model.weights


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _apply_modelopt_sparsify(weights: str, imgsz: int):
    """Return a ultralytics YOLO whose backbone weights carry the 2:4 pattern.

    Loads a fresh YOLO (preserves ultralytics metadata), runs modelopt 2:4
    magnitude pruning on ``yolo.model``, exports the sparsified module to
    strip modelopt wrappers, then swaps the weights back into the YOLO
    wrapper via ``load_state_dict``. Downstream ONNX export path stays
    identical to the non-sparse case — same ``YOLO.export(...)`` entry
    point.

    Fallback: if this torch-level path proves brittle against future
    ultralytics/modelopt versions, Plan B is ONNX graph-level weight
    masking via onnx-graphsurgeon (zero out the right lanes per 4-column
    block). Not implemented now; add only if measured necessary.
    """
    try:
        from modelopt.torch.sparsity import sparsify as mts_sparsify
        from modelopt.torch.sparsity import export as mts_export
    except ImportError as e:
        raise RuntimeError(
            "nvidia-modelopt torch extension not installed. Install with: "
            "pip install --extra-index-url https://pypi.nvidia.com nvidia-modelopt"
        ) from e

    import torch
    from ultralytics import YOLO

    yolo = YOLO(weights)
    inner = yolo.model
    device = next(inner.parameters()).device
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)

    def _loader():
        yield dummy

    config = {"data_loader": _loader(), "collect_func": lambda x: x}
    sparse_model = mts_sparsify(inner, mode="sparse_magnitude", config=config)
    sparse_model = mts_export(sparse_model)

    yolo.model.load_state_dict(sparse_model.state_dict())
    print("[info] modelopt 2:4 sparse_magnitude applied + exported", file=sys.stderr)
    return yolo


def _modelopt_onnx_tag(recipe: Recipe, imgsz: int, *, dynamic: bool = True) -> str:
    """Generate the cache filename for a modelopt-quantized ONNX artifact.

    Extracted for testability (Wave 16 T7). Pure: no filesystem access, same
    inputs → same output.

    Tag components, in order:
      - weights stem (e.g. ``best_qr``)
      - image size
      - ``modelopt``
      - base calibrator (``max``/``entropy``/``percentile``)
      - ``_asym`` if ``*_asymmetric`` calibrator suffix present (Wave 14 A5)
      - ``_sparse24`` if 2:4 sparsity preprocess enabled
      - ``_ex<sha8>`` if ``nodes_to_exclude`` is non-empty (Wave 16 T7 fix)
      - ``bs1`` or ``dyn``

    The Wave 16 T7 fix: recipes #09 (no exclusions) and #12 (4 Convs excluded)
    previously shared the same cache filename. Whichever ran first poisoned
    the other. Hashing a sorted ``|``-joined key keeps the tag short and
    order-insensitive; empty ``nodes_to_exclude`` still produces the legacy
    path so existing #09 artifacts remain valid.
    """
    calibrator = recipe.technique.calibrator or "max"
    use_zero_point = calibrator.endswith("_asymmetric")
    base_calibrator = calibrator[: -len("_asymmetric")] if use_zero_point else calibrator
    sparsity_tag = "_sparse24" if recipe.technique.sparsity_preprocess == "2:4" else ""
    asym_tag = "_asym" if use_zero_point else ""
    bs_tag = "dyn" if dynamic else "bs1"
    excludes_tag = ""
    if recipe.technique.nodes_to_exclude:
        excludes_key = "|".join(sorted(recipe.technique.nodes_to_exclude))
        excludes_tag = "_ex" + hashlib.sha256(excludes_key.encode("utf-8")).hexdigest()[:8]
    return (
        f"{Path(recipe.model.weights).stem}_{imgsz}_modelopt_"
        f"{base_calibrator}{asym_tag}{sparsity_tag}{excludes_tag}_{bs_tag}.onnx"
    )


def _prepare_modelopt_onnx(recipe: Recipe, imgsz: int, cache_dir: Path,
                           dynamic: bool = True) -> Path:
    """Quantize via modelopt.onnx — takes ultralytics' clean ONNX export and
    injects QDQ nodes using COCO calibration images.

    This path preserves ultralytics' inference-head wiring (NMS-ready output
    tensor), unlike direct torch-level quantize + torch.onnx.export which
    bypasses the wrapper and breaks validator post-processing.

    When ``technique.sparsity_preprocess == '2:4'``, we first pre-prune the
    weights via modelopt.torch.sparsity so the resulting QDQ-ONNX carries
    zeros in the 2:4 pattern; TensorRT's SPARSE_WEIGHTS flag then actually
    selects sparse INT8 kernels. When ``technique.nodes_to_exclude`` is set,
    those ONNX node names stay at FP16 inside the QDQ graph.
    """
    try:
        from modelopt.onnx.quantization import quantize as moq_quantize
    except ImportError as e:
        raise RuntimeError(
            "nvidia-modelopt not installed. Install with: "
            "pip install --extra-index-url https://pypi.nvidia.com nvidia-modelopt"
        ) from e

    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = _modelopt_onnx_tag(recipe, imgsz, dynamic=dynamic)
    cached = cache_dir / tag
    if cached.exists():
        return cached

    # Re-derive the calibrator/sparsity bits the quantize call + log line need.
    # Kept in sync with _modelopt_onnx_tag (the source of truth for the naming).
    calibrator = recipe.technique.calibrator or "max"
    use_zero_point = calibrator.endswith("_asymmetric")
    base_calibrator = calibrator[: -len("_asymmetric")] if use_zero_point else calibrator
    sparsity_tag = "_sparse24" if recipe.technique.sparsity_preprocess == "2:4" else ""

    if recipe.technique.sparsity_preprocess == "2:4":
        # For modelopt trained recipes, _get_weights_or_yolo() returns the
        # already-restored YOLO; _apply_modelopt_sparsify expects a str path,
        # so fall through to the plain weights when a trained YOLO is present.
        if _MAIN_TRAINED_YOLO is not None:
            yolo = _MAIN_TRAINED_YOLO
        else:
            yolo = _apply_modelopt_sparsify(recipe.model.weights, imgsz)
        clean_onnx = _export_onnx(yolo, imgsz, half=False,
                                  cache_dir=cache_dir, dynamic=dynamic,
                                  tag_suffix="_sparse24")
    else:
        clean_onnx = _export_onnx(_get_weights_or_yolo(recipe), imgsz, half=False,
                                  cache_dir=cache_dir, dynamic=dynamic)

    samples = recipe.technique.calibration_samples or 512
    seed = recipe.technique.calibration_seed or 42
    val_yaml = _split.calib_yaml()
    calib_data = _build_calib_numpy(val_yaml, samples, imgsz, seed)

    quant_kwargs = dict(
        onnx_path=str(clean_onnx),
        quantize_mode="int8",
        calibration_method=base_calibrator,
        calibration_data=calib_data,
        output_path=str(cached),
        log_level="WARNING",
        use_zero_point=use_zero_point,
    )
    if recipe.technique.nodes_to_exclude:
        quant_kwargs["nodes_to_exclude"] = list(recipe.technique.nodes_to_exclude)

    print(
        f"[info] modelopt.onnx.quantize: method={base_calibrator}, "
        f"asymmetric={use_zero_point}, "
        f"samples={calib_data.shape[0]}, "
        f"sparsity={sparsity_tag or 'none'}, "
        f"excludes={len(recipe.technique.nodes_to_exclude or [])}, "
        f"onnx={clean_onnx.name}",
        file=sys.stderr,
    )
    moq_quantize(**quant_kwargs)
    print(f"[info] modelopt wrote QDQ onnx: {cached}", file=sys.stderr)
    return cached


def _prepare_ort_quant_onnx(recipe: Recipe, imgsz: int, cache_dir: Path,
                            dynamic: bool = True) -> Path:
    """Quantize via ``onnxruntime.quantization.quantize_static`` — emits QDQ
    ONNX that TensorRT's explicit-quantization path consumes identically to
    modelopt's output.

    Calibration method map (1:1 with ``onnxruntime.quantization.CalibrationMethod``):
    ``minmax`` / ``entropy`` / ``percentile`` / ``distribution``.

    TRT requires symmetric per-tensor activation and per-channel symmetric
    weight quantization. ORT defaults to asymmetric activation, which TRT
    rejects (or silently falls back), so we force symmetry via ``extra_options``
    — this matches what modelopt does internally.
    """
    # Validate calibrator BEFORE importing onnxruntime so a bad recipe fails
    # with a clear ValueError even in environments without onnxruntime
    # installed (tests, minimal dev envs).
    _ORT_CALIBRATORS = ("minmax", "entropy", "percentile", "distribution")
    calibrator = (recipe.technique.calibrator or "minmax").lower()
    if calibrator not in _ORT_CALIBRATORS:
        raise ValueError(
            f"ort_quant backend supports calibrator in {list(_ORT_CALIBRATORS)}, "
            f"got {calibrator!r}"
        )

    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime.quantization not available. Install onnxruntime>=1.17."
        ) from e

    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "distribution": CalibrationMethod.Distribution,
    }

    requested_samples = int(recipe.technique.calibration_samples or 512)
    # Histogram-based ORT calibrators (entropy / percentile / distribution)
    # accumulate every intermediate activation tensor across every calibration
    # sample in RAM before computing histograms — for YOLO26n at 640×640 this
    # runs out of memory above ~128 samples. MinMax keeps running min/max only,
    # so it scales to 512. Cap here with a warning rather than silently OOM'ing
    # mid-run.
    _ORT_HISTOGRAM_SAMPLE_CAP = 128
    if calibrator != "minmax" and requested_samples > _ORT_HISTOGRAM_SAMPLE_CAP:
        print(
            f"[warn] ort_quant {calibrator} calibrator: capping samples "
            f"{requested_samples} -> {_ORT_HISTOGRAM_SAMPLE_CAP} to avoid "
            f"activation-accumulator OOM",
            file=sys.stderr,
        )
        n_samples = _ORT_HISTOGRAM_SAMPLE_CAP
    else:
        n_samples = requested_samples
    seed = int(recipe.technique.calibration_seed or 42)
    bs_tag = "dyn" if dynamic else "bs1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / (
        f"{Path(recipe.model.weights).stem}_{imgsz}_ort_{calibrator}_"
        f"{n_samples}_s{seed}_{bs_tag}.qdq.onnx"
    )
    if cached.exists():
        print(f"[info] ort_quant cache hit: {cached.name}", file=sys.stderr)
        return cached

    clean_onnx = _export_onnx(_get_weights_or_yolo(recipe), imgsz, half=False,
                              cache_dir=cache_dir, dynamic=dynamic)

    # ORT's quantize_static recommends running shape inference + model
    # optimization first. Without it, histogram-based calibrators (entropy,
    # percentile, distribution) can hit "bad allocation" mid-inference on
    # attention-heavy graphs because unfolded shape ops inflate the activation
    # memory footprint. This preprocess pass produces a leaner ONNX we can
    # hand to quantize_static.
    preproc_path = cache_dir / (clean_onnx.stem + ".ortpp.onnx")
    if not preproc_path.exists():
        try:
            from onnxruntime.quantization.shape_inference import quant_pre_process

            quant_pre_process(
                input_model_path=str(clean_onnx),
                output_model_path=str(preproc_path),
                skip_optimization=False,
                skip_onnx_shape=False,
                skip_symbolic_shape=False,
                auto_merge=True,
                verbose=0,
            )
            print(f"[info] ort_quant preprocessed: {preproc_path.name}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] ort_quant preprocess failed ({e}); using raw ONNX", file=sys.stderr)
            preproc_path = clean_onnx
    source_onnx = preproc_path

    val_yaml = _split.calib_yaml()
    calib_arr = _build_calib_numpy(val_yaml, n_samples, imgsz, seed)

    import onnx

    model_proto = onnx.load(str(source_onnx))
    input_name = model_proto.graph.input[0].name
    del model_proto

    class _NumpyReader(CalibrationDataReader):
        def __init__(self, arr, name: str):
            self._iter = iter(arr)
            self._name = name

        def get_next(self):
            try:
                x = next(self._iter)
            except StopIteration:
                return None
            if x.ndim == 3:
                x = x[None, ...]
            return {self._name: x}

    nodes_to_exclude = list(recipe.technique.nodes_to_exclude or [])
    print(
        f"[info] onnxruntime.quantize_static: method={calibrator}, "
        f"samples={n_samples}, excludes={len(nodes_to_exclude)}, "
        f"onnx={clean_onnx.name}",
        file=sys.stderr,
    )
    quantize_static(
        model_input=str(source_onnx),
        model_output=str(cached),
        calibration_data_reader=_NumpyReader(calib_arr, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=method_map[calibrator],
        nodes_to_exclude=nodes_to_exclude or None,
        # Restrict QDQ injection to op types TRT benefits from. ORT's default
        # covers ~30 op types including Unsqueeze / Reshape / etc., which
        # produces QDQ on scalar constants inside YOLO's attention — TRT's
        # ONNX importer then rejects the engine build with
        #   "Assertion failed: (axis >= 0 && axis <= nbDims)"
        # because per-channel axis=1 on a rank-0 tensor is nonsensical.
        # Matches modelopt's default (Conv + MatMul + Gemm only).
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            # QuantizeBias=False: TRT computes INT32 Conv bias internally from
            # activation_scale × weight_scale during INT8 execution. Leaving
            # bias as FP32 in the QDQ graph is the contract TRT expects.
            # ORT's default (True) folds bias into an INT32 initializer + DQ,
            # which TRT rejects ("only activation datatypes allowed as input").
            "QuantizeBias": False,
            # AddQDQPairToWeight OFF: folded single-DQ weight is TRT's preferred
            # format. The pair variant adds redundant Q on INT8 weights.
            "DedicatedQDQPair": True,
        },
    )
    print(f"[info] ort_quant wrote QDQ onnx: {cached}", file=sys.stderr)
    return cached


# Short tags appear in engine cache filenames. Long ``technique.source`` names
# push paths past Windows' 260-char MAX_PATH when combined with the
# imgsz/calibrator/bs/version suffixes already in the tag.
_SOURCE_TAG = {
    "trt_builtin": "",
    "modelopt": "_modelopt",
    "ort_quant": "_ort",
}


def _prepare_onnx(recipe: Recipe, imgsz: int, cache_dir: Path,
                  bs: int) -> tuple[Path, bool]:
    """Dispatch to the right ONNX preparation path based on technique.source.

    For bs=1 we prefer a static ONNX because ultralytics' dynamic export
    carries extra shape-tracking nodes that TRT does not fold as aggressively
    — measured ~40% bs=1 slowdown on YOLO26n. For bs>1 we must use dynamic.

    Returns (onnx_path, quant_preapplied). When quant_preapplied is True the
    ONNX carries QDQ nodes and `_build_engine` must skip its calibrator.
    """
    source = recipe.technique.source
    dynamic = bs > 1
    if source == "trt_builtin":
        path = _export_onnx(_get_weights_or_yolo(recipe), imgsz, half=False,
                            cache_dir=cache_dir, dynamic=dynamic)
        return path, False
    if source == "modelopt":
        return _prepare_modelopt_onnx(recipe, imgsz, cache_dir, dynamic=dynamic), True
    if source == "ort_quant":
        return _prepare_ort_quant_onnx(recipe, imgsz, cache_dir, dynamic=dynamic), True
    raise ValueError(f"unknown technique.source: {source!r}")


def _resolve_val_image_paths(yaml_path: str) -> list[str]:
    """Thin alias for :func:`scripts._split.resolve_val_image_paths`.
    Kept for backwards compat with callers below; prefer the shared helper."""
    return _split.resolve_val_image_paths(yaml_path)


def _make_coco_calibrator(shape, n_samples: int, cache_path: Path, seed: int,
                          val_yaml_path: str):
    """INT8 entropy calibrator backed by real COCO val images.

    Far more accurate than the random-normal stand-in because the entropy
    calibrator derives per-tensor scales from activation *distributions* — and
    real imagery produces distributions that match the deployment regime.
    Expect the random fallback to drop mAP double-digit %p; this one should
    bring the drop into the sub-1%p range for reasonable PTQ settings.
    """
    import numpy as np
    import tensorrt as trt
    import torch

    all_paths = _resolve_val_image_paths(val_yaml_path)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_paths)
    img_paths = all_paths[:n_samples]
    imgsz = shape[2]
    bs = shape[0]

    class _CocoCal(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self._buf = torch.empty(shape, device="cuda", dtype=torch.float32)
            self._idx = 0

        def get_batch_size(self):
            return bs

        def get_batch(self, names):  # noqa: ARG002
            import cv2

            if self._idx >= len(img_paths):
                return None
            batch = []
            for k in range(bs):
                j = self._idx + k
                if j >= len(img_paths):
                    batch.append(batch[-1])
                    continue
                img = cv2.imread(img_paths[j])
                if img is None:
                    batch.append(np.zeros((3, imgsz, imgsz), dtype=np.float32))
                else:
                    batch.append(_letterbox(img, imgsz))
            host = np.stack(batch, axis=0)
            self._buf.copy_(torch.from_numpy(host))
            self._idx += bs
            return [int(self._buf.data_ptr())]

        def read_calibration_cache(self):
            if cache_path.exists():
                return cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            cache_path.write_bytes(cache)

    return _CocoCal()


def _make_random_calibrator(shape, n_samples: int, cache_path: Path, seed: int):
    """Random-normal INT8 calibrator. Used as fallback when no dataset yaml is
    available. Known to produce large mAP drops — see ``_make_coco_calibrator``
    for the real path.
    """
    import numpy as np
    import tensorrt as trt
    import torch

    class _TorchCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            trt.IInt8EntropyCalibrator2.__init__(self)
            rng = np.random.default_rng(seed)
            host = rng.standard_normal(tuple(shape), dtype=np.float32)
            # Keep the tensor alive on self so its device pointer stays valid
            # for the whole calibration loop.
            self._t = torch.from_numpy(host).to("cuda")
            self._i = 0

        def get_batch_size(self):
            return shape[0]

        def get_batch(self, names):  # noqa: ARG002
            if self._i >= n_samples:
                return None
            self._i += 1
            return [int(self._t.data_ptr())]

        def read_calibration_cache(self):
            if cache_path.exists():
                return cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            cache_path.write_bytes(cache)

    return _TorchCalibrator()


def _timing_cache_path() -> Path:
    """Shared layer-timing cache across recipes. Kept per (TRT, CUDA)
    by relying on TRT's own ignore_mismatch=False — if the version
    changes we just get a fresh cache, not a corrupted reuse."""
    return Path("results/_trt_timing.cache")


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    dtype: str,
    sparsity: Optional[str],
    batch_size: int,
    imgsz: int,
    calib_samples: int,
    calib_seed: int,
    quant_preapplied: bool = False,
    enable_tf32: bool = False,
    builder_optimization_level: Optional[int] = None,
    build_ceiling_s: Optional[int] = None,
) -> tuple[Optional[Path], Optional[str], Optional[float]]:
    """Build a TensorRT engine. Returns (engine_path, note-or-None, build_time_s).

    Shares a layer-timing cache across recipes — first build warms it,
    subsequent builds reuse layer-tactic timings so only truly new layers
    are re-profiled. Per the TensorRT performance guide, the cache is
    keyed on device / CUDA / TRT version / builder flags, so reusing it
    across recipes with different dtype or flags is safe (TRT rejects
    mismatched entries rather than silently using wrong timings).

    Wave 14 A1: ``builder_optimization_level`` maps directly to
    ``nvinfer.BuilderConfig.builder_optimization_level`` (range 0-5,
    default=3). Level 5 runs exhaustive autotune; build time grows 3-5x
    but tactic selection stabilizes and runtime fps typically climbs
    +5-15%.

    Wave 15 D3: ``build_ceiling_s`` (default 600s) is diagnostic only —
    exceeding it logs a structured warning but the build still completes
    and the engine is returned. Rationale: ceiling-triggered hard-fails
    force the operator to rebuild with a different flag set; for audit /
    reproducibility runs we prefer a completed engine + clear note so
    recommend.py can still rank the recipe (with the build-time penalty
    visible in Result.build_time_s).
    """
    import time as _time
    t0 = _time.perf_counter()
    try:
        import tensorrt as trt  # noqa: F401
    except Exception as e:
        return None, f"tensorrt import failed: {e}", None

    import tensorrt as trt

    if engine_path.exists():
        return engine_path, None, None

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            return None, f"onnx parse failed:\n{errs}"

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # Load (or create) the shared timing cache before any flags that
    # affect layer selection — TRT consults the cache during tactic
    # timing, which happens later in build_serialized_network.
    cache_path = _timing_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_bytes = cache_path.read_bytes() if cache_path.exists() else b""
    timing_cache = config.create_timing_cache(cache_bytes)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)

    if dtype == "fp32" and enable_tf32:
        # TF32 (10-bit mantissa, FP32 dynamic range) on Ampere+ tensor cores.
        # Near-zero accuracy impact; small speedup on Conv-heavy workloads.
        config.set_flag(trt.BuilderFlag.TF32)
        print("[info] FP32 with TF32 tensor cores enabled", file=sys.stderr)
    elif dtype == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif dtype == "bf16":
        # Wave 14 A2: BF16 on Ampere sm_80+ (RTX 3060 Laptop = sm_86 OK).
        # BuilderFlag.BF16 present from TRT 9.0+. BF16 shares FP32 exponent range
        # so overflow-sensitive weights fare better than FP16 at similar throughput.
        if sparsity == "2:4":
            return None, (
                "BF16 + SPARSE_WEIGHTS is untested on sm_86 and the matrix is not "
                "guaranteed in TRT 10.x docs — Wave 14 Task 2.6 guard."
            ), None
        config.set_flag(trt.BuilderFlag.BF16)
        print("[info] BF16 builder flag set", file=sys.stderr)
    elif dtype == "int8":
        if not builder.platform_has_fast_int8:
            return None, "platform does not support fast INT8", None
        config.set_flag(trt.BuilderFlag.INT8)
        if quant_preapplied:
            # QDQ nodes carry the scales; no calibrator needed. INT8 flag is
            # still required so TRT selects INT8 tactics. modelopt's default
            # high_precision_dtype=fp16 marks residual layers as fp16, so we
            # also enable FP16 — mixed INT8/FP16 is what modelopt targets.
            config.set_flag(trt.BuilderFlag.FP16)
            print("[info] INT8+FP16: QDQ-preapplied ONNX (no calibrator)", file=sys.stderr)
        else:
            cache_file = engine_path.with_suffix(".calib")
            val_yaml = _split.calib_yaml()
            calibrator = None
            # Calibrator batch must match the optimization profile's batch,
            # otherwise TRT rejects the calibration shape on engine build
            # for bs > 1. Previously hardcoded to 1, which silently relied
            # on a cached bs=8 engine from an earlier era.
            calib_shape = (batch_size, 3, imgsz, imgsz)
            if val_yaml and Path(val_yaml).exists():
                try:
                    calibrator = _make_coco_calibrator(
                        shape=calib_shape,
                        n_samples=calib_samples,
                        cache_path=cache_file,
                        seed=calib_seed,
                        val_yaml_path=val_yaml,
                    )
                    print(f"[info] INT8 calib: coco images from {val_yaml} (bs={batch_size})", file=sys.stderr)
                except Exception as e:
                    print(f"[warn] coco calibrator failed ({e})", file=sys.stderr)
            if calibrator is None:
                if os.environ.get("OMNI_ALLOW_RANDOM_CALIB") != "1":
                    return None, (
                        "INT8 calibration data unavailable: set OMNI_COCO_YAML to a "
                        "real ultralytics dataset yaml, or pass OMNI_ALLOW_RANDOM_CALIB=1 "
                        "to explicitly opt in to the random-normal fallback "
                        "(known to drop mAP by double digits)."
                    ), None
                calibrator = _make_random_calibrator(
                    shape=calib_shape,
                    n_samples=calib_samples,
                    cache_path=cache_file,
                    seed=calib_seed,
                )
                print("[warn] OMNI_ALLOW_RANDOM_CALIB=1: random-normal INT8 calibration",
                      file=sys.stderr)
            config.int8_calibrator = calibrator

    if sparsity == "2:4":
        try:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        except AttributeError:
            return None, "TensorRT build lacks SPARSE_WEIGHTS flag", None

    # Wave 14 A1: builder_optimization_level (TRT 10.x, 0-5, default=3).
    if builder_optimization_level is not None:
        config.builder_optimization_level = int(builder_optimization_level)
        print(
            f"[info] builder_optimization_level={builder_optimization_level} "
            f"(TRT default=3; higher = longer build, better tactic search)",
            file=sys.stderr,
        )

    # Fixed shape profile so latency is deterministic.
    profile = builder.create_optimization_profile()
    name = network.get_input(0).name
    shape = (batch_size, 3, imgsz, imgsz)
    profile.set_shape(name, shape, shape, shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    build_time_s = _time.perf_counter() - t0
    if serialized is None:
        return None, "engine build returned None", build_time_s

    # Wave 14 outside voice F7 + Wave 15 D3: recipe-configurable ceiling.
    # opt_level=5 on bigger models can take 10+ minutes; warn so the
    # operator can decide whether to lower the level for later recipes.
    # Never hard-fails — diagnostic output only, engine returned regardless.
    _ceiling = build_ceiling_s if build_ceiling_s is not None else 600
    if build_time_s > _ceiling:
        print(
            f"[warn] build_time_s={build_time_s:.0f}s exceeds {_ceiling}s ceiling "
            f"(builder_optimization_level={builder_optimization_level or 3}) — "
            f"consider lowering the level or raising measurement.build_ceiling_s",
            file=sys.stderr,
        )

    # Persist the timing cache so the next engine build benefits. TRT
    # updates the cache in place during tactic profiling; we just
    # serialize whatever is in the config's cache slot back to disk.
    try:
        updated_cache = config.get_timing_cache()
        if updated_cache is not None:
            cache_path.write_bytes(bytes(updated_cache.serialize()))
    except Exception as e:
        print(f"[warn] timing cache write failed: {e}", file=sys.stderr)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return engine_path, None, build_time_s


def _make_trt_forward(engine_path: Path, batch_size: int, imgsz: int):  # pragma: no cover
    """Build a TRT forward callable backed entirely by torch.cuda memory.

    No pycuda: we share torch's CUDA context, which coexists cleanly with the
    ultralytics+torch ONNX export path earlier in the pipeline. All I/O tensors
    live on the returned closure as attributes so they are kept alive for the
    duration of the benchmarking loop.
    """
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    input_shape = (batch_size, 3, imgsz, imgsz)
    context.set_input_shape(input_name, input_shape)

    trt_to_torch = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }

    device = torch.device("cuda")
    io_tensors: list = []  # hold refs to prevent GC freeing device memory
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype_torch = trt_to_torch.get(engine.get_tensor_dtype(name), torch.float32)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            t = torch.randn(input_shape, device=device, dtype=torch.float32).to(dtype_torch)
        else:
            out_shape = tuple(context.get_tensor_shape(name))
            t = torch.empty(out_shape, device=device, dtype=dtype_torch)
        io_tensors.append(t)
        context.set_tensor_address(name, int(t.data_ptr()))

    stream = torch.cuda.Stream()

    # Attempt CUDA graph capture to amortise Python + TRT per-call launch
    # overhead (H1 in docs/improvements/2026-04-18-trt-modelopt-audit.md).
    # IO tensor addresses were already registered via set_tensor_address above
    # and must stay stable across replays — io_tensors is kept alive on the
    # returned closure. On capture failure (e.g. a kernel with data-dependent
    # control flow sneaks in) we fall back to direct execute_async_v3.
    graph: Optional["torch.cuda.CUDAGraph"] = None
    try:
        # Prime internal state once before capture, as recommended by the TRT
        # CUDA graphs guide.
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            context.execute_async_v3(stream.cuda_stream)
        graph = g
        print("[info] CUDA graph captured", file=sys.stderr)
    except Exception as e:  # pragma: no cover - runtime-only CUDA path
        print(
            f"[warn] CUDA graph capture failed ({e}); falling back to "
            "direct execute_async_v3",
            file=sys.stderr,
        )
        graph = None

    if graph is not None:
        _graph_ref = graph

        def fwd() -> None:
            _graph_ref.replay()
            stream.synchronize()
    else:
        def fwd() -> None:
            with torch.cuda.stream(stream):
                context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()

    # Pin long-lived references so Python doesn't free them between calls.
    fwd._io = io_tensors  # type: ignore[attr-defined]
    fwd._context = context  # type: ignore[attr-defined]
    fwd._stream = stream  # type: ignore[attr-defined]
    fwd._graph = graph  # type: ignore[attr-defined]
    return fwd, engine


def run(recipe_path: str, out_path: str) -> int:
    global _MAIN_TRAINED_YOLO
    recipe: Recipe = load_recipe(recipe_path)
    _seed_all(recipe.measurement.seed)

    # spec §6 / Task 8: resolve weights. May return YOLO instance for
    # modelopt trained recipes; downstream call sites use _get_weights_or_yolo().
    _MAIN_TRAINED_YOLO = None
    resolved = _resolve_weights(recipe)
    if isinstance(resolved, (str, Path)):
        recipe.model.weights = str(resolved)
    else:
        _MAIN_TRAINED_YOLO = resolved

    env = collect_env()
    clock_note = lock_gpu_clock(recipe.measurement.gpu_clock_lock)
    if clock_note:
        env["clock_lock_note"] = clock_note

    imgsz = recipe.measurement.input_size
    onnx_cache = Path("results/_onnx")
    engine_cache = Path("results/_engines")
    engine_cache.mkdir(parents=True, exist_ok=True)

    # Engine plan files are not portable across TRT / CUDA versions.
    # Tagging the filename with trt<major>.<minor>_cuda<major>.<minor>
    # keeps old plans out of new runs after a library upgrade, instead of
    # silently re-using an incompatible engine. Falls back to "trtX_cudaY"
    # if the version string is not in an expected form.
    def _version_tag() -> str:
        def _mm(s: str, prefix: str) -> str:
            if not s:
                return f"{prefix}?"
            parts = s.split(".")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                return f"{prefix}{parts[0]}.{parts[1]}"
            return f"{prefix}{parts[0]}"
        return f"{_mm(str(env.get('tensorrt')), 'trt')}_{_mm(str(env.get('cuda')), 'cuda')}"

    version_tag = _version_tag()

    dtype = recipe.runtime.dtype
    sparsity = recipe.runtime.sparsity
    # runtime.mode == "tf32" opts the FP32 path into TF32 tensor cores on
    # Ampere+. No-op for FP16/INT8.
    enable_tf32 = (recipe.runtime.mode or "").lower() == "tf32"

    started = datetime.now(timezone.utc).isoformat()
    note_parts: list[str] = []

    per_bs: dict[int, dict] = {}
    cold_start_ms: Optional[float] = None
    bs1_engine: Optional[Path] = None  # referenced later for mAP eval
    build_time_s_bs1: Optional[float] = None  # Wave 14 A1: bs1 build time
    # Wave 16 D1: True once any bs build exceeded the ceiling, False once any
    # build succeeded under the ceiling, None if every build failed. Writes
    # True are sticky — a later under-ceiling build cannot mask an earlier
    # breach.
    build_ceiling_breached: Optional[bool] = None

    source = recipe.technique.source
    source_suffix = _SOURCE_TAG.get(source, f"_{source}")
    tf32_suffix = "_tf32" if enable_tf32 else ""
    opt_suffix = (
        f"_opt{recipe.runtime.builder_optimization_level}"
        if recipe.runtime.builder_optimization_level is not None
        else ""
    )
    for bs in recipe.measurement.batch_sizes:
        onnx_path, quant_preapplied = _prepare_onnx(recipe, imgsz, onnx_cache, bs)
        engine_tag = f"{onnx_path.stem}_{dtype}{tf32_suffix}{opt_suffix}{'_sparse' if sparsity else ''}{source_suffix}_bs{bs}_{version_tag}.engine"
        engine_path = engine_cache / engine_tag

        built, err, build_time_s = _build_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            dtype=dtype,
            sparsity=sparsity,
            batch_size=bs,
            imgsz=imgsz,
            calib_samples=recipe.technique.calibration_samples or 0,
            calib_seed=recipe.technique.calibration_seed or recipe.measurement.seed,
            quant_preapplied=quant_preapplied,
            enable_tf32=enable_tf32,
            builder_optimization_level=recipe.runtime.builder_optimization_level,
            build_ceiling_s=recipe.measurement.build_ceiling_s,
        )
        if built is None:
            note_parts.append(f"bs={bs}: build failed ({err})")
            continue
        if bs == 1:
            bs1_engine = built
            build_time_s_bs1 = build_time_s

        # Wave 16 D1: track ceiling breach for round-trip into Result JSON.
        # Mirror _build_engine's 600s default when recipe doesn't override.
        _ceiling = (
            recipe.measurement.build_ceiling_s
            if recipe.measurement.build_ceiling_s is not None
            else 600
        )
        if build_time_s > _ceiling:
            build_ceiling_breached = True
        elif build_ceiling_breached is None:
            build_ceiling_breached = False

        try:
            def _load(e=built, b=bs):
                return _make_trt_forward(e, b, imgsz)

            (fwd_and_engine, cold_ms) = measure_cold_start(_load)
            fwd, _engine = fwd_and_engine
            if cold_start_ms is None:
                cold_start_ms = cold_ms
            stats = measure_latency(
                fwd,
                warmup_iters=recipe.measurement.warmup_iters,
                measure_iters=recipe.measurement.measure_iters,
            )
            per_bs[bs] = stats
        except Exception as e:
            note_parts.append(f"bs={bs}: run failed ({e})")

    if not per_bs:
        # Record the failure as a Result so recommend.py can still surface it.
        finished = datetime.now(timezone.utc).isoformat()
        result = Result(
            recipe=recipe.name,
            started_at=started,
            finished_at=finished,
            env=env,  # type: ignore[arg-type]
            latency_ms=LatencyStats(p50=float("nan"), p95=float("nan"), p99=float("nan")),
            throughput_fps=ThroughputStats(),
            peak_gpu_mem_mb=None,
            cold_start_ms=None,
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
    peak_mem = max((v.get("peak_gpu_mem_mb") or 0.0) for v in per_bs.values()) or None

    # Accuracy: ultralytics can load .engine directly via model = YOLO('x.engine')
    acc = AccuracyStats()
    if os.environ.get("OMNI_SKIP_ACCURACY"):
        print("[info] OMNI_SKIP_ACCURACY set — skipping mAP eval", file=sys.stderr)
    else:
        try:
            from ultralytics import YOLO

            # Prefer the bs=1 engine we just built; fall back to any engine
            # from this run. This keeps the mAP eval at batch=1 (the setup
            # ultralytics + the engine's optimization profile both expect).
            eng_for_val = bs1_engine
            if eng_for_val is None or not eng_for_val.exists():
                matches = sorted(engine_cache.glob(f"*{dtype}{source_suffix}_bs1_{version_tag}.engine"))
                if not matches:
                    raise RuntimeError("no bs=1 engine available for mAP eval")
                eng_for_val = matches[-1]
            m = YOLO(str(eng_for_val))
            data_yaml = _split.eval_yaml(
                os.environ.get("OMNI_COCO_YAML", "coco.yaml"),
                calib_yaml_path=_split.calib_yaml(),
                calib_seed=recipe.technique.calibration_seed or 42,
                calib_n=recipe.technique.calibration_samples or 512,
            )
            metrics = m.val(data=data_yaml, imgsz=imgsz, batch=1, device=0,
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
        env=env,  # type: ignore[arg-type]
        model_size_mb=None,
        latency_ms=LatencyStats(**{k: v for k, v in lat.items() if k in {"p50", "p95", "p99", "stddev_ms"}}),
        throughput_fps=throughput,
        peak_gpu_mem_mb=peak_mem,
        cold_start_ms=cold_start_ms,
        accuracy=acc,
        build_time_s=build_time_s_bs1,
        build_ceiling_breached=build_ceiling_breached,
        meets_constraints=meets,
        notes="; ".join(note_parts) or None,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json.loads(result.model_dump_json()), f, indent=2)
    print(f"wrote {out_path}")
    _MAIN_TRAINED_YOLO = None  # reset to prevent state leak between test runs
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    return run(args.recipe, args.out)


if __name__ == "__main__":
    sys.exit(main())
