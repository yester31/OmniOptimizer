"""Tests for scripts/_weights_io.py — TRT-independent weight/ONNX helpers
extracted from run_trt.py so CPU runners (Wave 6) can import them without
pulling TensorRT or CUDA at module load time."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_weights_io_imports_without_tensorrt():
    """Importing scripts._weights_io must not transitively load tensorrt or
    pycuda. A CPU-only environment without TRT installed must be able to
    import this module cleanly."""
    # Snapshot sys.modules before the import so prior test imports don't mask
    # a leak introduced by _weights_io itself. We only assert that the import
    # didn't ADD tensorrt/pycuda — other tests may have already loaded them,
    # which is fine.
    pre_trt = "tensorrt" in sys.modules
    pre_pycuda = "pycuda" in sys.modules

    import scripts._weights_io  # noqa: F401

    assert ("tensorrt" in sys.modules) == pre_trt, \
        "scripts._weights_io must not import tensorrt at module load time"
    assert ("pycuda" in sys.modules) == pre_pycuda, \
        "scripts._weights_io must not import pycuda at module load time"


def test_weights_io_exports_expected_helpers():
    """The three public helpers must be importable with the same names used
    by the existing run_trt.py call sites."""
    from scripts._weights_io import (
        _export_onnx,
        _load_yolo_for_restore,
        _resolve_weights,
    )
    assert callable(_load_yolo_for_restore)
    assert callable(_resolve_weights)
    assert callable(_export_onnx)


def test_run_trt_reexports_for_backward_compat():
    """run_trt.py must keep exposing the same function names so existing
    callers / tests don't break. Implementation may delegate to _weights_io
    but the attributes should still be accessible on the run_trt module."""
    from scripts import run_trt
    assert callable(run_trt._resolve_weights)
    assert callable(run_trt._export_onnx)
    assert callable(run_trt._load_yolo_for_restore)


def test_resolve_weights_no_training_returns_path_string():
    """Recipes without technique.training should return recipe.model.weights
    verbatim as a string — no trained_weights/ lookup."""
    from scripts._schemas import Recipe
    from scripts._weights_io import _resolve_weights

    recipe = Recipe.model_validate({
        "name": "test_recipe",
        "model": {"family": "yolo26", "variant": "n", "weights": "yolo26n.pt"},
        "runtime": {"engine": "tensorrt", "dtype": "fp32"},
        "technique": {"name": "fp32", "source": "trt_builtin"},
        "measurement": {
            "dataset": "coco_val2017",
            "num_images": 500,
            "warmup_iters": 100,
            "measure_iters": 100,
            "batch_sizes": [1, 8],
        },
    })
    resolved = _resolve_weights(recipe)
    assert resolved == "yolo26n.pt"


def test_resolve_weights_training_but_missing_pt_raises(tmp_path, monkeypatch):
    """If recipe requires training (technique.training set) but
    trained_weights/{name}.pt doesn't exist, _resolve_weights must raise
    RuntimeError with an actionable message (train.py invocation)."""
    import pytest
    from scripts._schemas import Recipe
    from scripts import _weights_io

    # Point ROOT at a fresh tmp dir so trained_weights/ is guaranteed empty.
    monkeypatch.setattr(_weights_io, "ROOT", tmp_path)

    recipe = Recipe.model_validate({
        "name": "ghost_recipe",
        "model": {"family": "yolo26", "variant": "n", "weights": "yolo26n.pt"},
        "runtime": {"engine": "tensorrt", "dtype": "int8"},
        "technique": {
            "name": "training_sparse",
            "source": "modelopt",
            "training": {
                "base_checkpoint": "best_qr.pt",
                "epochs": 30,
                "modifier": "modelopt_sparsify",
            },
        },
        "measurement": {
            "dataset": "coco_val2017",
            "num_images": 500,
            "warmup_iters": 100,
            "measure_iters": 100,
            "batch_sizes": [1, 8],
        },
    })
    with pytest.raises(RuntimeError, match="requires training"):
        _weights_io._resolve_weights(recipe)


def test_export_onnx_caches_existing(tmp_path):
    """If the target .onnx already exists in cache_dir, _export_onnx returns
    it without invoking ultralytics export. This is the fast-path behavior
    the TRT runner relies on."""
    from scripts._weights_io import _export_onnx

    # Create a fake cached ONNX with the exact name _export_onnx expects.
    # Tag format: {stem}_{imgsz}_{fp32|fp16}{tag_suffix}_{bs_tag}.onnx
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stem = "fake_weights"
    imgsz = 640
    cached_name = f"{stem}_{imgsz}_fp32_dyn.onnx"
    cached_path = cache_dir / cached_name
    cached_path.write_bytes(b"onnx stub")

    # Point weights at a fake path with that stem — _export_onnx uses Path.stem
    # so we don't need a real .pt file when cache exists.
    fake_weights = tmp_path / f"{stem}.pt"
    result = _export_onnx(
        weights=str(fake_weights),
        imgsz=imgsz,
        half=False,
        cache_dir=cache_dir,
    )
    assert result == cached_path
    assert result.read_bytes() == b"onnx stub"  # untouched by export
