"""Tests for Wave 6 Task 4 — ORT CPU EP static / dynamic INT8 paths."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


def _make_int8_recipe(calibrator=None, calibration_samples=None):
    from scripts._schemas import Recipe
    technique = {"name": "int8", "source": "ort_cpu"}
    if calibrator is not None:
        technique["calibrator"] = calibrator
    if calibration_samples is not None:
        technique["calibration_samples"] = calibration_samples
    return Recipe.model_validate({
        "name": "test_int8",
        "model": {"family": "yolo26", "variant": "n", "weights": "yolo26n.pt"},
        "runtime": {"engine": "onnxruntime", "dtype": "int8"},
        "technique": technique,
        "measurement": {
            "dataset": "coco_val2017",
            "num_images": 10,
            "warmup_iters": 2,
            "measure_iters": 2,
            "batch_sizes": [1],
        },
    })


def test_dispatch_int8_dynamic_routes_to_dynamic_handler(monkeypatch):
    """calibrator=None with dtype=int8 → _prepare_ort_cpu_int8_dynamic."""
    from scripts import run_cpu

    called = {"path": None}

    def fake_dynamic(recipe_, so_factory):
        called["path"] = "dynamic"
        return "stub"

    monkeypatch.setattr(run_cpu, "_prepare_ort_cpu_int8_dynamic", fake_dynamic)
    recipe = _make_int8_recipe(calibrator=None)
    run_cpu._prepare_cpu_session(recipe)
    assert called["path"] == "dynamic"


def test_dispatch_int8_static_routes_to_static_handler(monkeypatch):
    """calibrator=entropy with dtype=int8 → _prepare_ort_cpu_int8_static."""
    from scripts import run_cpu

    called = {"path": None}

    def fake_static(recipe_, so_factory):
        called["path"] = "static"
        return "stub"

    monkeypatch.setattr(run_cpu, "_prepare_ort_cpu_int8_static", fake_static)
    recipe = _make_int8_recipe(calibrator="entropy", calibration_samples=32)
    run_cpu._prepare_cpu_session(recipe)
    assert called["path"] == "static"


def test_calib_loader_in_weights_io_module():
    """_build_calib_numpy extracted from run_trt.py lives in _weights_io
    so CPU runner doesn't import run_trt (and thus avoids TRT/CUDA)."""
    from scripts._weights_io import _build_calib_numpy
    assert callable(_build_calib_numpy)


def test_letterbox_in_weights_io_module():
    from scripts._weights_io import _letterbox
    assert callable(_letterbox)


def test_weights_io_still_trt_free_after_calib_extract():
    """The calib / letterbox additions must not change the invariant."""
    pre_trt = "tensorrt" in sys.modules
    pre_pycuda = "pycuda" in sys.modules
    import scripts._weights_io  # noqa: F401
    assert ("tensorrt" in sys.modules) == pre_trt
    assert ("pycuda" in sys.modules) == pre_pycuda


def test_build_calib_numpy_array_wrapper_takes_recipe():
    """The run_cpu wrapper derives val_yaml / n_samples / imgsz / seed from
    Recipe + env vars (OMNI_CALIB_YAML fallback to OMNI_COCO_YAML)."""
    from scripts import run_cpu

    assert callable(run_cpu._build_calib_numpy_array)


def test_build_calib_numpy_array_uses_random_fallback(monkeypatch):
    """When OMNI_ALLOW_RANDOM_CALIB=1 and no real yaml is available, the
    wrapper should return a numpy array of the requested shape."""
    import numpy as np

    from scripts import run_cpu

    monkeypatch.delenv("OMNI_CALIB_YAML", raising=False)
    monkeypatch.delenv("OMNI_COCO_YAML", raising=False)
    monkeypatch.setenv("OMNI_ALLOW_RANDOM_CALIB", "1")
    recipe = _make_int8_recipe(calibrator="entropy", calibration_samples=8)
    arr = run_cpu._build_calib_numpy_array(recipe)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.shape == (8, 3, 640, 640)


def test_build_calib_numpy_array_missing_yaml_raises(monkeypatch):
    """Without OMNI_ALLOW_RANDOM_CALIB and no valid yaml, the wrapper must
    raise — silent random calibration would tank mAP (Wave 3 lesson)."""
    from scripts import run_cpu

    monkeypatch.delenv("OMNI_CALIB_YAML", raising=False)
    monkeypatch.delenv("OMNI_COCO_YAML", raising=False)
    monkeypatch.delenv("OMNI_ALLOW_RANDOM_CALIB", raising=False)
    recipe = _make_int8_recipe(calibrator="entropy", calibration_samples=8)
    with pytest.raises(RuntimeError, match="Calibration data unavailable"):
        run_cpu._build_calib_numpy_array(recipe)
