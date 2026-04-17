"""Phase 3 runner tests: 2:4 sparsity preprocess dispatcher + nodes_to_exclude.

Two unit tests use monkeypatch on ``_apply_modelopt_sparsify`` itself so we
don't pull modelopt into the unit-test path. A third integration test only
runs when modelopt is actually installed (otherwise skipped) and confirms
our sparsify call signature matches the installed version.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_recipe(tmp_path, **technique_kwargs):
    """Build a minimal Recipe object pointed at a real weights stub."""
    import yaml

    from scripts._schemas import load_recipe

    weights = tmp_path / "yolo26n.pt"
    weights.write_bytes(b"stub")

    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump({
        "name": "phase3_test",
        "model": {"family": "yolo26", "variant": "n", "weights": str(weights)},
        "runtime": {"engine": "tensorrt", "dtype": "int8"},
        "technique": {
            "name": "int8_ptq",
            "source": "modelopt",
            "calibrator": "entropy",
            "calibration_samples": 8,
            "calibration_seed": 42,
            **technique_kwargs,
        },
        "measurement": {
            "dataset": "coco_val2017",
            "num_images": 1,
            "warmup_iters": 1,
            "measure_iters": 1,
            "batch_sizes": [1],
            "input_size": 640,
            "gpu_clock_lock": False,
            "seed": 42,
        },
    }))
    return load_recipe(str(recipe_path))


def test_dispatcher_invokes_sparsify_helper_before_quantize(tmp_path, monkeypatch):
    """Dispatcher: when sparsity_preprocess='2:4', _apply_modelopt_sparsify
    runs before the ONNX quantize step. Does not assert on modelopt internals."""
    from scripts import run_trt

    calls = []

    class FakeYolo:
        ckpt_path = "yolo26n.pt"

        def export(self, **kwargs):
            out = tmp_path / "exported.onnx"
            out.write_bytes(b"onnx-stub")
            return str(out)

    def fake_apply_sparsify(weights, imgsz):
        calls.append("sparsify")
        return FakeYolo()

    def fake_quantize(onnx_path, output_path, **kwargs):
        calls.append("quantize")
        Path(output_path).write_bytes(b"qdq-stub")

    monkeypatch.setattr(run_trt, "_apply_modelopt_sparsify", fake_apply_sparsify)
    monkeypatch.setattr(
        run_trt,
        "_build_calib_numpy",
        lambda *a, **k: __import__("numpy").zeros((1, 3, 640, 640), dtype="float32"),
    )
    fake_moq_onnx = types.ModuleType("modelopt.onnx.quantization")
    fake_moq_onnx.quantize = fake_quantize
    monkeypatch.setitem(sys.modules, "modelopt.onnx.quantization", fake_moq_onnx)

    recipe = _make_recipe(tmp_path, sparsity_preprocess="2:4")
    out = run_trt._prepare_modelopt_onnx(
        recipe, imgsz=640, cache_dir=tmp_path / "onnx", dynamic=True,
    )
    assert out.exists()
    assert calls == ["sparsify", "quantize"]


def test_dispatcher_skips_sparsify_when_preprocess_none(tmp_path, monkeypatch):
    """Default path: no sparsify call."""
    from scripts import run_trt

    calls = []

    def fake_apply_sparsify(*a, **k):
        calls.append("sparsify")

    def fake_export(weights, imgsz, **k):
        out = tmp_path / f"{Path(str(weights)).stem}.onnx"
        out.write_bytes(b"onnx-stub")
        return out

    def fake_quantize(onnx_path, output_path, **kwargs):
        calls.append("quantize")
        Path(output_path).write_bytes(b"qdq-stub")

    monkeypatch.setattr(run_trt, "_apply_modelopt_sparsify", fake_apply_sparsify)
    monkeypatch.setattr(run_trt, "_export_onnx", fake_export)
    monkeypatch.setattr(
        run_trt,
        "_build_calib_numpy",
        lambda *a, **k: __import__("numpy").zeros((1, 3, 640, 640), dtype="float32"),
    )
    fake_moq_onnx = types.ModuleType("modelopt.onnx.quantization")
    fake_moq_onnx.quantize = fake_quantize
    monkeypatch.setitem(sys.modules, "modelopt.onnx.quantization", fake_moq_onnx)

    recipe = _make_recipe(tmp_path)
    run_trt._prepare_modelopt_onnx(
        recipe, imgsz=640, cache_dir=tmp_path / "onnx", dynamic=True,
    )
    assert calls == ["quantize"]


@pytest.mark.skipif(
    importlib.util.find_spec("modelopt") is None
    or importlib.util.find_spec("modelopt.torch.sparsity") is None,
    reason="modelopt not installed",
)
def test_real_modelopt_sparsify_api_signature(tmp_path):
    """Integration test: hit real modelopt to confirm our sparsify call
    signature matches the installed version. Runs only when modelopt is
    present so CI without GPU/modelopt still passes the rest of the suite."""
    pytest.importorskip("ultralytics")
    from scripts.run_trt import _apply_modelopt_sparsify

    # This test only verifies that the call does not raise; it does not
    # check the pruning pattern (that is what Task 7 verification is for).
    yolo = _apply_modelopt_sparsify("yolo26n.pt", 640)
    assert hasattr(yolo, "model")
