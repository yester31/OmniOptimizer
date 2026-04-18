"""Wave 3 schema + dispatch tests for ort_quant and neural_compressor.

Scope:
- TechniqueSpec.source Literal accepts the two new backend names.
- QAT-only fields (qat_epochs, qat_lr) are defined explicitly on the schema
  so `#19 inc_int8_qat` doesn't silently lose its training hyperparameters.
- _prepare_onnx dispatcher routes ort_quant / neural_compressor to the right
  helper (mocked — actual quantization is exercised in end-to-end recipe runs).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._schemas import TechniqueSpec, load_recipe  # noqa: E402


@pytest.mark.parametrize("src", ["trt_builtin", "modelopt", "ort_quant", "neural_compressor"])
def test_source_literal_accepts_all_backends(src):
    spec = TechniqueSpec(name="int8_ptq", source=src)
    assert spec.source == src


def test_source_literal_rejects_unknown_backend():
    with pytest.raises(Exception):
        TechniqueSpec(name="int8_ptq", source="bogus_backend")


def test_qat_fields_default_none():
    spec = TechniqueSpec(name="int8_qat", source="neural_compressor")
    assert spec.qat_epochs is None
    assert spec.qat_lr is None


def test_qat_fields_parse_explicit_values():
    spec = TechniqueSpec(
        name="int8_qat", source="neural_compressor", qat_epochs=3, qat_lr=1.0e-5
    )
    assert spec.qat_epochs == 3
    assert spec.qat_lr == pytest.approx(1.0e-5)


def _write_min_recipe(tmp_path: Path, source: str, calibrator: str) -> Path:
    p = tmp_path / f"{source}.yaml"
    p.write_text(
        "name: smoke\n"
        "model: {family: yolo26, variant: n, weights: yolo26n.pt}\n"
        "runtime: {engine: tensorrt, dtype: int8}\n"
        f"technique: {{name: int8_ptq, source: {source},\n"
        f"            calibrator: {calibrator}, calibration_samples: 8,\n"
        f"            calibration_seed: 42}}\n"
        "measurement:\n"
        "  dataset: coco_val2017\n  num_images: 8\n"
        "  warmup_iters: 1\n  measure_iters: 2\n"
        "  batch_sizes: [1]\n  input_size: 640\n"
        "  gpu_clock_lock: false\n  seed: 42\n"
        "constraints: {max_map_drop_pct: 5.0}\n",
        encoding="utf-8",
    )
    return p


def test_prepare_onnx_routes_to_ort_quant(tmp_path):
    from scripts.run_trt import _prepare_onnx

    recipe = load_recipe(str(_write_min_recipe(tmp_path, "ort_quant", "minmax")))
    with patch("scripts.run_trt._prepare_ort_quant_onnx") as m:
        m.return_value = tmp_path / "fake.onnx"
        path, is_qdq = _prepare_onnx(recipe, 640, tmp_path, bs=1)
    assert is_qdq is True
    assert path == tmp_path / "fake.onnx"
    m.assert_called_once()


def test_source_tag_windows_path_friendly():
    """Engine cache filenames must stay short enough on Windows (MAX_PATH=260).
    Long source names (e.g. 'neural_compressor') collapse to '_inc' via the
    shared _SOURCE_TAG mapping.
    """
    from scripts.run_trt import _SOURCE_TAG

    assert _SOURCE_TAG["trt_builtin"] == ""
    assert _SOURCE_TAG["modelopt"] == "_modelopt"
    assert _SOURCE_TAG["ort_quant"] == "_ort"
    assert _SOURCE_TAG["neural_compressor"] == "_inc"


def test_ort_quant_rejects_unknown_calibrator(tmp_path):
    """Guardrail: unknown calibrator strings should fail fast rather than
    silently dispatching to CalibrationMethod.MinMax."""
    from scripts.run_trt import _prepare_ort_quant_onnx

    recipe = load_recipe(
        str(_write_min_recipe(tmp_path, "ort_quant", "bogus_method"))
    )
    with pytest.raises(ValueError, match="calibrator"):
        _prepare_ort_quant_onnx(recipe, 640, tmp_path, dynamic=False)
