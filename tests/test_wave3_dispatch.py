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


# NOTE: _prepare_onnx dispatch tests for ort_quant and neural_compressor are
# added in Task 2 and Task 4 of the Wave 3 plan, at the same time the runner
# gains _prepare_ort_quant_onnx and _prepare_inc_onnx. Keeping this file to
# schema-only for Task 1 preserves a green test suite during each landing.
