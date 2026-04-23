"""Tests for scripts/audit_capabilities.py — Wave 16 prereq (TODOS T1).

Keeps the audit honest:
- _check_yolo26n_mha must classify a synthetic MatMul→Softmax graph as MHA.
- _check_yolo26n_mha must return ok=False (never raise) when the ONNX is missing.
- _check_cpu_flags must expose the boolean convenience fields Wave 16 reads.
- main() must produce a well-formed JSON file with all four sections.

No network, no GPU, no external downloads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_matmul_softmax_onnx(path: Path) -> None:
    """Synthesize the minimal MatMul→Softmax graph that _check_yolo26n_mha
    should recognize as an MHA signature."""
    import onnx
    from onnx import TensorProto, helper

    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 4, 8])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 8, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 4])
    matmul = helper.make_node("MatMul", ["a", "b"], ["mm_out"], name="mm")
    softmax = helper.make_node("Softmax", ["mm_out"], ["out"], axis=-1, name="sm")
    graph = helper.make_graph([matmul, softmax], "mha_min", [a, b], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def test_mha_detection_positive(tmp_path):
    from scripts.audit_capabilities import _check_yolo26n_mha

    onnx_path = tmp_path / "mha.onnx"
    _make_matmul_softmax_onnx(onnx_path)
    result = _check_yolo26n_mha(onnx_path)
    assert result["ok"] is True
    assert result["matmul_count"] == 1
    assert result["softmax_count"] == 1
    assert result["matmul_directly_feeds_softmax_count"] == 1
    assert result["has_mha_pattern"] is True


def test_mha_detection_missing_file_is_ok_false(tmp_path):
    from scripts.audit_capabilities import _check_yolo26n_mha

    result = _check_yolo26n_mha(tmp_path / "does_not_exist.onnx")
    assert result["ok"] is False
    assert "not found" in result["reason"]


def test_cpu_flags_shape():
    from scripts.audit_capabilities import _check_cpu_flags

    result = _check_cpu_flags()
    for key in (
        "cpu_model", "flags",
        "has_avx512_bf16", "has_avx512_vnni",
        "has_amx_tile", "has_amx_bf16", "has_amx_int8",
    ):
        assert key in result, f"missing key: {key}"
    assert isinstance(result["flags"], list)
    for bool_key in (
        "has_avx512_bf16", "has_avx512_vnni",
        "has_amx_tile", "has_amx_bf16", "has_amx_int8",
    ):
        assert isinstance(result[bool_key], bool), f"{bool_key} must be bool"


def test_main_writes_capabilities_json(tmp_path, monkeypatch):
    """End-to-end smoke: main() produces the expected JSON structure."""
    from scripts import audit_capabilities as mod

    # Redirect output to tmp so we don't clobber the real artifact.
    monkeypatch.setattr(mod, "OUTPUT_PATH", tmp_path / "_capabilities.json")
    # Force ONNX path to a non-existent file so MHA check returns ok=False
    # (we don't require the real YOLO26n ONNX for the smoke test).
    monkeypatch.setattr(mod, "ONNX_PATH", tmp_path / "missing.onnx")

    rc = mod.main()
    assert rc == 0

    data = json.loads((tmp_path / "_capabilities.json").read_text(encoding="utf-8"))
    for section in ("yolo26n_onnx", "host_cpu", "modelopt_onnx_int8", "ort_graph_opt_levels"):
        assert section in data, f"missing section: {section}"
    assert data["generated_by"] == "scripts/audit_capabilities.py"
