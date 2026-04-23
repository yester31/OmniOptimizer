"""Audit YOLO26n + runtime capabilities into results/_capabilities.json.

Wave 16 prereq (TODOS T1). Records four facts the Wave 16 plan's scope
decisions depend on:

    1. YOLO26n ONNX contains MatMul / MHA-like subgraphs
    2. Host CPU DL ISA flags (avx512_bf16, amx_tile, ...)
    3. modelopt.onnx default op_types_to_quantize allowlist
    4. ORT GraphOptimizationLevel ALL vs EXTENDED delta

Re-runnable so model/wheel upgrades regenerate the snapshot; inline grep is
not reproducible three months later.

Run:
    python scripts/audit_capabilities.py
Output:
    results/_capabilities.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.env_lock import _collect_cpu_info  # noqa: E402

ONNX_PATH = ROOT / "results" / "_onnx" / "best_qr_640_fp32_bs1.onnx"
OUTPUT_PATH = ROOT / "results" / "_capabilities.json"


def _check_yolo26n_mha(onnx_path: Path) -> dict:
    """Probe YOLO26n for MatMul nodes whose output feeds a Softmax.

    That pattern is the MHA attention signature. Wave 16 uses this to decide
    whether a `disable_mha_qdq` schema field is needed — if the count is zero,
    the field is dead weight.
    """
    if not onnx_path.exists():
        return {"ok": False, "reason": f"{onnx_path} not found — export FP32 ONNX first"}
    try:
        import onnx
    except ImportError as e:
        return {"ok": False, "reason": f"onnx import failed: {e}"}
    model = onnx.load(str(onnx_path))
    nodes = list(model.graph.node)
    matmul = [n for n in nodes if n.op_type == "MatMul"]
    softmax = [n for n in nodes if n.op_type == "Softmax"]
    mm_outputs = {o for n in matmul for o in n.output}
    sm_inputs = {i for n in softmax for i in n.input}
    mm_feeds_sm = len(mm_outputs & sm_inputs)
    return {
        "ok": True,
        "matmul_count": len(matmul),
        "softmax_count": len(softmax),
        "matmul_directly_feeds_softmax_count": mm_feeds_sm,
        "has_mha_pattern": mm_feeds_sm > 0,
        "total_nodes": len(nodes),
    }


def _check_cpu_flags() -> dict:
    """Reuse env_lock's CPU probe, trimmed to Wave 16 decision points."""
    info = _collect_cpu_info()
    flags = set(info.get("cpu_flags") or [])
    return {
        "cpu_model": info.get("cpu_model"),
        "flags": sorted(flags),
        "has_avx512_bf16": "avx512_bf16" in flags,
        "has_avx512_vnni": "avx512_vnni" in flags,
        "has_amx_tile": "amx_tile" in flags,
        "has_amx_bf16": "amx_bf16" in flags,
        "has_amx_int8": "amx_int8" in flags,
    }


def _check_modelopt_op_allowlist() -> dict:
    """Default op_types_to_quantize from modelopt's ORT integration layer.

    Recipes #09/#12/#42 pass neither op_types_to_quantize nor nodes_to_quantize,
    so modelopt falls back to this allowlist. Wave 16 checks whether MatMul /
    Attention ops are in the default set to size the IgnoredScope work.
    """
    try:
        from modelopt.onnx.quantization.ort_utils import get_quantizable_op_types
    except ImportError as e:
        return {"ok": False, "reason": f"modelopt import failed: {e}"}
    ops = list(get_quantizable_op_types([]))
    return {"ok": True, "default_allowlist": sorted(ops), "count": len(ops)}


def _check_ort_opt_level_delta() -> dict:
    """Record the ALL-vs-EXTENDED delta plus ORT version.

    ORT registers graph transformers in C++ — not programmatically enumerable
    from Python. Source of truth:
    ``onnxruntime/core/optimizer/graph_transformer_utils.cc::GenerateTransformers``.
    ALL (level 99) beyond EXTENDED (level 2) adds NCHWc layout rewrites and
    platform-specific MLAS fusion passes. On hardware without matching kernels
    the passes run but may not pay off.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        return {"ok": False, "reason": f"onnxruntime import failed: {e}"}
    return {
        "ok": True,
        "onnxruntime_version": ort.__version__,
        "all_minus_extended": [
            "NCHWc layout rewrites (x86 kernel format)",
            "MLAS-specific fusion passes",
        ],
        "source_reference": (
            "onnxruntime/core/optimizer/graph_transformer_utils.cc"
            "::GenerateTransformers"
        ),
    }


def main() -> int:
    caps = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/audit_capabilities.py",
        "yolo26n_onnx": _check_yolo26n_mha(ONNX_PATH),
        "host_cpu": _check_cpu_flags(),
        "modelopt_onnx_int8": _check_modelopt_op_allowlist(),
        "ort_graph_opt_levels": _check_ort_opt_level_delta(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(caps, indent=2) + "\n", encoding="utf-8")
    try:
        display = OUTPUT_PATH.relative_to(ROOT)
    except ValueError:
        display = OUTPUT_PATH
    print(f"[ok] wrote {display}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
