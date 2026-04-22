"""Recipe smoke tests (Wave 11 Task 6).

Regression guard for recipes that previously broke silently. Each test reads
a Result JSON produced by the runner and asserts a minimum perf / accuracy
floor. Tests skip gracefully when the JSON is absent (fresh clone, CI without
GPU) so they are safe to run anywhere.

Targets (post-archive):
- #04 ort_trt_fp16:       fps_bs1 > 150   (ORT-via-TRT ceiling ~211 on 3060 L)
- #08 modelopt_int8_ptq:  fps_bs1 > 300   (variance-tolerant lower bound —
                                            Task 4 B4 analysis documents the
                                            400-760 nondeterminism range)
- #33 ort_cpu_int8_static: mAP_50 > 0.5   (QDQ coverage regression — was 0.0)

Archived (no longer in recipe bank, test absent on purpose):
- #02 torchcompile_fp16 (Wave 11 Task 2)
- #03 ort_cuda_fp16      (Wave 11 Task 1)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts._schemas import Result

ROOT = Path(__file__).resolve().parents[1]
RESULTS_QR = ROOT / "results_qr"
RESULTS_CPU_QR = ROOT / "results_cpu_qr"


def _load(results_dir: Path, filename: str) -> Result:
    p = results_dir / filename
    if not p.exists():
        pytest.skip(f"result JSON not found: {p} (skip on fresh clone / CI)")
    data = json.loads(p.read_text(encoding="utf-8"))
    return Result.model_validate(data)


def test_recipe_04_ort_trt_fp16_fps_floor():
    """ORT-via-TRT wrapper should exceed 150 fps bs1. Historical measurement
    (pre-Wave 11) was 211 fps; Wave 11 Task 3 DLL fix re-measured 188.6.
    Anything below 150 indicates a regression (DLL path broken, TRT EP
    silently fell back to CUDA EP / CPU EP).
    """
    r = _load(RESULTS_QR, "04_ort_trt_fp16.json")
    fps = r.throughput_fps.bs1
    assert fps is not None, "fps_bs1 missing — session prep likely failed"
    assert fps > 150.0, (
        f"#04 ort_trt_fp16 fps regression: {fps:.1f} < 150 "
        f"— check TRT EP DLL path (scripts/run_ort.py::_add_tensorrt_dll_dir)"
    )


def test_recipe_08_modelopt_int8_ptq_fps_floor():
    """modelopt_int8_ptq (max calibrator) has shown fps 430-760 range due to
    TRT builder autotune nondeterminism (see
    docs/improvements/2026-04-23-modelopt-ptq-tactic-analysis.md). 300 is a
    safety floor well below all observed values — anything lower means the
    recipe isn't producing valid INT8 at all.
    """
    r = _load(RESULTS_QR, "08_modelopt_int8_ptq.json")
    fps = r.throughput_fps.bs1
    assert fps is not None
    assert fps > 300.0, (
        f"#08 modelopt_int8_ptq fps regression: {fps:.1f} < 300 "
        f"— INT8 tactic not selected? See Task 4 analysis doc."
    )


def test_recipe_33_ort_cpu_int8_static_map_floor():
    """#33 mAP was 0.0 in Wave 6 ship (Detect head Q/DQ destroyed NMS indices).
    Wave 11 Task 5 excluded Detect head via nodes_to_exclude=[/model.23/]
    pattern; re-measurement produced mAP 0.983 / fps 9.1 (up from 0.0 / 6.2).
    Smoke floor 0.5 catches any future "mAP=0" regression without demanding
    the exact Wave 11 target (varies with dataset/weights).
    """
    r = _load(RESULTS_CPU_QR, "33_ort_cpu_int8_static.json")
    map_50 = r.accuracy.map_50
    assert map_50 is not None, "map_50 missing — accuracy eval failed"
    assert map_50 > 0.5, (
        f"#33 ort_cpu_int8_static mAP regression: {map_50:.3f} <= 0.5 "
        f"— nodes_to_exclude Detect head exclusion broken?"
    )
