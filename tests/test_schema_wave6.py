"""Wave 6 schema tests: CPU backends (ort_cpu, openvino), bf16 dtype, CPU
hardware fields, thread_count measurement field, CPU env info."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from pydantic import ValidationError

from scripts._schemas import (
    EnvInfo,
    HardwareSpec,
    MeasurementSpec,
    RuntimeSpec,
    TechniqueSpec,
)


# ---------------------------------------------------------------------------
# TechniqueSpec.source — Wave 6 adds ort_cpu and openvino
# ---------------------------------------------------------------------------

def test_technique_source_accepts_ort_cpu():
    t = TechniqueSpec(name="fp32", source="ort_cpu")
    assert t.source == "ort_cpu"


def test_technique_source_accepts_openvino():
    t = TechniqueSpec(name="fp32", source="openvino")
    assert t.source == "openvino"


def test_technique_source_rejects_unknown():
    with pytest.raises(ValidationError):
        TechniqueSpec(name="x", source="does_not_exist")


def test_technique_source_backward_compat_existing_sources():
    for s in ("trt_builtin", "modelopt", "ort_quant"):
        t = TechniqueSpec(name="x", source=s)
        assert t.source == s


# ---------------------------------------------------------------------------
# RuntimeSpec.dtype — Wave 6 adds bf16 (for #31 ort_cpu_bf16)
# ---------------------------------------------------------------------------

def test_runtime_dtype_accepts_bf16():
    r = RuntimeSpec(engine="onnxruntime", dtype="bf16")
    assert r.dtype == "bf16"


def test_runtime_dtype_backward_compat_existing_dtypes():
    for d in ("fp32", "fp16", "int8"):
        r = RuntimeSpec(engine="tensorrt", dtype=d)
        assert r.dtype == d


def test_runtime_dtype_rejects_unknown():
    with pytest.raises(ValidationError):
        RuntimeSpec(engine="tensorrt", dtype="int4")


# ---------------------------------------------------------------------------
# RuntimeSpec.engine — Wave 6 adds openvino
# ---------------------------------------------------------------------------

def test_runtime_engine_accepts_openvino():
    r = RuntimeSpec(engine="openvino", dtype="int8")
    assert r.engine == "openvino"


def test_runtime_engine_backward_compat_existing_engines():
    for e in ("pytorch", "onnxruntime", "tensorrt"):
        r = RuntimeSpec(engine=e, dtype="fp32")
        assert r.engine == e


# ---------------------------------------------------------------------------
# HardwareSpec — Wave 6 adds CPU fields
# ---------------------------------------------------------------------------

def test_hardware_spec_accepts_cpu_fields():
    h = HardwareSpec(
        cpu_model="Intel Xeon Platinum 8480+",
        cpu_cores_physical=56,
        cpu_flags=["avx2", "avx512f", "avx512_vnni", "avx512_bf16", "amx_tile"],
        numa_node=0,
        governor="performance",
    )
    assert h.cpu_model == "Intel Xeon Platinum 8480+"
    assert h.cpu_cores_physical == 56
    assert "avx512_vnni" in h.cpu_flags
    assert h.numa_node == 0
    assert h.governor == "performance"


def test_hardware_spec_cpu_fields_default_none():
    h = HardwareSpec()
    assert h.cpu_model is None
    assert h.cpu_cores_physical is None
    assert h.cpu_flags is None
    assert h.numa_node is None
    assert h.governor is None


def test_hardware_spec_gpu_fields_still_work():
    """Backward-compat: GPU recipes should validate unchanged."""
    h = HardwareSpec(
        gpu="NVIDIA GeForce RTX 3060",
        cuda="12.9",
        driver="576.80",
        requires_compute_capability_min=8.0,
    )
    assert h.gpu == "NVIDIA GeForce RTX 3060"
    assert h.requires_compute_capability_min == 8.0


# ---------------------------------------------------------------------------
# MeasurementSpec.thread_count — Wave 6 adds
# ---------------------------------------------------------------------------

def _base_measurement(**kw) -> dict:
    base = dict(
        dataset="coco_val2017",
        num_images=500,
        warmup_iters=200,
        measure_iters=300,
        batch_sizes=[1, 8],
    )
    base.update(kw)
    return base


def test_measurement_thread_count_defaults_to_none():
    m = MeasurementSpec(**_base_measurement())
    assert m.thread_count is None


def test_measurement_thread_count_accepts_positive_int():
    m = MeasurementSpec(**_base_measurement(thread_count=8))
    assert m.thread_count == 8


def test_measurement_thread_count_rejects_zero():
    """thread_count=0 is ambiguous (ORT treats 0 as 'use all logical cores'
    which drags hyperthreads and typically hurts perf). Force explicit positive
    int or None."""
    with pytest.raises(ValidationError):
        MeasurementSpec(**_base_measurement(thread_count=0))


def test_measurement_thread_count_rejects_negative():
    with pytest.raises(ValidationError):
        MeasurementSpec(**_base_measurement(thread_count=-1))


# ---------------------------------------------------------------------------
# EnvInfo — Wave 6 adds CPU fields
# ---------------------------------------------------------------------------

def test_env_info_accepts_cpu_fields():
    e = EnvInfo(
        cpu_model="Intel Core i7-11375H",
        cpu_cores_physical=4,
        cpu_flags=["avx2", "avx512f", "avx512_vnni"],
    )
    assert e.cpu_model == "Intel Core i7-11375H"
    assert e.cpu_cores_physical == 4
    assert "avx512_vnni" in e.cpu_flags


def test_env_info_cpu_fields_default_none():
    e = EnvInfo()
    assert e.cpu_model is None
    assert e.cpu_cores_physical is None
    assert e.cpu_flags is None


def test_env_info_gpu_fields_still_work():
    """Backward-compat: GPU Result.env parses unchanged."""
    e = EnvInfo(
        gpu="NVIDIA GeForce RTX 3060 Laptop GPU",
        cuda="12.9",
        tensorrt="10.16.0.72",
    )
    assert e.gpu.startswith("NVIDIA")
    assert e.cuda == "12.9"
