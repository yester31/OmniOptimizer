"""Tests for scripts/run_cpu.py — Wave 6 CPU inference runner skeleton.

Covers FP32 dispatch + module-level import hygiene. INT8 / BF16 / OpenVINO
paths are intentionally NotImplementedError at this stage — those are
Task 4 / Task 5."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


def _make_recipe(source: str = "ort_cpu", dtype: str = "fp32",
                 engine: str = "onnxruntime"):
    from scripts._schemas import Recipe
    return Recipe.model_validate({
        "name": "test_cpu_recipe",
        "model": {"family": "yolo26", "variant": "n", "weights": "yolo26n.pt"},
        "runtime": {"engine": engine, "dtype": dtype},
        "technique": {"name": dtype, "source": source},
        "measurement": {
            "dataset": "coco_val2017",
            "num_images": 10,
            "warmup_iters": 2,
            "measure_iters": 2,
            "batch_sizes": [1],
        },
    })


def test_run_cpu_imports_without_tensorrt():
    """Importing scripts.run_cpu must not pull tensorrt/pycuda. CPU-only
    environments need this to be safe."""
    pre_trt = "tensorrt" in sys.modules
    pre_pycuda = "pycuda" in sys.modules
    import scripts.run_cpu  # noqa: F401
    assert ("tensorrt" in sys.modules) == pre_trt, \
        "scripts.run_cpu must not import tensorrt at module load"
    assert ("pycuda" in sys.modules) == pre_pycuda, \
        "scripts.run_cpu must not import pycuda at module load"


def test_run_cpu_cli_surface():
    """Expose main() and run() at module level with the expected signatures."""
    from scripts import run_cpu
    assert callable(run_cpu.main)
    assert callable(run_cpu.run)


def test_dispatch_ort_cpu_fp32_creates_ort_session(tmp_path, monkeypatch):
    """For source=ort_cpu + dtype=fp32, _prepare_cpu_session must return a
    callable that invokes onnxruntime. We don't execute the forward here —
    just verify the dispatch picks the right branch."""
    from scripts import run_cpu

    recipe = _make_recipe(source="ort_cpu", dtype="fp32")
    # Stub the ONNX export so the test doesn't require ultralytics + a real
    # .pt file. The stub returns a path that doesn't exist; the dispatcher
    # must only branch on (source, dtype), not try to open the file yet.
    called = {"branch": None}

    def fake_prepare(recipe_, session_options_factory):
        called["branch"] = (recipe_.technique.source, recipe_.runtime.dtype)
        return "stub_session"

    monkeypatch.setattr(run_cpu, "_prepare_ort_cpu_fp32", fake_prepare)
    session = run_cpu._prepare_cpu_session(recipe)
    assert called["branch"] == ("ort_cpu", "fp32")
    assert session == "stub_session"


def test_dispatch_openvino_raises_not_implemented():
    """OpenVINO path is Task 5 territory."""
    from scripts import run_cpu

    recipe = _make_recipe(source="openvino", dtype="fp32", engine="openvino")
    with pytest.raises(NotImplementedError, match="Task 5"):
        run_cpu._prepare_cpu_session(recipe)


def test_dispatch_bf16_skips_when_cpu_lacks_bf16_and_amx(monkeypatch):
    """BF16 requires AMX (SPR+) or AVX-512 BF16 (Cooper Lake+). Without
    either, dispatcher must raise NotImplementedError with a message
    that run_cpu.run() will surface as 'meets_constraints=False' + a
    hardware-gate note — not a silent fall-through."""
    from scripts import run_cpu

    monkeypatch.setattr(run_cpu, "_collect_cpu_info",
                        lambda: {"cpu_flags": ["avx2", "avx512f"]})
    recipe = _make_recipe(source="ort_cpu", dtype="bf16")
    with pytest.raises(NotImplementedError, match="lacks BF16"):
        run_cpu._prepare_cpu_session(recipe)


def test_dispatch_bf16_raises_unimpl_when_hardware_capable(monkeypatch):
    """When the host CPU reports AMX or AVX-512 BF16, the hardware gate
    passes but BF16 inference on ORT CPU EP still isn't wired up (the
    actual float→bfloat16 model conversion is deferred). The dispatcher
    must raise a distinct NotImplementedError so it's clear the gate
    isn't the blocker — the backend work is."""
    from scripts import run_cpu

    monkeypatch.setattr(run_cpu, "_collect_cpu_info",
                        lambda: {"cpu_flags": ["avx2", "avx512f", "amx_tile"]})
    recipe = _make_recipe(source="ort_cpu", dtype="bf16")
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        run_cpu._prepare_cpu_session(recipe)


def test_dispatch_bf16_cpu_flags_none_treated_as_no_support(monkeypatch):
    """cpu_flags=None (detection failed or unavailable) is treated as
    'unknown capability → assume missing', so the recipe still records
    'BF16 skipped' rather than silently trying to build an unsupported
    session."""
    from scripts import run_cpu

    monkeypatch.setattr(run_cpu, "_collect_cpu_info",
                        lambda: {"cpu_flags": None})
    recipe = _make_recipe(source="ort_cpu", dtype="bf16")
    with pytest.raises(NotImplementedError, match="lacks BF16"):
        run_cpu._prepare_cpu_session(recipe)


def test_dispatch_unknown_source_raises():
    """Non-CPU sources routed to run_cpu should fail loudly, not run silently."""
    from scripts import run_cpu

    # brevitas is a GPU-only source — run_cpu should not accept it.
    recipe = _make_recipe(source="brevitas", dtype="int8", engine="tensorrt")
    with pytest.raises((NotImplementedError, ValueError, RuntimeError)):
        run_cpu._prepare_cpu_session(recipe)


def test_thread_count_auto_detect_returns_positive_int():
    """When measurement.thread_count is None, _resolve_thread_count must
    return a positive int matching physical cores (via psutil) or a
    reasonable fallback."""
    from scripts import run_cpu

    recipe = _make_recipe()
    assert recipe.measurement.thread_count is None
    n = run_cpu._resolve_thread_count(recipe)
    assert isinstance(n, int)
    assert n > 0


def test_thread_count_honors_explicit_value():
    """When measurement.thread_count is set to an integer, that value is
    used verbatim (experiment knob for thread-scaling studies)."""
    from scripts import run_cpu
    from scripts._schemas import Recipe

    recipe = Recipe.model_validate({
        "name": "test",
        "model": {"family": "yolo26", "variant": "n", "weights": "yolo26n.pt"},
        "runtime": {"engine": "onnxruntime", "dtype": "fp32"},
        "technique": {"name": "fp32", "source": "ort_cpu"},
        "measurement": {
            "dataset": "coco_val2017",
            "num_images": 10,
            "warmup_iters": 2,
            "measure_iters": 2,
            "batch_sizes": [1],
            "thread_count": 3,
        },
    })
    assert run_cpu._resolve_thread_count(recipe) == 3
