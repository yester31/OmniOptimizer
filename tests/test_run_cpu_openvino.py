"""Tests for Wave 6 Task 5 — OpenVINO runtime + NNCF INT8 PTQ.

Scope:
- Dispatcher routing for (source=openvino, dtype in {fp32, int8}).
- OVRunnerAsORT adapter: exposes .run / .get_inputs / .get_outputs so the
  existing measure_latency forward closure in run_cpu.run() works unchanged.
- TRT-free invariant: importing the OpenVINO code path does NOT pull in
  tensorrt or pycuda (scripts.run_cpu module-level import hygiene).

Heavy smoke (actual NNCF quantize + IR compile) is deferred to Task 10.
These tests stub the OpenVINO import surface so they run in seconds even
on laptops where the NNCF install is optional.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


def _make_openvino_recipe(dtype="fp32", calibrator=None):
    from scripts._schemas import Recipe
    technique: dict = {"name": dtype, "source": "openvino"}
    if calibrator is not None:
        technique["calibrator"] = calibrator
    return Recipe.model_validate({
        "name": "test_ov",
        "model": {"family": "yolo26", "variant": "n", "weights": "yolo26n.pt"},
        "runtime": {"engine": "openvino", "dtype": dtype},
        "technique": technique,
        "measurement": {
            "dataset": "coco_val2017",
            "num_images": 10,
            "warmup_iters": 2,
            "measure_iters": 2,
            "batch_sizes": [1],
        },
    })


def test_dispatch_openvino_fp32_routes_to_fp32_handler(monkeypatch):
    """source=openvino + dtype=fp32 → _prepare_openvino_fp32."""
    from scripts import run_cpu

    called = {"branch": None}

    def fake_fp32(recipe_):
        called["branch"] = "openvino_fp32"
        return "stub"

    monkeypatch.setattr(run_cpu, "_prepare_openvino_fp32", fake_fp32)
    recipe = _make_openvino_recipe(dtype="fp32")
    run_cpu._prepare_cpu_session(recipe)
    assert called["branch"] == "openvino_fp32"


def test_dispatch_openvino_int8_routes_to_nncf_handler(monkeypatch):
    """source=openvino + dtype=int8 → _prepare_openvino_int8_nncf
    (calibrator value is ignored for the routing decision — NNCF is the
    only int8 quantizer wired up for openvino in Wave 6)."""
    from scripts import run_cpu

    called = {"branch": None}

    def fake_int8(recipe_):
        called["branch"] = "openvino_int8_nncf"
        return "stub"

    monkeypatch.setattr(run_cpu, "_prepare_openvino_int8_nncf", fake_int8)
    recipe = _make_openvino_recipe(dtype="int8", calibrator="nncf")
    run_cpu._prepare_cpu_session(recipe)
    assert called["branch"] == "openvino_int8_nncf"


def test_ov_runner_as_ort_adapter_exposes_required_surface():
    """OVRunnerAsORT must expose .run(output_names, input_dict),
    .get_inputs()[i].name, .get_outputs()[i].name so measure_latency's
    forward closure and the existing output-name collection code in
    run_cpu.run() both keep working."""
    from scripts.run_cpu import OVRunnerAsORT
    import numpy as np

    class _FakePort:
        def __init__(self, name):
            self._name = name

    class _FakeCompiled:
        def __init__(self):
            self._out_ports = [_FakePort("out0")]

        def __call__(self, inputs):
            # Return whatever the harness asked for; shape doesn't matter here.
            return {self._out_ports[0]: np.zeros((1, 10), dtype=np.float32)}

        def output(self, i):
            return self._out_ports[i]

    compiled = _FakeCompiled()
    runner = OVRunnerAsORT(compiled, "images", ["out0"])

    # Interface shape matches ORT InferenceSession
    assert runner.get_inputs()[0].name == "images"
    assert runner.get_outputs()[0].name == "out0"

    # Forward call via .run(...) like ORT
    x = np.random.randn(1, 3, 640, 640).astype(np.float32)
    out = runner.run(["out0"], {"images": x})
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0].shape == (1, 10)


def test_run_cpu_still_trt_free_with_openvino_branch():
    """Importing scripts.run_cpu after Task 5 must not have pulled in
    tensorrt or pycuda. The openvino/nncf imports live inside the
    openvino dispatcher branches, not at module load."""
    pre_trt = "tensorrt" in sys.modules
    pre_pycuda = "pycuda" in sys.modules
    import scripts.run_cpu  # noqa: F401
    assert ("tensorrt" in sys.modules) == pre_trt
    assert ("pycuda" in sys.modules) == pre_pycuda


def test_openvino_recipe_34_35_load_cleanly():
    """recipes/34_openvino_fp32.yaml and 35_openvino_int8_nncf.yaml
    must pass load_recipe without edits to the schema."""
    from scripts._schemas import load_recipe

    r34 = load_recipe(str(ROOT / "recipes" / "34_openvino_fp32.yaml"))
    assert r34.technique.source == "openvino"
    assert r34.runtime.dtype == "fp32"

    r35 = load_recipe(str(ROOT / "recipes" / "35_openvino_int8_nncf.yaml"))
    assert r35.technique.source == "openvino"
    assert r35.runtime.dtype == "int8"
    assert r35.technique.calibrator == "nncf"
