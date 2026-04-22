"""Wave 15 D4.4 — verify _make_session injects TRT EP optimization opts.

Regression guard: D1.2 adds trt_builder_optimization_level=5 + timing cache
keys to the TensorrtExecutionProvider options dict. If a refactor drops
these, recipe #04 silently reverts to fps ~211 (pre-Wave 15 baseline) with
no test failure. This module verifies the dict passed to InferenceSession
carries the Wave 15 keys for the TRT EP path and leaves non-TRT EPs alone.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _install_fake_onnxruntime(monkeypatch, *, providers_listed=None, capture=None):
    """Install a fake ``onnxruntime`` module.

    capture: a dict to stash the ``providers`` arg from InferenceSession so
    the test can assert against it after _make_session returns.
    """
    fake_ort = types.ModuleType("onnxruntime")

    class _GraphOptimizationLevel:
        ORT_ENABLE_ALL = "ORT_ENABLE_ALL"

    class _SessionOptions:
        def __init__(self):
            self.graph_optimization_level = None

    def _get_available_providers():
        return providers_listed or [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    class _InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            if capture is not None:
                capture["providers"] = providers
                capture["sess_options"] = sess_options
            self._providers = providers or []

        def get_providers(self):
            # Surface the EP name as the primary so _make_session's
            # "silent fallback guard" passes.
            if not self._providers:
                return []
            first = self._providers[0]
            if isinstance(first, tuple):
                return [first[0]] + [p for p in self._providers[1:] if isinstance(p, str)]
            return [p if isinstance(p, str) else p[0] for p in self._providers]

    fake_ort.GraphOptimizationLevel = _GraphOptimizationLevel  # type: ignore[attr-defined]
    fake_ort.SessionOptions = _SessionOptions  # type: ignore[attr-defined]
    fake_ort.get_available_providers = _get_available_providers  # type: ignore[attr-defined]
    fake_ort.InferenceSession = _InferenceSession  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    return fake_ort


def test_trt_ep_receives_wave15_opts(monkeypatch, tmp_path):
    from scripts import run_ort

    capture = {}
    _install_fake_onnxruntime(monkeypatch, capture=capture)

    onnx_stub = tmp_path / "model.onnx"
    onnx_stub.write_bytes(b"")  # _make_session doesn't actually parse the ONNX

    run_ort._make_session(onnx_stub, "TensorrtExecutionProvider", dtype="fp16")

    providers = capture["providers"]
    assert providers, "providers list was empty"
    first = providers[0]
    assert isinstance(first, tuple) and first[0] == "TensorrtExecutionProvider", (
        f"Expected TRT EP tuple as first provider, got {first!r}"
    )
    trt_opts = first[1]

    # Wave 11 baseline keys — must still be present
    assert trt_opts["trt_engine_cache_enable"] is True
    assert "trt_engine_cache_path" in trt_opts
    assert trt_opts["trt_fp16_enable"] is True  # dtype=="fp16"

    # Wave 15 D1.2 additions
    assert trt_opts.get("trt_builder_optimization_level") == 5, (
        "Wave 15 regression: trt_builder_optimization_level=5 missing or wrong"
    )
    assert trt_opts.get("trt_timing_cache_enable") is True
    assert trt_opts.get("trt_timing_cache_path") == trt_opts["trt_engine_cache_path"]
    assert trt_opts.get("trt_detailed_build_log") is True


def test_trt_ep_fp32_still_disables_fp16(monkeypatch, tmp_path):
    from scripts import run_ort

    capture = {}
    _install_fake_onnxruntime(monkeypatch, capture=capture)

    onnx_stub = tmp_path / "model.onnx"
    onnx_stub.write_bytes(b"")

    run_ort._make_session(onnx_stub, "TensorrtExecutionProvider", dtype="fp32")

    trt_opts = capture["providers"][0][1]
    assert trt_opts["trt_fp16_enable"] is False
    # Wave 15 keys still set regardless of dtype
    assert trt_opts["trt_builder_optimization_level"] == 5


def test_non_trt_ep_gets_no_trt_opts(monkeypatch, tmp_path):
    """CUDA/CPU EP paths must NOT receive TRT-specific keys — catches a
    refactor bug where the trt_opts dict leaks into the else branch."""
    from scripts import run_ort

    capture = {}
    _install_fake_onnxruntime(
        monkeypatch,
        providers_listed=["CUDAExecutionProvider", "CPUExecutionProvider"],
        capture=capture,
    )

    onnx_stub = tmp_path / "model.onnx"
    onnx_stub.write_bytes(b"")

    run_ort._make_session(onnx_stub, "CUDAExecutionProvider", dtype="fp16")

    providers = capture["providers"]
    # CUDA EP path: providers is a plain list of strings (no tuple, no opts)
    for p in providers:
        assert not isinstance(p, tuple), (
            f"Non-TRT EP path leaked provider_options tuple: {p!r}"
        )


def test_legacy_ort_fallback_strips_wave15_keys(monkeypatch, tmp_path):
    """Simulate an older ORT that rejects the new TRT EP keys. The runner
    must retry with the legacy key set rather than propagate the error."""
    from scripts import run_ort

    # Custom fake that rejects on first call if Wave 15 keys present,
    # succeeds on second call after they're stripped.
    call_log: list[dict] = []

    fake_ort = types.ModuleType("onnxruntime")

    class _GraphOptimizationLevel:
        ORT_ENABLE_ALL = "ORT_ENABLE_ALL"

    class _SessionOptions:
        def __init__(self):
            self.graph_optimization_level = None

    def _get_available_providers():
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

    class _InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            trt_opts = providers[0][1] if providers and isinstance(providers[0], tuple) else {}
            call_log.append({"trt_opts": dict(trt_opts)})
            if "trt_builder_optimization_level" in trt_opts:
                raise ValueError(
                    "Unknown provider option: trt_builder_optimization_level"
                )
            self._providers = providers

        def get_providers(self):
            return [p[0] if isinstance(p, tuple) else p for p in self._providers]

    fake_ort.GraphOptimizationLevel = _GraphOptimizationLevel  # type: ignore[attr-defined]
    fake_ort.SessionOptions = _SessionOptions  # type: ignore[attr-defined]
    fake_ort.get_available_providers = _get_available_providers  # type: ignore[attr-defined]
    fake_ort.InferenceSession = _InferenceSession  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    onnx_stub = tmp_path / "model.onnx"
    onnx_stub.write_bytes(b"")

    # Should NOT raise — the fallback path strips Wave 15 keys and retries.
    run_ort._make_session(onnx_stub, "TensorrtExecutionProvider", dtype="fp16")

    assert len(call_log) == 2, f"Expected 2 InferenceSession attempts (initial + retry), got {len(call_log)}"
    first, second = call_log
    assert "trt_builder_optimization_level" in first["trt_opts"]
    assert "trt_builder_optimization_level" not in second["trt_opts"]
    assert "trt_timing_cache_enable" not in second["trt_opts"]
    # Wave 11 baseline keys survive the retry
    assert second["trt_opts"]["trt_engine_cache_enable"] is True
    assert second["trt_opts"]["trt_fp16_enable"] is True
