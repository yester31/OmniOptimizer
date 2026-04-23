"""Wave 15 D4.3 — verify _get_ov_core wires up CACHE_DIR.

The goal is to catch silent regressions where a refactor drops the
``set_property({"CACHE_DIR": ...})`` call, which would make every OV recipe
pay full kernel-compile cold_start every session with no visible error.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


def _install_fake_openvino(monkeypatch):
    """Install a minimal fake ``openvino`` module so run_cpu._get_ov_core
    can execute without a real OV install. Returns the fake Core instance
    so tests can inspect set_property calls.
    """
    fake_core = MagicMock(name="Core_instance")

    def _Core():  # factory that always hands back the same instance
        return fake_core

    fake_ov = types.ModuleType("openvino")
    fake_ov.Core = _Core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openvino", fake_ov)
    return fake_core


def test_get_ov_core_sets_cache_dir(monkeypatch, tmp_path):
    import scripts.run_cpu as run_cpu

    # Reset the module-level singleton so _get_ov_core re-enters the init branch.
    monkeypatch.setattr(run_cpu, "_OV_CORE", None)

    fake_core = _install_fake_openvino(monkeypatch)

    # Run from tmp_path so we don't pollute the real results_cpu/ tree.
    monkeypatch.chdir(tmp_path)

    core = run_cpu._get_ov_core()

    # Assert: it's the fake we installed
    assert core is fake_core

    # Assert: set_property was called exactly once with a dict carrying CACHE_DIR
    assert fake_core.set_property.call_count == 1
    (kwargs_or_pos,), _ = fake_core.set_property.call_args
    assert isinstance(kwargs_or_pos, dict), "OV set_property expects a dict of properties"
    assert "CACHE_DIR" in kwargs_or_pos, "Wave 15 D1.1 regression: CACHE_DIR key missing"

    # Assert: the directory path set as CACHE_DIR was actually created on disk
    cache_path = Path(kwargs_or_pos["CACHE_DIR"])
    assert cache_path.is_dir(), f"Cache dir {cache_path} was not created"


def test_get_ov_core_singleton_avoids_second_init(monkeypatch, tmp_path):
    import scripts.run_cpu as run_cpu

    monkeypatch.setattr(run_cpu, "_OV_CORE", None)
    fake_core = _install_fake_openvino(monkeypatch)
    monkeypatch.chdir(tmp_path)

    run_cpu._get_ov_core()
    run_cpu._get_ov_core()
    run_cpu._get_ov_core()

    # Singleton: only the first call should invoke set_property.
    # A regression where every call re-initializes would show call_count > 1.
    assert fake_core.set_property.call_count == 1


def test_get_ov_core_survives_property_failure(monkeypatch, tmp_path, capsys):
    """CACHE_DIR is a speedup, not a correctness requirement. Permission or
    IO failures must leave the Core usable (runner continues without cache)."""
    import scripts.run_cpu as run_cpu

    monkeypatch.setattr(run_cpu, "_OV_CORE", None)
    fake_core = _install_fake_openvino(monkeypatch)
    fake_core.set_property.side_effect = PermissionError("simulated access denied")
    monkeypatch.chdir(tmp_path)

    core = run_cpu._get_ov_core()
    assert core is fake_core  # Core still returned, not None

    err = capsys.readouterr().err
    assert "CACHE_DIR" in err or "cache" in err.lower(), (
        "Expected a warning log on cache setup failure"
    )
