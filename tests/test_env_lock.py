"""Tests for scripts/env_lock.py — CPU info collection + clock lock.

Wave 6 Task 2: env_lock gains CPU awareness so Result.env carries
cpu_model / cpu_cores_physical / cpu_flags on every run (historical
GPU results stay valid — all fields optional, defaults to None).

Platform coverage:
- Windows: real-system smoke (this test host). CI matrix will broaden.
- Linux: mocked /proc/cpuinfo + sysfs governor path.
- macOS: best-effort (sysctl).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_collect_env_returns_cpu_model_on_current_host():
    """On any host, collect_env() must populate cpu_model to something
    truthy (not None, not empty). The exact string is host-dependent."""
    from scripts.env_lock import collect_env

    env = collect_env()
    assert "cpu_model" in env
    assert env["cpu_model"], f"cpu_model must be non-empty, got {env['cpu_model']!r}"


def test_collect_env_returns_cpu_cores_physical_positive_int():
    """cpu_cores_physical must be a positive integer on the current host."""
    from scripts.env_lock import collect_env

    env = collect_env()
    assert "cpu_cores_physical" in env
    n = env["cpu_cores_physical"]
    assert isinstance(n, int) and n >= 1, \
        f"cpu_cores_physical must be positive int, got {n!r}"


def test_collect_env_cpu_flags_is_list_or_none():
    """cpu_flags is a list[str] when ISA detection available,
    None when not installable (e.g. py-cpuinfo absent on Windows
    and no /proc/cpuinfo)."""
    from scripts.env_lock import collect_env

    env = collect_env()
    flags = env.get("cpu_flags")
    assert flags is None or (isinstance(flags, list) and all(isinstance(f, str) for f in flags))


def test_collect_cpu_info_linux_proc_cpuinfo(monkeypatch, tmp_path):
    """Linux path: parse /proc/cpuinfo for model name + flags and
    /sys/.../scaling_governor for the governor."""
    from scripts import env_lock

    fake_cpuinfo = tmp_path / "cpuinfo"
    fake_cpuinfo.write_text(
        "processor\t: 0\n"
        "model name\t: Intel(R) Xeon(R) Platinum 8480+\n"
        "cpu cores\t: 56\n"
        "physical id\t: 0\n"
        "core id\t: 0\n"
        "flags\t\t: fpu vme avx2 avx512f avx512_vnni avx512_bf16 amx_tile\n"
        "\n"
        "processor\t: 1\n"
        "model name\t: Intel(R) Xeon(R) Platinum 8480+\n"
        "cpu cores\t: 56\n"
        "physical id\t: 0\n"
        "core id\t: 1\n"
        "flags\t\t: fpu vme avx2 avx512f avx512_vnni avx512_bf16 amx_tile\n"
        "\n"
    )
    fake_gov = tmp_path / "scaling_governor"
    fake_gov.write_text("performance\n")

    monkeypatch.setattr(env_lock, "_LINUX_CPUINFO", str(fake_cpuinfo))
    monkeypatch.setattr(env_lock, "_LINUX_GOVERNOR", str(fake_gov))
    # Force the Linux code path even if pytest runs on Windows.
    monkeypatch.setattr(env_lock, "_SYSTEM", "Linux")

    info = env_lock._collect_cpu_info()
    assert info["cpu_model"] == "Intel(R) Xeon(R) Platinum 8480+"
    assert info["cpu_cores_physical"] == 56
    assert info["cpu_flags"] is not None
    assert "avx512_vnni" in info["cpu_flags"]
    assert "amx_tile" in info["cpu_flags"]
    assert "fpu" not in info["cpu_flags"], (
        "cpu_flags must be filtered to the ISA features we care about "
        "(avx2/avx512*/amx/sse*), not every flag"
    )
    assert info["governor"] == "performance"


def test_normalize_flags_resolves_py_cpuinfo_aliases():
    """py-cpuinfo emits AVX-512 VNNI/BF16/AMX flags without the underscore
    Linux uses (avx512vnni vs avx512_vnni). The alias table must resolve
    these so Windows hosts don't appear to lack VNNI on Tiger Lake+."""
    from scripts.env_lock import _normalize_flags

    # py-cpuinfo style (no underscore)
    assert "avx512_vnni" in _normalize_flags(["avx512vnni"])
    assert "avx512_bf16" in _normalize_flags(["avx512bf16"])
    assert "amx_tile" in _normalize_flags(["amxtile"])
    # Linux kernel style passes through
    assert "avx512_vnni" in _normalize_flags(["avx512_vnni"])
    # Unknown flags are dropped
    assert "fpu" not in _normalize_flags(["fpu", "vme", "pge"])
    assert _normalize_flags([]) == set()
    assert _normalize_flags(None) == set()


def test_lock_cpu_clock_disabled_returns_note():
    """lock_cpu_clock(False) returns a 'disabled' note, no subprocess runs."""
    from scripts.env_lock import lock_cpu_clock

    note = lock_cpu_clock(False)
    assert note is not None
    assert "disabled" in note.lower()


def test_lock_cpu_clock_unsupported_platform_returns_note(monkeypatch):
    """On macOS or unknown platforms, lock_cpu_clock(True) must not raise
    — it returns a best-effort note so run_cpu.py can record it in
    Result.notes and carry on (degrade-not-crash)."""
    from scripts import env_lock

    monkeypatch.setattr(env_lock, "_SYSTEM", "Darwin")
    note = env_lock.lock_cpu_clock(True)
    assert note is not None
    # Must mention the platform limitation somehow
    assert any(kw in note.lower() for kw in ("darwin", "macos", "not supported", "unsupported"))


def test_run_cpu_no_longer_uses_platform_processor_fallback():
    """After Task 2, run_cpu.py should drop its platform.processor()
    shim — collect_env() now populates cpu_model properly."""
    run_cpu_src = (ROOT / "scripts" / "run_cpu.py").read_text(encoding="utf-8")
    assert "platform.processor()" not in run_cpu_src, (
        "run_cpu.py still carries the platform.processor() shim; "
        "remove it now that env_lock.collect_env() fills cpu_model."
    )
