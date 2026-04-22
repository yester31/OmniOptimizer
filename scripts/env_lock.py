"""Take a snapshot of the runtime environment and (optionally) lock GPU/CPU clocks.

Writes a JSON blob that every recipe run can embed verbatim into its result file,
so two different runs can be compared like-for-like.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Iterable, Optional

# Module-level constants so tests can monkey-patch the platform / paths without
# touching the real filesystem or reimplementing OS detection.
_SYSTEM = platform.system()
_LINUX_CPUINFO = "/proc/cpuinfo"
_LINUX_GOVERNOR = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"

# ISA features we care about for vision inference. Anything outside this set
# (fpu, vme, pge, ...) gets dropped so Result.env.cpu_flags stays focused on
# what actually influences kernel selection on CPU backends.
_TRACKED_FLAGS = frozenset({
    # x86 baseline
    "sse4_1", "sse4_2", "avx", "avx2", "fma",
    # AVX-512
    "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512cd",
    # Deep-learning INT8 / BF16 extensions
    "avx512_vnni", "avx_vnni",
    "avx512_bf16", "avx512_fp16",
    # Sapphire Rapids AMX
    "amx_bf16", "amx_int8", "amx_tile",
    # ARM
    "neon", "sve", "sve2",
})

# py-cpuinfo and macOS sysctl report several DL-relevant flags without the
# underscore Linux uses (avx512vnni vs avx512_vnni). Normalize to the Linux
# kernel name so Result.env.cpu_flags reads the same across platforms —
# otherwise a Tiger Lake host appears VNNI-less on Windows.
_FLAG_ALIASES = {
    "avx512vnni": "avx512_vnni",
    "avx512bf16": "avx512_bf16",
    "avx512fp16": "avx512_fp16",
    "amxbf16": "amx_bf16",
    "amxint8": "amx_int8",
    "amxtile": "amx_tile",
    "avxvnni": "avx_vnni",
}


def _normalize_flags(raw: Optional[Iterable[str]]) -> set[str]:
    """Lowercase + alias-resolve + filter to tracked features."""
    if not raw:
        return set()
    normalized = {_FLAG_ALIASES.get(f.lower(), f.lower()) for f in raw}
    return normalized & _TRACKED_FLAGS


def _run(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10)
        return out.strip()
    except Exception:
        return None


def nvidia_smi_query(field: str) -> Optional[str]:
    if not shutil.which("nvidia-smi"):
        return None
    return _run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits", "-i", "0"])


def _collect_cpu_info() -> dict:
    """Per-platform CPU detection.

    Returns a dict with keys:
      - cpu_model: str | None
      - cpu_cores_physical: int | None  (hyperthreading excluded)
      - cpu_flags: list[str] | None     (filtered to _TRACKED_FLAGS)
      - governor: str | None            (Linux only)
      - numa_node: int | None           (best effort, currently always None)

    Never raises — downstream callers rely on this to merge into the env dict
    unconditionally.
    """
    out: dict = {
        "cpu_model": None,
        "cpu_cores_physical": None,
        "cpu_flags": None,
        "governor": None,
        "numa_node": None,
    }
    if _SYSTEM == "Linux":
        out.update(_collect_linux())
    elif _SYSTEM == "Windows":
        out.update(_collect_windows())
    elif _SYSTEM == "Darwin":
        out.update(_collect_darwin())

    # Universal fallbacks — if a platform-specific path left something blank,
    # fill from crossplatform sources before returning.
    if not out["cpu_model"]:
        m = platform.processor() or platform.machine() or None
        out["cpu_model"] = m or None
    if out["cpu_cores_physical"] is None:
        try:
            import psutil
            n = psutil.cpu_count(logical=False)
            if n and n > 0:
                out["cpu_cores_physical"] = int(n)
        except Exception:
            pass
    return out


def _collect_linux() -> dict:
    out: dict = {}
    try:
        with open(_LINUX_CPUINFO, "r", encoding="utf-8") as f:
            model = None
            cores_per_socket = None
            flags_str = None
            physical_ids: set[str] = set()
            for line in f:
                if ":" not in line:
                    continue
                k, _, v = line.partition(":")
                k = k.strip().lower()
                v = v.strip()
                if k == "model name" and model is None:
                    model = v
                elif k == "cpu cores" and cores_per_socket is None:
                    try:
                        cores_per_socket = int(v)
                    except ValueError:
                        pass
                elif k == "flags" and flags_str is None:
                    flags_str = v
                elif k == "physical id":
                    physical_ids.add(v)
        if model:
            out["cpu_model"] = model
        if cores_per_socket is not None:
            sockets = max(1, len(physical_ids))
            out["cpu_cores_physical"] = cores_per_socket * sockets
        if flags_str:
            tracked = _normalize_flags(flags_str.split())
            out["cpu_flags"] = sorted(tracked) if tracked else None
    except OSError:
        pass
    try:
        with open(_LINUX_GOVERNOR, "r", encoding="utf-8") as f:
            gov = f.read().strip()
            if gov:
                out["governor"] = gov
    except OSError:
        pass
    return out


def _collect_windows() -> dict:
    out: dict = {}
    try:
        import cpuinfo  # py-cpuinfo (optional)
        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw") or info.get("brand_string") or info.get("brand")
        if brand:
            out["cpu_model"] = brand
        tracked = _normalize_flags(info.get("flags") or [])
        if tracked:
            out["cpu_flags"] = sorted(tracked)
    except Exception:
        pass
    return out


def _collect_darwin() -> dict:
    out: dict = {}
    model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if model:
        out["cpu_model"] = model
    cores = _run(["sysctl", "-n", "hw.physicalcpu"])
    if cores and cores.isdigit():
        out["cpu_cores_physical"] = int(cores)
    features_raw = _run(["sysctl", "-n", "machdep.cpu.features"]) or ""
    leaf7 = _run(["sysctl", "-n", "machdep.cpu.leaf7_features"]) or ""
    combined = f"{features_raw} {leaf7}".split()
    # sysctl reports flags like SSE4.1 / AVX1.0 — strip the dots so the
    # alias table can match (SSE4.1 → sse4_1 is already in _TRACKED_FLAGS).
    combined = [tok.replace(".", "_") for tok in combined]
    tracked = _normalize_flags(combined)
    if tracked:
        out["cpu_flags"] = sorted(tracked)
    return out


def collect_env() -> dict:
    env: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }

    # GPU via pynvml (preferred) or nvidia-smi
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        env["gpu"] = name.decode() if isinstance(name, bytes) else name
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
        env["gpu_compute_capability"] = f"{major}.{minor}"
        env["driver"] = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(env["driver"], bytes):
            env["driver"] = env["driver"].decode()
        pynvml.nvmlShutdown()
    except Exception:
        env["gpu"] = nvidia_smi_query("name")
        env["driver"] = nvidia_smi_query("driver_version")

    # CUDA runtime version
    try:
        import torch

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["cudnn"] = str(torch.backends.cudnn.version())
    except Exception:
        pass

    # ONNX Runtime
    try:
        import onnxruntime as ort

        env["onnxruntime"] = ort.__version__
    except Exception:
        pass

    # TensorRT
    try:
        import tensorrt as trt

        env["tensorrt"] = trt.__version__
    except Exception:
        # fall back to dpkg/nvidia-smi if available
        pass

    # Ultralytics
    try:
        import ultralytics

        env["ultralytics"] = ultralytics.__version__
    except Exception:
        pass

    # OpenVINO (Wave 6)
    try:
        import openvino as _ov

        env["openvino"] = _ov.__version__
    except Exception:
        pass

    # CPU (Wave 6) — only the fields EnvInfo carries. governor / numa_node
    # live on HardwareSpec so they stay out of the env dict.
    cpu = _collect_cpu_info()
    env["cpu_model"] = cpu["cpu_model"]
    env["cpu_cores_physical"] = cpu["cpu_cores_physical"]
    env["cpu_flags"] = cpu["cpu_flags"]

    return env


def lock_gpu_clock(enable: bool = True) -> Optional[str]:
    """Attempt to pin GPU clocks to their max. Requires sudo/admin on most systems.

    Returns a note string describing success/failure, or None if not attempted.
    """
    if not shutil.which("nvidia-smi"):
        return "nvidia-smi not found; clock not locked"

    if not enable:
        return "gpu_clock_lock disabled by recipe"

    max_clock = nvidia_smi_query("clocks.max.graphics")
    if not max_clock or not max_clock.isdigit():
        return "could not read max graphics clock"

    res = _run(["nvidia-smi", "-lgc", f"{max_clock},{max_clock}"])
    if res is None:
        return "nvidia-smi -lgc failed (likely needs admin/root)"
    return f"locked graphics clock to {max_clock} MHz"


def lock_cpu_clock(enable: bool = True) -> Optional[str]:
    """Best-effort CPU clock / governor lock for measurement hygiene.

    - Linux: `cpupower frequency-set -g performance` (needs sudo)
    - Windows: `powercfg /setactive <high-performance GUID>` (needs admin);
      this switches the power plan, which removes most DVFS headroom but
      does not hard-lock frequency. Thermal throttle is still possible —
      record note and let measure.py account for variance.
    - macOS: unsupported at the user-level API surface; return a descriptive
      note so callers can log it in Result.notes (degrade-not-crash).
    """
    if not enable:
        return "cpu_clock_lock disabled by recipe"

    if _SYSTEM == "Linux":
        if not shutil.which("cpupower"):
            return "cpupower not found; cpu clock not locked"
        res = _run(["cpupower", "frequency-set", "-g", "performance"])
        if res is None:
            return "cpupower frequency-set failed (likely needs sudo)"
        return "cpu governor set to performance"

    if _SYSTEM == "Windows":
        if not shutil.which("powercfg"):
            return "powercfg not found; cpu clock not locked"
        # GUID of the built-in 'High performance' power plan
        res = _run(["powercfg", "/setactive", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"])
        if res is None:
            return ("powercfg /setactive failed (likely needs admin); "
                    "frequency governor not locked, thermal variance possible")
        return ("Windows power plan set to High performance; "
                "frequency governor not strictly locked, thermal variance possible")

    if _SYSTEM == "Darwin":
        return ("cpu_clock_lock not supported on Darwin/macOS; "
                "thermal variance possible")

    return f"cpu_clock_lock not supported on {_SYSTEM}; thermal variance possible"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/_env.json")
    ap.add_argument("--lock-clock", action="store_true",
                    help="Lock GPU clock (legacy flag; also applies CPU clock lock).")
    ap.add_argument("--lock-cpu-clock", action="store_true",
                    help="Lock CPU frequency governor only (Wave 6 CPU recipes).")
    args = ap.parse_args()

    env = collect_env()
    notes: list[str] = []
    if args.lock_clock:
        gpu_note = lock_gpu_clock(True)
        if gpu_note:
            notes.append(f"gpu: {gpu_note}")
    if args.lock_clock or args.lock_cpu_clock:
        cpu_note = lock_cpu_clock(True)
        if cpu_note:
            notes.append(f"cpu: {cpu_note}")
    if notes:
        env["clock_lock_note"] = "; ".join(notes)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
