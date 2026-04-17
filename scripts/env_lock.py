"""Take a snapshot of the runtime environment and (optionally) lock GPU clocks.

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
from typing import Optional


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/_env.json")
    ap.add_argument("--lock-clock", action="store_true")
    args = ap.parse_args()

    env = collect_env()
    note = lock_gpu_clock(args.lock_clock) if args.lock_clock else None
    if note:
        env["clock_lock_note"] = note

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
