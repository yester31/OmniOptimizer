"""Wave 8 spike: Verify (R1) ONNX -> ncnn conversion of YOLO26n via pnnx +
(R2) ncnn pip wheel availability on Windows. Run BEFORE Wave 8 Task 1.

Four stages mirror the plan's Task 0 steps:
  0. `pip install ncnn` wheel exists for current Python + OS.
  1. `pip install pnnx` wheel exists (the converter pipeline).
  2. pnnx converts Wave 6 FP32 ONNX to ncnn .param / .bin.
  3. ncnn.Net loads + runs 1 forward on the converted model.

Exit 0 if all four pass. Exit 1 with findings doc reference if any blocker.

CAUTION: Running this spike spawns pnnx subprocess that downloads /
builds large artifacts. Plan for a few minutes on first run.
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMGSZ = 640
FP32_ONNX = ROOT / "results_cpu" / "_onnx" / "best_qr_640_fp32_dyn.onnx"
OUT_DIR = ROOT / "results_cpu" / "_ncnn_spike"
PNNX_EXE = Path(os.environ.get("APPDATA", "")) / "Python" / "Python313" / "Scripts" / "pnnx.exe"


def stage(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def _stage0_ncnn_import() -> bool:
    try:
        import ncnn
        print(f"[ncnn] version: {getattr(ncnn, '__version__', 'unknown')}")
        print(f"[ncnn] Net.load_onnx exists: {hasattr(ncnn.Net, 'load_onnx')}")
        return True
    except Exception:
        traceback.print_exc()
        return False


def _stage1_pnnx_exe() -> bool:
    if not PNNX_EXE.exists():
        print(f"[pnnx] executable missing at {PNNX_EXE}")
        print(f"[pnnx] install with: pip install pnnx")
        return False
    print(f"[pnnx] executable: {PNNX_EXE}")
    return True


def _stage2_convert() -> bool:
    if not FP32_ONNX.exists():
        print(f"[convert] input ONNX missing: {FP32_ONNX}")
        return False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PNNX_EXE),
        str(FP32_ONNX),
        f"inputshape=[1,3,{IMGSZ},{IMGSZ}]",
        f"pnnxparam={OUT_DIR}/best_qr.pnnx.param",
        f"pnnxbin={OUT_DIR}/best_qr.pnnx.bin",
        f"ncnnparam={OUT_DIR}/best_qr.ncnn.param",
        f"ncnnbin={OUT_DIR}/best_qr.ncnn.bin",
        f"ncnnpy={OUT_DIR}/best_qr_ncnn.py",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[convert] pnnx exit={result.returncode}")
            print(result.stderr[:500])
            return False
        ignore_lines = [ln for ln in result.stdout.splitlines() if ln.startswith("ignore ")]
        print(f"[convert] pnnx returned 0, but {len(ignore_lines)} ops marked 'ignore'")
        if ignore_lines:
            print("[convert] sample ignored ops:")
            for ln in ignore_lines[:8]:
                print(f"  {ln}")
            print("[convert] WARNING: these ops are SILENTLY DROPPED from the ncnn graph")
        return True
    except Exception:
        traceback.print_exc()
        return False


def _stage3_ncnn_forward() -> bool:
    import numpy as np
    import ncnn

    param = OUT_DIR / "best_qr.ncnn.param"
    bin_ = OUT_DIR / "best_qr.ncnn.bin"
    if not param.exists() or not bin_.exists():
        print(f"[forward] converted files missing: {param} / {bin_}")
        return False
    try:
        net = ncnn.Net()
        net.load_param(str(param))
        net.load_model(str(bin_))
        ex = net.create_extractor()
        x = np.random.randn(3, IMGSZ, IMGSZ).astype(np.float32)
        ex.input("in0", ncnn.Mat(x).clone())
        ret, out_mat = ex.extract("out0")
        if ret != 0:
            print(f"[forward] extract failed with ret={ret}")
            return False
        out = np.array(out_mat)
        print(f"[forward] OK, output shape: {out.shape}")
        return True
    except Exception:
        traceback.print_exc()
        return False


def main() -> int:
    print(f"Wave 8 R1 spike -- pnnx path, ROOT={ROOT}")
    results = {"ncnn_import": False, "pnnx_exe": False, "convert": False, "forward": False}

    stage("Stage 0: ncnn Python import")
    results["ncnn_import"] = _stage0_ncnn_import()

    stage("Stage 1: pnnx executable")
    results["pnnx_exe"] = _stage1_pnnx_exe()

    if results["pnnx_exe"]:
        stage("Stage 2: pnnx ONNX -> ncnn conversion")
        results["convert"] = _stage2_convert()

    if results["convert"]:
        stage("Stage 3: ncnn.Net forward smoke")
        results["forward"] = _stage3_ncnn_forward()

    stage("Summary")
    for k, v in results.items():
        print(f"  {k:15} {v}")

    if all(results.values()):
        print("\nWAVE 8 R1/R2 VERDICT: CLEARED")
        return 0
    if results["convert"] and not results["forward"]:
        print("\nWAVE 8 R1 BLOCKED: conversion succeeded but ncnn runtime rejects the graph")
        print("  (unsupported op dropped silently -- 'layer X not exists or registered')")
        return 1
    if results["ncnn_import"] and not results["convert"]:
        print("\nWAVE 8 R1 BLOCKED: pnnx conversion failed")
        return 1
    print("\nWAVE 8 R2 BLOCKED: ncnn or pnnx pip install missing")
    return 1


if __name__ == "__main__":
    sys.exit(main())
