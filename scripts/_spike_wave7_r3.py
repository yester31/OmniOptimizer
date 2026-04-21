"""Wave 7 spike: Verify (R3) torch.export + ultralytics YOLO26n compat and
(R5) XNNPACK EP QDQ INT8 fallback behavior -- both ahead of Wave 7 Task 2/3
implementation.

Five stages mirror the plan's Task 0 steps:
  0. XNNPACK EP availability in the installed onnxruntime.
  1. torch.export on YOLO26n (strict=True -> strict=False fallback).
  2. PT2E X86InductorQuantizer prepare + calibrate + convert (1 sample).
  3. torch.compile latency (CPU default mode) -- informational only.
  4. XNNPACK EP on a Wave 6 static INT8 ONNX -- verify actual provider
     assignment via session.get_providers().

Exit 0 if all critical stages pass (0, 1 mandatory; 2 failure downgrades
Wave 7 to FP32-only; 4 failure downgrades #43 to re-quantize-required).

Runs on CPU only. ~3-8 minutes depending on first-torch-import cache.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMGSZ = 640
WEIGHTS = ROOT / "best_qr.pt"
CACHED_INT8_ONNX = ROOT / "results_cpu" / "_onnx" / "ort_cpu_int8_static_int8_entropy.onnx"


def stage(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def _stage0_xnnpack_available() -> bool:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"[xnnpack] ort={ort.__version__}, providers={providers}")
    ok = "XnnpackExecutionProvider" in providers
    print(f"[xnnpack] XnnpackExecutionProvider registered: {ok}")
    return ok


def _stage1_torch_export():
    """Try strict=True, fall back to strict=False. Return the exported
    program or None."""
    import torch
    from ultralytics import YOLO

    print(f"[export] torch={torch.__version__}")
    model = YOLO(str(WEIGHTS)).model.eval().cpu()
    example = torch.randn(1, 3, IMGSZ, IMGSZ)

    for strict in (True, False):
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                exported = torch.export.export(model, (example,), strict=strict)
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"[export] strict={strict} OK in {dt:.1f}ms")
            # Smoke: 1 forward via exported.module()
            with torch.no_grad():
                out = exported.module()(example)
            out_shape = tuple(out.shape) if isinstance(out, torch.Tensor) else \
                        [tuple(o.shape) for o in out]
            print(f"[export] smoke forward OK, output shape={out_shape}")
            return exported, strict
        except Exception as e:
            print(f"[export] strict={strict} FAILED: {type(e).__name__}: {e}")
            if strict:
                print("[export] trying strict=False fallback...")
            else:
                traceback.print_exc()
    return None, None


def _stage2_pt2e_quantize(exported):
    import torch
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
        X86InductorQuantizer, get_default_x86_inductor_quantization_config,
    )

    try:
        t0 = time.perf_counter()
        quantizer = X86InductorQuantizer()
        quantizer.set_global(get_default_x86_inductor_quantization_config())
        prepared = prepare_pt2e(exported.module(), quantizer)
        print(f"[pt2e] prepare_pt2e OK in {(time.perf_counter()-t0)*1000:.1f}ms")

        # 1-sample calibrate
        t0 = time.perf_counter()
        with torch.no_grad():
            prepared(torch.randn(1, 3, IMGSZ, IMGSZ))
        print(f"[pt2e] calibrate 1 sample OK in {(time.perf_counter()-t0)*1000:.1f}ms")

        t0 = time.perf_counter()
        quantized = convert_pt2e(prepared)
        print(f"[pt2e] convert_pt2e OK in {(time.perf_counter()-t0)*1000:.1f}ms")

        # Forward on the quantized graph (no torch.compile yet)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = quantized(torch.randn(1, 3, IMGSZ, IMGSZ))
        out_shape = tuple(out.shape) if isinstance(out, torch.Tensor) else \
                    [tuple(o.shape) for o in out]
        print(f"[pt2e] forward quantized OK in {(time.perf_counter()-t0)*1000:.1f}ms, shape={out_shape}")
        return quantized
    except Exception:
        print("[pt2e] FAILED")
        traceback.print_exc()
        return None


def _stage3_torch_compile_latency(module):
    """Measure first torch.compile(module) call latency -- informational,
    used to size Wave 7's cold_start_ms expectation."""
    import torch

    try:
        t0 = time.perf_counter()
        compiled = torch.compile(module)  # CPU default mode
        print(f"[compile] torch.compile wrap: {(time.perf_counter()-t0)*1000:.1f}ms (lazy)")

        example = torch.randn(1, 3, IMGSZ, IMGSZ)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = compiled(example)
        first = (time.perf_counter() - t0) * 1000.0
        print(f"[compile] first call (trigger compile): {first:.1f}ms")

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = compiled(example)
        second = (time.perf_counter() - t0) * 1000.0
        print(f"[compile] second call (cached): {second:.1f}ms")
        return first, second
    except Exception:
        print("[compile] FAILED")
        traceback.print_exc()
        return None, None


def _stage4_xnnpack_fallback_probe():
    """Load a Wave 6 static INT8 ONNX with XNNPACK EP first in providers
    list. Report which providers actually got assigned."""
    if not CACHED_INT8_ONNX.exists():
        print(f"[xnnpack-fallback] {CACHED_INT8_ONNX} missing, skipping")
        return None

    import numpy as np
    import onnxruntime as ort

    try:
        t0 = time.perf_counter()
        session = ort.InferenceSession(
            str(CACHED_INT8_ONNX),
            providers=["XnnpackExecutionProvider", "CPUExecutionProvider"],
        )
        print(f"[xnnpack-fallback] InferenceSession OK in {(time.perf_counter()-t0)*1000:.1f}ms")
        assigned = session.get_providers()
        print(f"[xnnpack-fallback] providers assigned: {assigned}")
        xnnpack_used = "XnnpackExecutionProvider" in assigned
        print(f"[xnnpack-fallback] XNNPACK actually assigned: {xnnpack_used}")

        # 1 forward smoke
        x = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)
        in_name = session.get_inputs()[0].name
        t0 = time.perf_counter()
        out = session.run(None, {in_name: x})
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[xnnpack-fallback] forward OK {dt:.1f}ms, output shapes={[o.shape for o in out]}")
        return xnnpack_used
    except Exception:
        print("[xnnpack-fallback] FAILED")
        traceback.print_exc()
        return None


def main() -> int:
    print(f"Wave 7 R3/R5 spike -- CPU host, weights={WEIGHTS}")
    if not WEIGHTS.exists():
        print(f"[abort] {WEIGHTS} missing -- this spike requires best_qr.pt")
        return 1

    results = {
        "xnnpack_registered": False,
        "torch_export": False,
        "export_strict": None,
        "pt2e_quantize": False,
        "torch_compile": False,
        "xnnpack_used_on_int8": None,
    }

    try:
        stage("Stage 0: XNNPACK EP availability")
        results["xnnpack_registered"] = _stage0_xnnpack_available()

        stage("Stage 1: torch.export YOLO26n")
        exported, strict = _stage1_torch_export()
        results["torch_export"] = exported is not None
        results["export_strict"] = strict

        quantized = None
        if exported is not None:
            stage("Stage 2: PT2E X86InductorQuantizer")
            quantized = _stage2_pt2e_quantize(exported)
            results["pt2e_quantize"] = quantized is not None

        if quantized is not None:
            stage("Stage 3: torch.compile latency (CPU default mode)")
            first, second = _stage3_torch_compile_latency(quantized)
            results["torch_compile"] = first is not None

        stage("Stage 4: XNNPACK EP QDQ fallback probe")
        results["xnnpack_used_on_int8"] = _stage4_xnnpack_fallback_probe()
    except Exception:
        traceback.print_exc()

    stage("Summary")
    for k, v in results.items():
        print(f"  {k:24} {v}")

    # Verdict mirrors plan Task 0 Step 6
    export_ok = results["torch_export"]
    quant_ok = results["pt2e_quantize"]
    xnnpack_registered = results["xnnpack_registered"]

    if export_ok and quant_ok and xnnpack_registered:
        print("\nR3/R5 VERDICT: CLEARED -- Wave 7 full scope can proceed")
        return 0
    if export_ok and not quant_ok:
        print("\nR3 PARTIAL: FP32 export OK, INT8 PT2E failed -- Wave 7 #41 parked, #40 only")
        return 1
    if not export_ok:
        print("\nR3 BLOCKED: torch.export failed -- Wave 7 PT2E both #40 and #41 parked, XNNPACK-only Wave")
        return 1
    if not xnnpack_registered:
        print("\nR5 BLOCKED: XNNPACK EP not registered -- Wave 7 #42/#43 parked, PT2E-only Wave")
        return 1
    print("\nR3/R5 VERDICT: unclear state, see stages above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
