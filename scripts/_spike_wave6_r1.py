"""Wave 6 spike: Verify YOLO26n compatibility with OpenVINO + NNCF.

Checks the 4 stages that must succeed for recipe #35 (openvino_int8_nncf) to be
viable:

  1. Plain FP32 ONNX export from best_qr.pt.
  2. OpenVINO `read_model` + `compile_model("CPU")` + 1 forward pass on FP32 IR.
  3. NNCF `nncf.quantize(...)` finishes without raising on the YOLO26n graph
     (the specific concern: attention block Reshape nodes going stale after
     quantization, as INC SmoothQuant did in Wave 3).
  4. Quantized IR compiles and runs 1 forward pass with output shape preserved.

Not in scope:
  - mAP measurement (Wave 6 Task 10).
  - Latency benchmarking (Wave 6 Task 7/10).
  - Calibration quality — random-noise calib is sufficient for graph compat.

Exit code: 0 if all 4 stages pass, 1 otherwise. One-line summary printed at end.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "best_qr.pt"
ONNX_PATH = ROOT / "best_qr.onnx"
OV_FP32_IR = ROOT / "results_cpu" / "_ov_ir" / "best_qr_fp32.xml"
OV_INT8_IR = ROOT / "results_cpu" / "_ov_ir" / "best_qr_int8_nncf.xml"
IMGSZ = 640
CALIB_SAMPLES = 32


def stage(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def _export_onnx() -> Path:
    if ONNX_PATH.exists():
        print(f"[reuse] {ONNX_PATH}")
        return ONNX_PATH
    from ultralytics import YOLO

    m = YOLO(str(WEIGHTS))
    out = m.export(format="onnx", imgsz=IMGSZ, half=False, simplify=True, dynamic=True)
    src = Path(out)
    if src != ONNX_PATH:
        src.rename(ONNX_PATH)
    print(f"[exported] {ONNX_PATH} ({ONNX_PATH.stat().st_size // 1024}KB)")
    return ONNX_PATH


def _stage1_fp32_ov(onnx_path: Path):
    import openvino as ov

    OV_FP32_IR.parent.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    ov_model = core.read_model(str(onnx_path))
    print(f"[fp32] read_model OK, inputs={len(ov_model.inputs)}, outputs={len(ov_model.outputs)}")
    print(f"[fp32] input: {ov_model.inputs[0].get_any_name()} shape={ov_model.inputs[0].partial_shape}")
    for i, o in enumerate(ov_model.outputs):
        print(f"[fp32] output[{i}]: {o.get_any_name()} shape={o.partial_shape}")

    ov.save_model(ov_model, str(OV_FP32_IR))
    print(f"[fp32] save_model → {OV_FP32_IR}")

    compiled = core.compile_model(ov_model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    print(f"[fp32] compile_model OK")

    # 1 forward pass
    x = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    t0 = time.perf_counter()
    out = compiled([x])
    dt = (time.perf_counter() - t0) * 1000
    out_shapes = [list(v.shape) for v in out.values()]
    print(f"[fp32] infer OK {dt:.1f}ms, output_shapes={out_shapes}")
    return compiled, out_shapes


def _build_calib_dataset():
    import nncf

    def _gen():
        rng = np.random.default_rng(42)
        for _ in range(CALIB_SAMPLES):
            yield rng.standard_normal(size=(1, 3, IMGSZ, IMGSZ)).astype(np.float32)

    return nncf.Dataset(_gen(), transform_func=lambda x: x)


def _stage2_nncf_quantize(onnx_path: Path):
    import openvino as ov
    import nncf

    core = ov.Core()
    ov_fp32 = core.read_model(str(onnx_path))
    calib = _build_calib_dataset()

    print(f"[nncf] quantize start, samples={CALIB_SAMPLES}")
    t0 = time.perf_counter()
    try:
        ov_int8 = nncf.quantize(
            ov_fp32,
            calib,
            preset=nncf.QuantizationPreset.MIXED,
            target_device=nncf.TargetDevice.CPU,
            subset_size=CALIB_SAMPLES,
        )
    except Exception:
        print(f"[nncf] quantize FAILED after {time.perf_counter()-t0:.1f}s")
        traceback.print_exc()
        return None
    dt = time.perf_counter() - t0
    print(f"[nncf] quantize OK in {dt:.1f}s")

    OV_INT8_IR.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_int8, str(OV_INT8_IR))
    print(f"[nncf] save_model → {OV_INT8_IR}")
    return ov_int8


def _stage3_int8_infer(ov_int8):
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(ov_int8, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    print(f"[int8] compile_model OK")

    x = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    t0 = time.perf_counter()
    out = compiled([x])
    dt = (time.perf_counter() - t0) * 1000
    out_shapes = [list(v.shape) for v in out.values()]
    print(f"[int8] infer OK {dt:.1f}ms, output_shapes={out_shapes}")
    return out_shapes


def main() -> int:
    print(f"Wave 6 R1 spike — best_qr.pt → ONNX → OpenVINO FP32 → NNCF INT8")
    print(f"ROOT={ROOT}")
    if not WEIGHTS.exists():
        print(f"[abort] {WEIGHTS} missing")
        return 1

    results = {
        "export": False,
        "fp32_ov": False,
        "nncf_quant": False,
        "int8_infer": False,
        "shapes_match": False,
    }
    fp32_shapes: list | None = None
    int8_shapes: list | None = None

    try:
        stage("Stage 0: ONNX export")
        onnx = _export_onnx()
        results["export"] = True

        stage("Stage 1: OpenVINO FP32")
        _, fp32_shapes = _stage1_fp32_ov(onnx)
        results["fp32_ov"] = True

        stage("Stage 2: NNCF PTQ")
        ov_int8 = _stage2_nncf_quantize(onnx)
        if ov_int8 is None:
            raise RuntimeError("NNCF quantize returned None")
        results["nncf_quant"] = True

        stage("Stage 3: OpenVINO INT8 inference")
        int8_shapes = _stage3_int8_infer(ov_int8)
        results["int8_infer"] = True

        results["shapes_match"] = fp32_shapes == int8_shapes
    except Exception:
        traceback.print_exc()

    stage("Summary")
    for k, v in results.items():
        print(f"  {k:16} {'PASS' if v else 'FAIL'}")
    if fp32_shapes and int8_shapes:
        print(f"  fp32_shapes={fp32_shapes}")
        print(f"  int8_shapes={int8_shapes}")

    all_pass = all(results.values())
    print(f"\nR1 VERDICT: {'CLEARED' if all_pass else 'BLOCKED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
