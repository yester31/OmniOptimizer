"""Phase 7 — Fair bench using scripts/run_trt.py protocol (same as report_qr.md).

ultralytics 가 export 한 .engine 은 metadata prefix 가 붙어 순수 TRT runtime 이
deserialize 못함. 따라서 ONNX → _build_engine 으로 순수 TRT engine 재빌드 후
measure_latency(warmup=100, measure=100) + CUDA event 기반 bench.

variants:
  (A) fastnas_fp16         best.onnx → fp16
  (B) fastnas_int8         best.onnx → int8 (QR calibration)
  (C) fastnas_sp_fp16      B_final.onnx → fp16 + sparsity="2:4"
  (D) fastnas_sp_int8      B_final.onnx → int8 + sparsity="2:4" (QR calibration)

환경변수:
  OMNI_COCO_YAML=qr_barcode.yaml   INT8 calibration 이미지 소스
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OMNI_COCO_YAML", str(REPO_ROOT / "qr_barcode.yaml"))

from scripts.measure import measure_latency, throughput_from_latency  # noqa: E402
from scripts.run_trt import _build_engine, _make_trt_forward  # noqa: E402


# (onnx_path, dtype, sparsity, calib_samples)
VARIANTS = {
    "A_fastnas_fp16": {
        "onnx": REPO_ROOT / "trained_weights" / "23_fastnas_p1_finetune" / "weights" / "best.onnx",
        "dtype": "fp16", "sparsity": None, "calib_samples": 0,
    },
    "B_fastnas_int8": {
        "onnx": REPO_ROOT / "trained_weights" / "23_fastnas_p1_finetune" / "weights" / "best.onnx",
        "dtype": "int8", "sparsity": None, "calib_samples": 512,
    },
    "C_fastnas_sp_fp16": {
        "onnx": REPO_ROOT / "trained_weights" / "23_fastnas_chain_ft" / "B_final.onnx",
        "dtype": "fp16", "sparsity": "2:4", "calib_samples": 0,
    },
    "D_fastnas_sp_int8": {
        "onnx": REPO_ROOT / "trained_weights" / "23_fastnas_chain_ft" / "B_final.onnx",
        "dtype": "int8", "sparsity": "2:4", "calib_samples": 512,
    },
}

ENGINE_DIR = REPO_ROOT / "trained_weights" / "23_fair_bench_engines"


def build_and_bench(label: str, spec: dict, batch_size: int = 1, imgsz: int = 640) -> dict:
    print(f"\n=== {label}  bs={batch_size} ===")
    onnx_path = spec["onnx"]
    dtype = spec["dtype"]
    sparsity = spec["sparsity"]
    calib_samples = spec["calib_samples"]

    assert onnx_path.exists(), f"missing {onnx_path}"
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    engine_path = ENGINE_DIR / f"{label}_bs{batch_size}.engine"

    # Build
    t0 = time.time()
    built, err = _build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        dtype=dtype,
        sparsity=sparsity,
        batch_size=batch_size,
        imgsz=imgsz,
        calib_samples=calib_samples,
        calib_seed=42,
        quant_preapplied=False,
        enable_tf32=False,
    )
    build_dur = time.time() - t0
    if built is None:
        print(f"  BUILD FAIL: {err}")
        return {"label": label, "status": "CRASH_BUILD", "error": err[:400]}
    print(f"  engine built in {build_dur:.1f}s  → {engine_path.name}  "
          f"({engine_path.stat().st_size/1e6:.2f}MB)")

    # Bench
    try:
        fwd, engine = _make_trt_forward(engine_path, batch_size, imgsz)
    except Exception as e:  # noqa: BLE001
        return {"label": label, "status": "CRASH_LOAD", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    try:
        stats = measure_latency(fwd, warmup_iters=100, measure_iters=100)
    except Exception as e:  # noqa: BLE001
        return {"label": label, "status": "CRASH_MEASURE", "error": f"{type(e).__name__}: {str(e)[:200]}"}

    fps = throughput_from_latency(stats["p50"], batch_size)
    r = {
        "label": label, "status": "OK",
        "onnx": onnx_path.name, "engine_file": engine_path.name,
        "dtype": dtype, "sparsity": sparsity,
        "batch_size": batch_size,
        "engine_size_mb": engine_path.stat().st_size / 1e6,
        "build_duration_s": build_dur,
        "p50_ms": stats["p50"], "p95_ms": stats.get("p95"), "p99_ms": stats.get("p99"),
        "stddev_ms": stats.get("stddev_ms"),
        "p50_gpu_ms": stats.get("p50_gpu"), "p95_gpu_ms": stats.get("p95_gpu"),
        "p99_gpu_ms": stats.get("p99_gpu"),
        "fps_bs1": fps,
        "peak_gpu_mem_mb_torch": stats.get("peak_gpu_mem_mb"),
        "peak_gpu_mem_mb_nvml_delta": stats.get("peak_gpu_mem_mb_nvml_delta"),
    }
    print(f"  p50={stats['p50']:.3f}ms  p95={stats.get('p95') or 0:.3f}ms  p99={stats.get('p99') or 0:.3f}ms  "
          f"stddev={stats.get('stddev_ms') or 0:.3f}ms")
    print(f"  p50_gpu={stats.get('p50_gpu') or 0:.3f}ms (pure CUDA kernel)")
    print(f"  fps_bs1={fps:.1f}  "
          f"mem torch={stats.get('peak_gpu_mem_mb') or 0:.1f}MB  "
          f"NVML delta={stats.get('peak_gpu_mem_mb_nvml_delta') or 0:.1f}MB")
    return r


def main() -> int:
    results = {}
    for key, spec in VARIANTS.items():
        results[key] = build_and_bench(key, spec)

    print("\n" + "=" * 105)
    print("PHASE 7 SUMMARY - fair bench (run_trt.py protocol, warmup=100 / measure=100)")
    print("=" * 105)
    print(f"{'variant':<24} {'p50 ms':<9} {'p50_gpu':<10} {'fps':<8} "
          f"{'peak_torch':<12} {'NVML delta':<12} {'engine MB':<11}")
    for k, r in results.items():
        if r.get("status") == "OK":
            print(f"{k:<24} {r['p50_ms']:<9.3f} "
                  f"{(r['p50_gpu_ms'] or 0):<10.3f} "
                  f"{r['fps_bs1']:<8.1f} "
                  f"{(r['peak_gpu_mem_mb_torch'] or 0):<12.1f} "
                  f"{(r['peak_gpu_mem_mb_nvml_delta'] or 0):<12.1f} "
                  f"{r['engine_size_mb']:<11.2f}")
        else:
            print(f"{k:<24} {r.get('status', '?')}  {r.get('error', '')[:60]}")

    print("\nREFERENCE (report_qr.md, 동일 프로토콜):")
    print(f"  {'modelopt_int8_entropy':<24} p50=1.31ms  fps=763.9   mem=38 MB   mAP=0.987")
    print(f"  {'modelopt_int8_sparsity':<24} p50=2.28ms  fps=439.4   mem=38 MB   mAP=0.987")
    print(f"  {'trt_int8_sparsity':<24} p50=1.54ms  fps=649.2   mem=38 MB   mAP=0.973 (drop 1.50%p)")
    print(f"  {'trt_fp16':<24} p50=2.30ms  fps=435.1   mem=38 MB   mAP=0.989")

    out_json = REPO_ROOT / "logs" / "wave10_p7_fair_bench.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[p7] saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
