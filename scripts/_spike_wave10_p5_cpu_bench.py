"""Phase 5 — FastNAS CPU bench (OpenVINO FP32/INT8, ORT CPU FP32).

입력:
  (base)      trained_weights/23_fastnas_p1_finetune/weights/best.pt
  (sparsity)  trained_weights/23_fastnas_chain_ft/B_final.pt

출력:
  trained_weights/23_fastnas_p1_finetune/weights/best_openvino_model/
  trained_weights/23_fastnas_p1_finetune/weights/best_int8_openvino_model/
  trained_weights/23_fastnas_chain_ft/B_final_openvino_model/
  trained_weights/23_fastnas_chain_ft/B_final_int8_openvino_model/
  logs/wave10_p5_cpu_bench.json

baseline (report_cpu_qr.md):
  openvino_int8_nncf  fps_bs1=23.9  mAP=0.988
  openvino_fp32       fps_bs1=18.6  mAP=0.988
  ort_cpu_fp32        fps_bs1=14.4  mAP=0.988
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = REPO_ROOT / "qr_barcode.yaml"

MODELS = {
    "fastnas_base": REPO_ROOT / "trained_weights" / "23_fastnas_p1_finetune" / "weights" / "best.pt",
    "fastnas_sparsity": REPO_ROOT / "trained_weights" / "23_fastnas_chain_ft" / "B_final.pt",
}


def val_cpu(model_path: Path, label: str, *, half: bool = False, int8: bool = False) -> dict:
    yolo = YOLO(str(model_path))
    t0 = time.time()
    try:
        metrics = yolo.val(
            data=str(DATA_YAML),
            imgsz=640,
            batch=1,
            device="cpu",
            half=half,
            int8=int8,
            verbose=False,
            plots=False,
            save=False,
            workers=0,
        )
    except Exception as e:  # noqa: BLE001
        return {"label": label, "status": "CRASH_VAL", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    dur = time.time() - t0
    speed = metrics.speed
    inf_ms = speed.get("inference", 0.0)
    fps = (1000.0 / inf_ms) if inf_ms > 0 else 0.0
    total_ms = sum(speed.get(k, 0) for k in ("preprocess", "inference", "postprocess"))
    fps_e2e = (1000.0 / total_ms) if total_ms > 0 else 0.0
    map50 = float(metrics.box.map50)
    print(f"  [{label}] mAP50={map50:.4f}  inf={inf_ms:.2f}ms  fps={fps:.2f}  e2e={fps_e2e:.2f}fps  (val={dur:.1f}s)")
    return {
        "label": label, "status": "OK",
        "model_file": str(model_path.relative_to(REPO_ROOT)),
        "map50": map50, "map50_95": float(metrics.box.map),
        "inference_ms": inf_ms, "e2e_ms": total_ms,
        "fps_inference": fps, "fps_e2e": fps_e2e,
        "val_duration_s": dur,
    }


def bench_openvino(pt_path: Path, label: str, *, int8: bool) -> dict:
    print(f"\n=== {label} ===  ({pt_path.stem})")
    yolo = YOLO(str(pt_path))
    try:
        ov_dir = yolo.export(
            format="openvino",
            imgsz=640,
            int8=int8,
            half=False,
            dynamic=False,
            batch=1,
            data=str(DATA_YAML) if int8 else None,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  export CRASH: {type(e).__name__}: {e}")
        return {"label": label, "status": "CRASH_EXPORT", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    ov_path = Path(ov_dir)
    size_mb = sum(p.stat().st_size for p in ov_path.rglob("*") if p.is_file()) / 1e6
    print(f"  OV dir: {ov_path}  ({size_mb:.2f}MB)")
    r = val_cpu(ov_path, label, int8=int8)
    r["ov_dir_size_mb"] = size_mb
    return r


def bench_ort_onnx(pt_path: Path, label: str) -> dict:
    """ultralytics는 .onnx 파일을 val에서 로드할 때 onnxruntime CPUExecutionProvider 사용."""
    print(f"\n=== {label} ===  ({pt_path.stem})")
    yolo = YOLO(str(pt_path))
    try:
        onnx_path = yolo.export(format="onnx", imgsz=640, dynamic=False, simplify=True, half=False, batch=1)
    except Exception as e:  # noqa: BLE001
        print(f"  export CRASH: {type(e).__name__}: {e}")
        return {"label": label, "status": "CRASH_EXPORT", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    onnx_pt = Path(onnx_path)
    print(f"  ONNX: {onnx_pt.name}  ({onnx_pt.stat().st_size/1e6:.2f}MB)")
    r = val_cpu(onnx_pt, label)
    r["onnx_size_mb"] = onnx_pt.stat().st_size / 1e6
    return r


def main() -> int:
    results = {}
    print("=" * 60)
    print("REFERENCE (report_cpu_qr.md)")
    print(f"  openvino_int8_nncf  fps_bs1=23.9  mAP=0.988")
    print(f"  openvino_fp32       fps_bs1=18.6  mAP=0.988")
    print(f"  ort_cpu_fp32        fps_bs1=14.4  mAP=0.988")
    print("=" * 60)

    for model_key, pt_path in MODELS.items():
        assert pt_path.exists(), f"missing {pt_path}"
        print(f"\n{'#' * 10} model = {model_key} {'#' * 10}")
        print(f"  path: {pt_path.relative_to(REPO_ROOT)}")
        print(f"  size: {pt_path.stat().st_size/1e6:.2f}MB")

        key_ov_fp32 = f"{model_key}__openvino_fp32"
        key_ov_int8 = f"{model_key}__openvino_int8_nncf"
        key_ort_fp32 = f"{model_key}__ort_cpu_fp32"

        results[key_ov_fp32] = bench_openvino(pt_path, key_ov_fp32, int8=False)
        results[key_ov_int8] = bench_openvino(pt_path, key_ov_int8, int8=True)
        results[key_ort_fp32] = bench_ort_onnx(pt_path, key_ort_fp32)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 5 SUMMARY")
    print("=" * 60)
    print(f"{'variant':<40} {'mAP50':<8} {'inf ms':<9} {'fps':<8} {'size MB':<8}")
    for k, r in results.items():
        if r.get("status") == "OK":
            size = r.get("ov_dir_size_mb") or r.get("onnx_size_mb") or 0
            print(f"{k:<40} {r['map50']:<8.4f} {r['inference_ms']:<9.2f} "
                  f"{r['fps_inference']:<8.2f} {size:<8.2f}")
        else:
            print(f"{k:<40} CRASH — {r.get('error', '')[:70]}")

    print("\nbaseline (report_cpu_qr.md):")
    print(f"  openvino_int8_nncf   mAP=0.988  fps=23.9")
    print(f"  openvino_fp32        mAP=0.988  fps=18.6")
    print(f"  ort_cpu_fp32         mAP=0.988  fps=14.4")

    out_json = REPO_ROOT / "logs" / "wave10_p5_cpu_bench.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[p5] saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
