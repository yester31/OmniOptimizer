"""Phase 1B — fine-tuned FastNAS pruned 모델 TRT FP16/INT8 engine build + bench.

입력: trained_weights/23_fastnas_p1_finetune/weights/best.pt (mAP50 0.9475)
출력:
  - best.onnx / best.engine (FP16) / best_int8.engine (INT8)
  - logs/wave10_p1b_bench.json  (fps, mAP, latency)

비교 대상: report_qr.md 의 modelopt_int8_entropy (fps 763.9 bs1, mAP 0.987).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent.parent
BEST_PT = REPO_ROOT / "trained_weights" / "23_fastnas_p1_finetune" / "weights" / "best.pt"
WORK_DIR = REPO_ROOT / "trained_weights" / "23_fastnas_p1_finetune" / "weights"
DATA_YAML = REPO_ROOT / "qr_barcode.yaml"


def run_val_bench(model_path: Path, label: str, *, half: bool = False, int8: bool = False) -> dict:
    print(f"\n[p1b] === {label} bench ===  ({model_path.name})")
    yolo = YOLO(str(model_path))
    t0 = time.time()
    metrics = yolo.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=1,
        device=0,
        half=half,
        int8=int8,
        verbose=False,
        plots=False,
        save=False,
        workers=0,
    )
    dur = time.time() - t0
    speed = metrics.speed  # {'preprocess', 'inference', 'loss', 'postprocess'}
    inf_ms = speed.get("inference", 0.0)
    fps_bs1 = (1000.0 / inf_ms) if inf_ms > 0 else 0.0
    total_ms = sum([speed.get(k, 0) for k in ("preprocess", "inference", "postprocess")])
    fps_e2e = (1000.0 / total_ms) if total_ms > 0 else 0.0
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    print(f"  mAP@0.5 = {map50:.4f}  mAP@0.5:0.95 = {map50_95:.4f}")
    print(f"  inference = {inf_ms:.2f} ms  →  fps_bs1 (inference only) = {fps_bs1:.1f}")
    print(f"  e2e total = {total_ms:.2f} ms  →  fps_e2e = {fps_e2e:.1f}")
    return {
        "label": label,
        "file": model_path.name,
        "size_mb": model_path.stat().st_size / 1e6,
        "map50": map50,
        "map50_95": map50_95,
        "inference_ms": inf_ms,
        "fps_bs1_inference": fps_bs1,
        "e2e_ms": total_ms,
        "fps_bs1_e2e": fps_e2e,
        "speed_breakdown": speed,
        "bench_duration_s": dur,
    }


def main() -> int:
    assert BEST_PT.exists(), f"missing {BEST_PT}"
    print(f"[p1b] best.pt: {BEST_PT}  ({BEST_PT.stat().st_size/1e6:.2f}MB)")

    results = {"baseline_reference": {
        "recipe": "modelopt_int8_entropy",
        "fps_bs1": 763.9,
        "fps_bs8": 1078.5,
        "map50": 0.987,
        "source": "report_qr.md rank 1",
    }}

    # 1) PyTorch val (fine-tuned 상태 기준 mAP 확인)
    print("\n[p1b] Step 1 — PyTorch val (FP32)")
    results["pytorch_fp32"] = run_val_bench(BEST_PT, "pytorch_fp32")

    # 2) ONNX export
    print("\n[p1b] Step 2 — ONNX export")
    yolo = YOLO(str(BEST_PT))
    onnx_path = yolo.export(format="onnx", imgsz=640, dynamic=False, half=False, simplify=True)
    onnx_pt = Path(onnx_path)
    print(f"  ONNX: {onnx_pt}  ({onnx_pt.stat().st_size/1e6:.2f}MB)")

    # 3) TRT FP16 engine export + bench
    print("\n[p1b] Step 3 — TRT FP16 engine build + bench")
    try:
        yolo = YOLO(str(BEST_PT))
        engine_path = yolo.export(format="engine", imgsz=640, half=True, dynamic=False, batch=1)
        engine_pt = Path(engine_path)
        print(f"  engine: {engine_pt}  ({engine_pt.stat().st_size/1e6:.2f}MB)")
        results["trt_fp16"] = run_val_bench(engine_pt, "trt_fp16", half=True)
    except Exception as e:  # noqa: BLE001
        print(f"  CRASH: {type(e).__name__}: {e}")
        results["trt_fp16"] = {"status": "CRASH", "error": f"{type(e).__name__}: {str(e)[:200]}"}

    # 4) TRT INT8 engine export + bench
    print("\n[p1b] Step 4 — TRT INT8 engine build + bench")
    try:
        yolo = YOLO(str(BEST_PT))
        # ultralytics 는 int8 export 시 자동으로 calibration dataset 사용
        # 기존 best.engine 을 INT8 engine 으로 덮어쓰지 않게 이름 분리 필요 → 이동
        int8_engine_path = yolo.export(
            format="engine", imgsz=640, int8=True, dynamic=False, batch=1,
            data=str(DATA_YAML),
        )
        int8_engine_pt = Path(int8_engine_path)
        # rename to avoid collision
        int8_renamed = int8_engine_pt.with_name("best_int8.engine")
        if int8_renamed.exists():
            int8_renamed.unlink()
        shutil.move(str(int8_engine_pt), str(int8_renamed))
        print(f"  engine: {int8_renamed}  ({int8_renamed.stat().st_size/1e6:.2f}MB)")
        results["trt_int8"] = run_val_bench(int8_renamed, "trt_int8", int8=True)
    except Exception as e:  # noqa: BLE001
        print(f"  CRASH: {type(e).__name__}: {e}")
        results["trt_int8"] = {"status": "CRASH", "error": f"{type(e).__name__}: {str(e)[:200]}"}

    # Summary
    print("\n========== PHASE 1B SUMMARY ==========")
    print(f"{'run':<16} {'mAP50':<8} {'inf ms':<10} {'fps (inf)':<12} {'fps (e2e)':<11}")
    for k in ("pytorch_fp32", "trt_fp16", "trt_int8"):
        r = results.get(k, {})
        if r.get("status") == "CRASH":
            print(f"{k:<16} CRASH — {r.get('error', '')[:60]}")
            continue
        print(f"{k:<16} {r.get('map50', 0):<8.4f} {r.get('inference_ms', 0):<10.2f} "
              f"{r.get('fps_bs1_inference', 0):<12.1f} {r.get('fps_bs1_e2e', 0):<11.1f}")

    print(f"\nBASELINE modelopt_int8_entropy: fps_bs1=763.9, mAP50=0.987")

    out_json = REPO_ROOT / "logs" / "wave10_p1b_bench.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[p1b] saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
