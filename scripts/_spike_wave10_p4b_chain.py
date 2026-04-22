"""Phase 4B — FastNAS + sparsity / INT8 / chain (modelopt ONNX segfault 우회).

modelopt 의 QuantConv2d 가 torch.onnx.export 에서 C-level crash (exit 139).
우회: ultralytics YOLO.export(format="engine", int8=True) 의 TRT native
entropy calibrator 사용 (modelopt INT8 entropy 와 알고리즘 동일). Sparsity 는
2:4 mask 직접 주입 (Phase 3 helper) — QuantConv2d 없어 pickle/ONNX 안전.

variants:
  (A) fastnas_int8       → Phase 1B trt_int8 결과 재사용 (이미 측정)
  (B) fastnas_sparsity   → 2:4 mask 주입 + TRT FP16 engine
  (C) fastnas_sp_int8    → 2:4 mask 주입 + TRT INT8 engine (calibration)

출력: trained_weights/23_fastnas_chain/{B,C}*.engine, logs/wave10_p4b_chain.json
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
FASTNAS_BEST = REPO_ROOT / "trained_weights" / "23_fastnas_p1_finetune" / "weights" / "best.pt"
DATA_YAML = REPO_ROOT / "qr_barcode.yaml"
OUT_DIR = REPO_ROOT / "trained_weights" / "23_fastnas_chain"


def apply_2x4_sparsity(model: torch.nn.Module) -> int:
    """2:4 structured sparsity — 4개 weight 중 L1 작은 2개 = 0."""
    count = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) and m.weight.shape[1] >= 4:
            w = m.weight.data
            shape = w.shape
            w_flat = w.reshape(-1, 4)
            _, idx = w_flat.abs().topk(2, dim=1, largest=False)
            mask = torch.ones_like(w_flat)
            mask.scatter_(1, idx, 0)
            m.weight.data = (w_flat * mask).reshape(shape)
            count += 1
    return count


def save_ultralytics_pickle(yolo: YOLO, out_pt: Path) -> None:
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": yolo.model.cpu(),
        "train_args": dict(getattr(yolo.model, "args", {})) if hasattr(yolo.model, "args") else {},
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, str(out_pt))
    yolo.model.cuda()


def export_and_bench(pickle_path: Path, label: str, *, int8: bool) -> dict:
    print(f"\n=== {label} engine export + bench ===")
    yolo = YOLO(str(pickle_path))
    try:
        engine_path = yolo.export(
            format="engine",
            imgsz=640,
            half=not int8,
            int8=int8,
            dynamic=False,
            batch=1,
            data=str(DATA_YAML) if int8 else None,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  engine build CRASH: {type(e).__name__}: {e}")
        return {"label": label, "status": "CRASH_ENGINE", "error": f"{type(e).__name__}: {str(e)[:200]}"}

    engine_pt = Path(engine_path)
    engine_renamed = pickle_path.parent / f"{pickle_path.stem}_{'int8' if int8 else 'fp16'}.engine"
    if engine_renamed.exists():
        engine_renamed.unlink()
    shutil.move(str(engine_pt), str(engine_renamed))
    print(f"  engine: {engine_renamed.name}  ({engine_renamed.stat().st_size/1e6:.2f}MB)")

    yolo_e = YOLO(str(engine_renamed))
    try:
        metrics = yolo_e.val(
            data=str(DATA_YAML), imgsz=640, batch=1, device=0,
            half=not int8, int8=int8,
            verbose=False, plots=False, save=False, workers=0,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  val CRASH: {type(e).__name__}: {e}")
        return {"label": label, "status": "CRASH_VAL",
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "engine_size_mb": engine_renamed.stat().st_size / 1e6}

    speed = metrics.speed
    inf_ms = speed.get("inference", 0.0)
    fps = (1000.0 / inf_ms) if inf_ms > 0 else 0.0
    total_ms = sum(speed.get(k, 0) for k in ("preprocess", "inference", "postprocess"))
    fps_e2e = (1000.0 / total_ms) if total_ms > 0 else 0.0
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    print(f"  mAP50={map50:.4f}  inf={inf_ms:.2f}ms  fps={fps:.1f}  e2e={total_ms:.2f}ms/{fps_e2e:.1f}fps")
    return {
        "label": label, "status": "OK",
        "engine_file": engine_renamed.name,
        "engine_size_mb": engine_renamed.stat().st_size / 1e6,
        "map50": map50, "map50_95": map50_95,
        "inference_ms": inf_ms, "e2e_ms": total_ms,
        "fps_inference": fps, "fps_e2e": fps_e2e,
    }


def main() -> int:
    assert FASTNAS_BEST.exists(), f"missing {FASTNAS_BEST}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # (A) FastNAS + INT8 — Phase 1B trt_int8 결과 재사용 (이미 측정)
    p1b = REPO_ROOT / "logs" / "wave10_p1b_bench.json"
    if p1b.exists():
        r = json.loads(p1b.read_text())
        results["A_fastnas_int8"] = {
            "source": "Phase 1B trt_int8 (재사용)",
            "engine_file": r["trt_int8"].get("file", ""),
            "engine_size_mb": r["trt_int8"].get("size_mb"),
            "map50": r["trt_int8"].get("map50"),
            "map50_95": r["trt_int8"].get("map50_95"),
            "inference_ms": r["trt_int8"].get("inference_ms"),
            "fps_inference": r["trt_int8"].get("fps_bs1_inference"),
            "fps_e2e": r["trt_int8"].get("fps_bs1_e2e"),
            "note": "ultralytics export(int8=True) — TRT native entropy calibrator, 알고리즘적으로 modelopt INT8 entropy 와 동일",
        }
        print(f"[A] re-used trt_int8 from Phase 1B: mAP50={results['A_fastnas_int8']['map50']:.4f}  "
              f"fps={results['A_fastnas_int8']['fps_inference']:.1f}")
    else:
        results["A_fastnas_int8"] = {"status": "MISSING", "error": "Phase 1B bench JSON 부재"}

    # (B) FastNAS + sparsity — 2:4 mask 주입 + FP16 engine
    print("\n========== (B) fastnas_sparsity ==========")
    yolo = YOLO(str(FASTNAS_BEST))
    model = yolo.model.cuda().eval()
    with torch.no_grad():
        _ = model(torch.randn(1, 3, 640, 640, device="cuda"))
    count = apply_2x4_sparsity(yolo.model)
    print(f"  [sparsity] 2:4 mask applied to {count} Conv layers")
    pickle_b = OUT_DIR / "fastnas_sparsity.pt"
    save_ultralytics_pickle(yolo, pickle_b)
    print(f"  saved {pickle_b.name}  ({pickle_b.stat().st_size/1e6:.2f}MB)")
    results["B_fastnas_sparsity"] = export_and_bench(pickle_b, "fastnas_sparsity_fp16", int8=False)
    torch.cuda.empty_cache()

    # (C) FastNAS + sparsity + INT8 — 같은 sparsified 모델, int8=True engine
    print("\n========== (C) fastnas_sp_int8 ==========")
    # 동일 pickle 재사용 — ultralytics int8 export가 내부 calibration
    results["C_fastnas_sp_int8"] = export_and_bench(pickle_b, "fastnas_sp_int8", int8=True)

    # Summary
    print("\n========== PHASE 4B SUMMARY ==========")
    print(f"{'variant':<25} {'mAP50':<8} {'inf ms':<9} {'fps (inf)':<11} {'size MB':<8}")
    for k, r in results.items():
        if r.get("status") in ("OK", None) and "map50" in r:
            print(f"{k:<25} {r['map50']:<8.4f} {r.get('inference_ms', 0):<9.2f} "
                  f"{r.get('fps_inference', 0):<11.1f} {r.get('engine_size_mb', 0):<8.2f}")
        else:
            print(f"{k:<25} CRASH/MISSING — {r.get('error', r.get('status', ''))[:70]}")

    print("\nREFERENCE (report_qr.md baselines):")
    print("  modelopt_int8_entropy  mAP=0.987  fps_bs1=763.9  (run_trt.py protocol)")
    print("  modelopt_int8_sparsity mAP=0.987  fps_bs1=439.4  (run_trt.py protocol)")

    out_json = REPO_ROOT / "logs" / "wave10_p4b_chain.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[p4b] saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
