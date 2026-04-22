"""Phase 4C — sparsity fine-tune + bench (B + C variants).

Phase 4B에서 2:4 mask 주입만으로는 mAP=0.0 확인. 학습 파이프라인에
sparsity-preserving callback을 추가해 fine-tune.

Flow:
  B: fastnas + 2:4 sparsity mask 주입 → 30 epoch fine-tune (매 batch 후 mask 재적용)
     → mask 재적용 (safety) → FP16 engine → bench
  C: B의 최종 fine-tuned + masked model → INT8 engine (TRT native calibrator)
     → bench

출력:
  trained_weights/23_fastnas_chain_ft/B_sparsity/weights/best.pt
  trained_weights/23_fastnas_chain_ft/B_final.pt
  trained_weights/23_fastnas_chain_ft/{B,C}.engine
  logs/wave10_p4c_chain.json
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
SPARSITY_PT = REPO_ROOT / "trained_weights" / "23_fastnas_chain" / "fastnas_sparsity.pt"
DATA_YAML = REPO_ROOT / "qr_barcode.yaml"
OUT_DIR = REPO_ROOT / "trained_weights" / "23_fastnas_chain_ft"


def collect_masks(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """각 Conv2d 의 2:4 mask 를 name → mask tensor 로 수집 + 현재 weight에도 적용."""
    masks: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) and m.weight.shape[1] >= 4:
            w = m.weight.data
            shape = w.shape
            flat = w.reshape(-1, 4)
            _, idx = flat.abs().topk(2, dim=1, largest=False)
            mask = torch.ones_like(flat)
            mask.scatter_(1, idx, 0)
            mask_shaped = mask.reshape(shape)
            masks[name] = mask_shaped.clone()
            m.weight.data = w * mask_shaped
    return masks


def make_sparsity_callback(masks: dict[str, torch.Tensor]):
    """매 batch 이후 mask 재적용 — gradient update로 복귀한 dense weight 제거."""
    def cb(trainer):
        # trainer.model 과 trainer.ema.ema 둘 다 적용
        for name, m in trainer.model.named_modules():
            if name in masks and hasattr(m, "weight"):
                m.weight.data.mul_(masks[name].to(m.weight.device))
        ema = getattr(trainer, "ema", None)
        if ema is not None and getattr(ema, "ema", None) is not None:
            for name, m in ema.ema.named_modules():
                if name in masks and hasattr(m, "weight"):
                    m.weight.data.mul_(masks[name].to(m.weight.device))
    return cb


def sparsity_fraction(model: torch.nn.Module) -> tuple[int, int, float]:
    total = 0
    zeros = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) and m.weight.shape[1] >= 4:
            total += m.weight.numel()
            zeros += (m.weight == 0).sum().item()
    return total, zeros, (zeros / total if total else 0.0)


def save_pickle(yolo: YOLO, out_pt: Path) -> None:
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    yolo.model.cpu()
    ckpt = {
        "model": yolo.model,
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
    renamed = pickle_path.parent / f"{pickle_path.stem}_{'int8' if int8 else 'fp16'}.engine"
    if renamed.exists():
        renamed.unlink()
    shutil.move(str(engine_pt), str(renamed))
    print(f"  engine: {renamed.name}  ({renamed.stat().st_size/1e6:.2f}MB)")

    yolo_e = YOLO(str(renamed))
    try:
        metrics = yolo_e.val(
            data=str(DATA_YAML), imgsz=640, batch=1, device=0,
            half=not int8, int8=int8,
            verbose=False, plots=False, save=False, workers=0,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  val CRASH: {type(e).__name__}: {e}")
        return {"label": label, "status": "CRASH_VAL", "error": f"{type(e).__name__}: {str(e)[:200]}",
                "engine_size_mb": renamed.stat().st_size / 1e6}

    speed = metrics.speed
    inf_ms = speed.get("inference", 0.0)
    fps = (1000.0 / inf_ms) if inf_ms > 0 else 0.0
    total_ms = sum(speed.get(k, 0) for k in ("preprocess", "inference", "postprocess"))
    fps_e2e = (1000.0 / total_ms) if total_ms > 0 else 0.0
    map50 = float(metrics.box.map50)
    print(f"  mAP50={map50:.4f}  inf={inf_ms:.2f}ms  fps={fps:.1f}  e2e={total_ms:.2f}ms/{fps_e2e:.1f}fps")
    return {
        "label": label, "status": "OK",
        "engine_file": renamed.name,
        "engine_size_mb": renamed.stat().st_size / 1e6,
        "map50": map50, "map50_95": float(metrics.box.map),
        "inference_ms": inf_ms, "e2e_ms": total_ms,
        "fps_inference": fps, "fps_e2e": fps_e2e,
    }


def main() -> int:
    assert SPARSITY_PT.exists(), f"missing {SPARSITY_PT}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- B: sparsity fine-tune -----
    print("========== (B) fastnas + sparsity fine-tune ==========")
    yolo = YOLO(str(SPARSITY_PT))
    yolo.model.cuda()
    masks = collect_masks(yolo.model)
    _, _, sp_initial = sparsity_fraction(yolo.model)
    print(f"[B] collected {len(masks)} masks  initial Conv sparsity = {sp_initial:.3f}")

    cb = make_sparsity_callback(masks)
    yolo.add_callback("on_train_batch_end", cb)
    yolo.add_callback("on_fit_epoch_end", cb)

    yolo.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=8,
        device="cuda",
        optimizer="AdamW",
        lr0=5e-4,
        project=str(OUT_DIR),
        name="B_sparsity",
        exist_ok=True,
        verbose=False,
        plots=False,
        save=True,
        val=True,
        workers=0,
        seed=42,
        patience=10,
    )
    best_pt = OUT_DIR / "B_sparsity" / "weights" / "best.pt"
    print(f"[B] fine-tune done. best.pt: {best_pt}")

    # reload best.pt + 최종 mask 재적용 (safety)
    yolo_b = YOLO(str(best_pt))
    yolo_b.model.cuda()
    # 주의: fine-tune 후 weight 가 조금 변했으므로 mask 는 재계산
    masks_b = collect_masks(yolo_b.model)
    _, _, sp_b = sparsity_fraction(yolo_b.model)
    print(f"[B] post-finetune re-masked  sparsity = {sp_b:.3f}  ({len(masks_b)} layers)")

    final_b = OUT_DIR / "B_final.pt"
    save_pickle(yolo_b, final_b)
    print(f"[B] saved {final_b}  ({final_b.stat().st_size/1e6:.2f}MB)")

    results = {}
    results["B_fastnas_sparsity_ft"] = export_and_bench(final_b, "B_sparsity_fp16", int8=False)
    torch.cuda.empty_cache()

    # ----- C: B의 fine-tuned 모델 + INT8 -----
    print("\n========== (C) chain sparsity + INT8 ==========")
    results["C_fastnas_sp_int8_ft"] = export_and_bench(final_b, "C_sp_int8", int8=True)

    # Summary
    print("\n========== PHASE 4C SUMMARY ==========")
    print(f"{'variant':<28} {'mAP50':<8} {'inf ms':<9} {'fps (inf)':<11} {'size MB':<8}")
    for k, r in results.items():
        if r.get("status") == "OK":
            print(f"{k:<28} {r['map50']:<8.4f} {r['inference_ms']:<9.2f} "
                  f"{r['fps_inference']:<11.1f} {r['engine_size_mb']:<8.2f}")
        else:
            print(f"{k:<28} CRASH — {r.get('error', '')[:70]}")

    out_json = REPO_ROOT / "logs" / "wave10_p4c_chain.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[p4c] saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
