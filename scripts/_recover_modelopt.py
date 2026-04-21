"""Recover modelopt_sparsify / modelopt_qat trained_weights from last.pt.

Windows paging file OSError prevented mto.save() during finalize() for
#11 and #17. ultralytics saved last.pt (EMA snapshot) successfully; we
re-run mto.save on that EMA model to produce trained_weights/*.pt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import modelopt.torch.opt as mto

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("runs/detect/modelopt_int8_sparsity/weights/last.pt",
     "trained_weights/modelopt_int8_sparsity.pt"),
    ("runs/detect/modelopt_int8_qat/weights/last.pt",
     "trained_weights/modelopt_int8_qat.pt"),
]


def _extract_model(ckpt: dict) -> torch.nn.Module:
    """ultralytics last.pt saves model=None during training; EMA holds weights."""
    ema = ckpt.get("ema")
    if ema is not None:
        # ultralytics ModelEMA wrapper may expose .ema as the actual model
        inner = getattr(ema, "ema", None)
        return inner if inner is not None else ema
    model = ckpt.get("model")
    if model is None:
        raise RuntimeError("Neither 'ema' nor 'model' in checkpoint")
    return model


def main() -> int:
    for src_rel, dst_rel in CASES:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        if not src.exists():
            print(f"[recover] SKIP: {src} (missing)")
            continue
        if dst.exists():
            print(f"[recover] SKIP: {dst} (already exists)")
            continue

        print(f"[recover] loading {src}")
        ckpt = torch.load(str(src), weights_only=False, map_location="cpu")
        model = _extract_model(ckpt)
        print(f"[recover] model type: {type(model).__name__}, "
              f"params={sum(p.numel() for p in model.parameters())}")

        # mto.save requires modelopt state present on model.
        # The EMA is a deepcopy of trainer.model, which was modified by
        # ms.sparsify / mtq.quantize in on_train_start callback.
        dst.parent.mkdir(parents=True, exist_ok=True)
        mto.save(model, str(dst))
        print(f"[recover] saved → {dst} ({dst.stat().st_size // 1024}KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
