"""Phase 1 — FastNAS pruned 모델 QR fine-tune.

입력:   trained_weights/_spike_wave10_pruned_ult.pt  (ratio 0.843)
출력:   trained_weights/23_fastnas_p1_finetune/weights/best.pt
목표:   mAP@0.5 ~0.98 복원 (baseline best_qr 수준)
"""

from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent.parent
PRUNED_PT = REPO_ROOT / "trained_weights" / "_spike_wave10_pruned_ult.pt"
DATA_YAML = REPO_ROOT / "qr_barcode.yaml"


def main() -> int:
    assert PRUNED_PT.exists(), f"missing {PRUNED_PT}"
    yolo = YOLO(str(PRUNED_PT))
    print(f"[p1] loaded pruned model from {PRUNED_PT}")
    print(f"[p1] model type: {type(yolo.model).__name__}")

    # QAT recipe (#17) 와 동일 스케줄 — lr0=5e-4, AdamW, 60 epochs
    results = yolo.train(
        data=str(DATA_YAML),
        epochs=60,
        imgsz=640,
        batch=8,
        device="cuda",
        optimizer="AdamW",
        lr0=5e-4,
        project=str(REPO_ROOT / "trained_weights"),
        name="23_fastnas_p1_finetune",
        exist_ok=True,
        verbose=False,
        plots=False,
        save=True,
        val=True,
        workers=0,
        seed=42,
        patience=15,
    )

    print(f"[p1] best mAP50: {float(results.box.map50):.4f}")
    print(f"[p1] best.pt: {REPO_ROOT / 'trained_weights' / '23_fastnas_p1_finetune' / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
