"""Common training loop shared by all modifier plugins.

Flow:
1. Load base YOLO from ``spec.base_checkpoint``.
2. Call modifier.apply(yolo, spec).
3. Run ``ultralytics YOLO.train(...)`` with recipe params (amp toggled
   per modifier to avoid modelopt fake-quant / AMP corruption).
4. Call modifier.finalize(yolo, spec, out_pt) to serialize. The in-memory
   ``yolo.model`` is used directly — we do NOT load ``last.pt`` so
   ultralytics' EMA-based best selection cannot leak validation data.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]

if TYPE_CHECKING:
    from ultralytics import YOLO
    from scripts._schemas import Recipe, TrainingSpec


# Modifiers that wrap model in modelopt fake-quant / sparsity modules.
# AMP (fp16 mixed precision) breaks modelopt scale correctness, so disable.
_MODELOPT_MODIFIERS = {"modelopt_sparsify", "modelopt_qat"}


def _load_yolo(path: str) -> "YOLO":
    from ultralytics import YOLO
    return YOLO(path)


def _resolve_base_checkpoint(spec: "TrainingSpec") -> Path:
    p = Path(spec.base_checkpoint)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"base_checkpoint not found: {p}. See README for best_qr.pt "
            f"placement."
        )
    return p


def _resolve_data_yaml(spec: "TrainingSpec") -> str:
    if spec.data_yaml:
        p = Path(spec.data_yaml)
        if not p.is_absolute():
            p = ROOT / p
        return str(p)
    env = os.environ.get("OMNI_TRAIN_YAML") or os.environ.get("OMNI_COCO_YAML")
    if env:
        return env
    return str(ROOT / "qr_barcode.yaml")


def _run_ultralytics_train(yolo: "YOLO", spec: "TrainingSpec",
                           run_name: str) -> Path:
    """Call ultralytics model.train(). Returns path to last.pt (unused for
    modelopt; kept for potential debugging). See spec §5.3 for why we
    don't consume this file."""
    smoke = os.environ.get("OMNI_TRAIN_SMOKE") == "1"
    kwargs = dict(
        data=_resolve_data_yaml(spec),
        epochs=1 if smoke else spec.epochs,
        batch=spec.batch,
        workers=int(os.environ.get("OMNI_TRAIN_WORKERS", spec.workers)),
        imgsz=spec.imgsz,
        lr0=spec.lr0,
        optimizer=spec.optimizer,
        seed=spec.seed,
        device=os.environ.get("OMNI_TRAIN_DEVICE", "0"),
        name=run_name,
        exist_ok=True,
        amp=spec.modifier not in _MODELOPT_MODIFIERS,
        verbose=True,
    )
    if smoke:
        kwargs["fraction"] = 0.1
    yolo.train(**kwargs)
    return ROOT / "runs" / "train" / run_name / "weights" / "last.pt"


def _load_modifier(name: str):
    return importlib.import_module(f"scripts._modifiers.{name}")


def train_with_modifier(recipe: "Recipe") -> Path:
    spec = recipe.technique.training
    if spec is None:
        raise ValueError(f"recipe {recipe.name} has no training section")
    modifier = _load_modifier(spec.modifier)

    base = _resolve_base_checkpoint(spec)
    print(f"[train] loading base: {base}")
    yolo = _load_yolo(str(base))

    print(f"[train] applying modifier: {spec.modifier}")
    modifier.apply(yolo, spec)

    run_name = recipe.name
    print(f"[train] ultralytics model.train(epochs={spec.epochs}, "
          f"amp={spec.modifier not in _MODELOPT_MODIFIERS}) → runs/train/{run_name}")
    started = time.time()
    _run_ultralytics_train(yolo, spec, run_name)
    duration = time.time() - started

    out_dir = ROOT / "trained_weights"
    out_dir.mkdir(exist_ok=True)
    out_pt = out_dir / f"{recipe.name}.pt"

    print(f"[train] calling {spec.modifier}.finalize(out_pt={out_pt})")
    modifier.finalize(yolo, spec, out_pt)

    _write_train_json(out_dir / f"{recipe.name}.train.json", recipe, duration)
    return out_pt


def _write_train_json(path: Path, recipe: "Recipe", duration_s: float) -> None:
    spec = recipe.technique.training
    assert spec is not None
    path.write_text(json.dumps({
        "recipe": recipe.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(duration_s, 1),
        "base_checkpoint": spec.base_checkpoint,
        "epochs": spec.epochs,
        "modifier": spec.modifier,
        "lr0": spec.lr0,
        "amp": spec.modifier not in _MODELOPT_MODIFIERS,
        "notes": "val mAP during training is EMA-based and unreliable for "
                 "modelopt modifiers; use run_trt.py for authoritative eval",
    }, indent=2))
