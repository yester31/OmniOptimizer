"""Deterministic calibration/eval split.

When calibration and evaluation draw from the same dataset, the INT8
calibration samples land inside the eval set, biasing absolute mAP up by
an amount proportional to (calibration_samples / |val|). For COCO
val2017 (5000 images) with 512 calib samples that is ~10%.

This module produces a derived dataset yaml whose ``val:`` listing
excludes the calibration subset so eval is overlap-free. The split is
only applied when the calibration yaml and the eval yaml point at the
same underlying file (i.e. same dataset). If they differ (e.g. QR eval
with COCO calibration), the eval yaml is returned unchanged.

Disable globally via ``OMNI_DISABLE_CALIB_EVAL_SPLIT=1``.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Optional


def resolve_val_image_paths(yaml_path: str) -> list[str]:
    """Parse an ultralytics dataset yaml and return absolute val image paths.
    Handles both .txt-listing (COCO) and directory (YOLOv5+) conventions."""
    import yaml as yaml_mod

    with open(yaml_path, "r", encoding="utf-8") as f:
        spec = yaml_mod.safe_load(f)
    root = Path(spec["path"])
    val_rel = spec["val"]
    val_target = (root / val_rel) if not Path(val_rel).is_absolute() else Path(val_rel)

    paths: list[str] = []
    if val_target.is_dir():
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for p in sorted(val_target.rglob("*")):
            if p.suffix.lower() in exts:
                paths.append(str(p.resolve()))
        return paths

    with open(val_target, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines:
        p = Path(ln)
        if not p.is_absolute():
            p = (root / ln).resolve()
        paths.append(str(p))
    return paths


def _same_file(a: str, b: str) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def eval_yaml(eval_yaml_path: Optional[str],
              calib_yaml_path: Optional[str] = None,
              calib_seed: int = 42,
              calib_n: int = 512) -> str:
    """Return the dataset yaml ultralytics should use for eval.

    If ``calib_yaml_path`` is the same file as ``eval_yaml_path`` (i.e.
    calibration and evaluation share a dataset), returns a derived yaml
    whose ``val:`` excludes the deterministic calibration subset
    (first ``calib_n`` images after ``random.Random(calib_seed).shuffle``
    on the resolved path list). The computation matches
    ``_build_calib_numpy`` in ``run_trt.py`` so the sets are disjoint by
    construction.

    Otherwise returns ``eval_yaml_path`` unchanged — different datasets
    cannot overlap.
    """
    if not eval_yaml_path:
        return eval_yaml_path or "coco.yaml"
    if os.environ.get("OMNI_DISABLE_CALIB_EVAL_SPLIT") == "1":
        return eval_yaml_path
    if calib_yaml_path is None:
        calib_yaml_path = os.environ.get("OMNI_CALIB_YAML") or eval_yaml_path
    if not _same_file(calib_yaml_path, eval_yaml_path):
        return eval_yaml_path

    try:
        all_paths = resolve_val_image_paths(eval_yaml_path)
    except Exception as e:
        print(f"[warn] calib/eval split: cannot resolve {eval_yaml_path} ({e}); "
              "evaluating on full val (overlap possible)",
              file=sys.stderr)
        return eval_yaml_path
    if len(all_paths) < calib_n * 2:
        print(f"[warn] calib/eval split: dataset has {len(all_paths)} val images "
              f"but calib_n={calib_n}; skipping split to avoid tiny eval set "
              "(OMNI_CALIB_YAML can point at a larger dataset for calibration)",
              file=sys.stderr)
        return eval_yaml_path

    rng = random.Random(calib_seed)
    shuffled = list(all_paths)
    rng.shuffle(shuffled)
    calib_set = set(shuffled[:calib_n])
    eval_paths = [p for p in all_paths if p not in calib_set]

    import yaml as yaml_mod

    work = Path("results/_calib_eval_split")
    work.mkdir(parents=True, exist_ok=True)
    base = Path(eval_yaml_path).stem
    tag = f"seed{calib_seed}_skip{calib_n}_of{len(all_paths)}"
    eval_txt = work / f"{base}_eval_{tag}.txt"
    if not eval_txt.exists():
        eval_txt.write_text("\n".join(eval_paths), encoding="utf-8")
    derived_yaml = work / f"{base}_eval_{tag}.yaml"
    if not derived_yaml.exists():
        with open(eval_yaml_path, "r", encoding="utf-8") as f:
            orig = yaml_mod.safe_load(f)
        derived = dict(orig)
        derived["val"] = str(eval_txt.resolve())
        with open(derived_yaml, "w", encoding="utf-8") as f:
            yaml_mod.safe_dump(derived, f)
    return str(derived_yaml.resolve())


def calib_yaml() -> Optional[str]:
    """Dataset yaml used for calibration. Prefers OMNI_CALIB_YAML so users
    can calibrate on a larger/more representative set (e.g. COCO) while
    evaluating on a target dataset (e.g. QR/Barcode). Falls back to
    OMNI_COCO_YAML for backwards compatibility."""
    return os.environ.get("OMNI_CALIB_YAML") or os.environ.get("OMNI_COCO_YAML")
