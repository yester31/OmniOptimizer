"""mAP evaluation on COCO val 2017.

For v1 we lean on ultralytics' own ``model.val`` which handles the whole
dataloader + pycocotools pipeline. That keeps the numbers directly comparable
to published YOLO benchmarks and means we only have to supply a callable
``predict_fn`` for non-ultralytics backends (ONNX Runtime, TensorRT).

``evaluate_via_ultralytics`` is the quick path used by PyTorch recipes.

``evaluate_generic`` is a placeholder for ONNX/TRT paths: it accumulates
predictions into COCO JSON format and lets pycocotools compute mAP. v1 stubs
it with a clear TODO so we don't silently report fake accuracy.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from ._schemas import AccuracyStats


def _coco_yaml() -> str:
    """Path to the dataset yaml ultralytics should load. Override via env var
    ``OMNI_COCO_YAML`` to point at a local val-only copy (the default
    ``coco.yaml`` triggers a ~25 GB train/test download)."""
    return os.environ.get("OMNI_COCO_YAML", "coco.yaml")


def evaluate_via_ultralytics(
    weights: str,
    num_images: Optional[int] = None,
    batch: int = 1,
    imgsz: int = 640,
    device: str | int = 0,
    half: bool = False,
) -> AccuracyStats:
    """Runs ultralytics' built-in COCO val.

    ``num_images`` is advisory: ultralytics evaluates the full val split.
    We expose it so callers can downsample for smoke tests by subclassing.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(
        data=_coco_yaml(),
        imgsz=imgsz,
        batch=batch,
        device=device,
        half=half,
        plots=False,
        verbose=False,
    )
    # ultralytics' DetMetrics exposes .box.map / .box.map50
    try:
        return AccuracyStats(
            map_50_95=float(metrics.box.map),
            map_50=float(metrics.box.map50),
        )
    except Exception:
        return AccuracyStats()


def evaluate_generic(
    predict_fn: Callable[[list[str]], list[dict]],
    ann_file: str,
    img_dir: str,
    num_images: Optional[int] = None,
) -> AccuracyStats:
    """Backend-agnostic COCO mAP eval.

    Expected shape of ``predict_fn(img_paths)``: list of dicts with
    ``image_id``, ``category_id``, ``bbox`` (xywh), ``score`` — i.e. COCO
    detection result format.

    v1 status: stub. Fill in when ORT/TRT paths start producing predictions.
    """
    raise NotImplementedError(
        "evaluate_generic: populate COCO results from predict_fn outputs and "
        "run pycocotools COCOeval. Pending in v1; ORT/TRT recipes currently "
        "reuse ultralytics val by exporting a throwaway wrapper model."
    )
