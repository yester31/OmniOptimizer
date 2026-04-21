"""TRT-independent helpers for weight resolution + ONNX export.

Extracted from run_trt.py in Wave 6 so CPU runners (run_cpu.py) can reuse
the same weight-resolution + ONNX-export logic without transitively pulling
TensorRT / pycuda / CUDA imports. Keeping these helpers in a neutral module
also protects against future regressions where someone adds an
``import tensorrt`` at the top of run_trt.py.

Module invariants:

- No top-level import of ``tensorrt``, ``pycuda``, ``onnxruntime``, or
  ``openvino``. The only runtime dependencies at import time are ``pathlib``
  and the ``scripts._schemas`` types.
- ``modelopt.torch.opt`` is imported lazily inside ``_resolve_weights``
  and only on the modelopt_sparsify / modelopt_qat code path.
- ``ultralytics.YOLO`` is imported lazily inside the helpers that need it
  (kept out of top level to shave startup time and to let unit tests that
  only touch string paths run without ultralytics installed).
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._schemas import Recipe  # noqa: E402
from scripts import _split  # noqa: E402


def _load_yolo_for_restore(base_path: str) -> Any:
    """Load a plain YOLO instance to serve as the architecture skeleton
    for mto.restore()."""
    from ultralytics import YOLO
    return YOLO(base_path)


def _resolve_weights(recipe: Recipe) -> Any:
    """Return runner input: either a path string, or a YOLO-like object
    whose ``.model`` has modelopt modules restored.

    - No training: str path (recipe.model.weights).
    - prune_24 trained: str path to trained_weights/{name}.pt (plain
      ultralytics checkpoint after prune.remove()).
    - modelopt_sparsify / modelopt_qat: YOLO instance with mto.restore()
      applied; downstream _export_onnx accepts YOLO objects directly.
    """
    if recipe.technique.training is None:
        return recipe.model.weights
    trained = ROOT / "trained_weights" / f"{recipe.name}.pt"
    if not trained.exists():
        raise RuntimeError(
            f"Recipe {recipe.name!r} requires training but {trained} is "
            f"missing. Run: python scripts/train.py --recipe "
            f"recipes/{recipe.name}.yaml"
        )
    modifier = recipe.technique.training.modifier
    if modifier == "prune_24":
        return str(trained)
    if modifier in ("modelopt_sparsify", "modelopt_qat"):
        import modelopt.torch.opt as mto
        yolo = _load_yolo_for_restore(recipe.model.weights)
        mto.restore(yolo.model, str(trained))
        return yolo
    raise RuntimeError(f"unexpected modifier: {modifier!r}")


def _export_onnx(weights: Any, imgsz: int, half: bool, cache_dir: Path,
                 dynamic: bool = True, tag_suffix: str = "") -> Path:
    """Export YOLO weights to ONNX, defaulting to dynamic batch so a single
    ONNX can drive both bs=1 and bs>1 engines.

    ``weights`` accepts either a filesystem path (str/Path) to an ultralytics
    checkpoint, or a live ``ultralytics.YOLO`` instance (used by the modelopt
    2:4 sparsify path, which returns an in-memory YOLO whose backbone
    weights already carry the pruning pattern — re-saving and re-loading
    would drop ultralytics metadata).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    bs_tag = "dyn" if dynamic else "bs1"
    is_path = isinstance(weights, (str, Path))
    if is_path:
        stem = Path(weights).stem
    else:
        stem = Path(getattr(weights, "ckpt_path", "yolo") or "yolo").stem
    tag = f"{stem}_{imgsz}_{'fp16' if half else 'fp32'}{tag_suffix}_{bs_tag}.onnx"
    cached = cache_dir / tag
    if cached.exists():
        return cached
    if is_path:
        from ultralytics import YOLO
        model = YOLO(weights)
    else:
        model = weights  # already a YOLO-like wrapper with .export(...)
    onnx_path = model.export(
        format="onnx", imgsz=imgsz, half=False, simplify=True, dynamic=dynamic,
    )
    src = Path(onnx_path)
    if src != cached:
        src.rename(cached)
    return cached


# ---------------------------------------------------------------------------
# Calibration helpers (moved from run_trt.py in Wave 6 Task 4 so run_cpu.py
# can consume them without pulling the TRT runner module)
# ---------------------------------------------------------------------------

def _letterbox(img, imgsz: int):
    """Classic YOLO letterbox: resize keeping aspect ratio, pad to imgsz×imgsz
    with value 114. Returns CHW float32 in [0, 1], RGB channel order.

    Exact equivalent of the original run_trt._letterbox so switching callers
    over never changes INT8 calibration bytes (QDQ cache stability)."""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    r = imgsz / max(h, w)
    new_h, new_w = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - new_h) // 2
    left = (imgsz - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    rgb_chw = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb_chw)


def _iter_calib_samples(val_yaml: Optional[str], n_samples: int, imgsz: int, seed: int):
    """Yield (1, 3, H, W) float32 samples one at a time — streaming variant
    of :func:`_build_calib_numpy` that avoids the 2× peak memory spike from
    np.stack on large N. Windows laptops with constrained paging file can
    OOM the bulk builder at N>64 (discovered on Tiger Lake during Task 4).

    Usage: wrap in a CalibrationDataReader that pulls one sample per call.
    """
    import numpy as np

    if val_yaml and Path(val_yaml).exists():
        import cv2

        paths = _split.resolve_val_image_paths(val_yaml)
        rng = random.Random(seed)
        rng.shuffle(paths)
        paths = paths[:n_samples]
        n_yielded = 0
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            sample = _letterbox(img, imgsz)[None, ...]  # add batch dim → (1,3,H,W)
            yield sample
            n_yielded += 1
        if n_yielded > 0:
            return

    if os.environ.get("OMNI_ALLOW_RANDOM_CALIB") != "1":
        raise RuntimeError(
            "Calibration data unavailable: neither OMNI_CALIB_YAML nor "
            "OMNI_COCO_YAML points at a dataset producing readable images. "
            "Random-normal calibration silently drops INT8 mAP by double "
            "digits. Set OMNI_CALIB_YAML (or OMNI_COCO_YAML) to a valid "
            "ultralytics dataset yaml (with a val split pointing at real "
            "images), or pass OMNI_ALLOW_RANDOM_CALIB=1 to explicitly opt in "
            "to the random-normal fallback."
        )
    print("[warn] OMNI_ALLOW_RANDOM_CALIB=1: using random-normal calibration "
          "(INT8 mAP will be degraded)", file=sys.stderr)
    rng_np = np.random.default_rng(seed)
    for _ in range(n_samples):
        yield rng_np.standard_normal((1, 3, imgsz, imgsz), dtype=np.float32)


def _build_calib_numpy(val_yaml: Optional[str], n_samples: int, imgsz: int, seed: int):
    """Return (N,3,H,W) float32 numpy array for INT8 calibration.

    Default behavior: REAL images from COCO val (or whatever ``val_yaml``
    points at). If the yaml is missing or produces no usable images,
    raises ``RuntimeError`` unless the ``OMNI_ALLOW_RANDOM_CALIB`` env
    var is set to ``1`` — the random-normal fallback silently tanks mAP
    (double-digit %p on detection), so making it opt-in prevents
    debugging wild-goose chases like the Phase 3 sparsify diagnosis.
    """
    import numpy as np

    if val_yaml and Path(val_yaml).exists():
        import cv2

        paths = _split.resolve_val_image_paths(val_yaml)
        rng = random.Random(seed)
        rng.shuffle(paths)
        paths = paths[:n_samples]
        buf = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            buf.append(_letterbox(img, imgsz))
        if buf:
            return np.stack(buf, axis=0).astype(np.float32)

    if os.environ.get("OMNI_ALLOW_RANDOM_CALIB") != "1":
        raise RuntimeError(
            "Calibration data unavailable: neither OMNI_CALIB_YAML nor "
            "OMNI_COCO_YAML points at a dataset producing readable images. "
            "Random-normal calibration silently drops INT8 mAP by double "
            "digits. Set OMNI_CALIB_YAML (or OMNI_COCO_YAML) to a valid "
            "ultralytics dataset yaml (with a val split pointing at real "
            "images), or pass OMNI_ALLOW_RANDOM_CALIB=1 to explicitly opt in "
            "to the random-normal fallback."
        )
    print("[warn] OMNI_ALLOW_RANDOM_CALIB=1: using random-normal calibration "
          "(INT8 mAP will be degraded)", file=sys.stderr)
    rng_np = np.random.default_rng(seed)
    return rng_np.standard_normal((n_samples, 3, imgsz, imgsz), dtype=np.float32)
