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

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._schemas import Recipe  # noqa: E402


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
