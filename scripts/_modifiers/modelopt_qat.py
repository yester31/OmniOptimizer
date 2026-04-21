"""INT8 QAT (Quantization-Aware Training) via nvidia-modelopt.

Inserts fake-quant modules before training; STE (straight-through
estimator) lets gradients flow through quant/dequant ops so scales are
learned. See spec §5.1-5.2 (lr0=1e-4, amp=False mandated at train layer).

Known limitation (spec §13): no forward_loop calibration — scales init
at default and train from there. Future iteration can add 512-image
COCO val calibration loop before training.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO
    from scripts._schemas import TrainingSpec


_QUANT_CONFIG_PRESETS = {
    "int8_default": "INT8_DEFAULT_CFG",
}


def _resolve_config(quant_config: str | None):
    import modelopt.torch.quantization as mtq

    key = quant_config or "int8_default"
    attr_name = _QUANT_CONFIG_PRESETS.get(key)
    if attr_name is None:
        raise ValueError(
            f"Unknown quant_config {key!r}. Known: "
            f"{list(_QUANT_CONFIG_PRESETS)}"
        )
    return getattr(mtq, attr_name)


def apply(yolo: "YOLO", spec: "TrainingSpec") -> None:
    import modelopt.torch.quantization as mtq

    cfg = _resolve_config(spec.quant_config)
    mtq.quantize(yolo.model, cfg)

    # Silent no-op guard: confirm at least one module has modelopt
    # quantizer hooks. modelopt swaps Conv/Linear with Quant* variants
    # AND attaches input/weight quantizer submodules. Check for the
    # latter since class names vary across modelopt versions.
    wrapped = sum(
        1 for _, m in yolo.model.named_modules()
        if any(s in type(m).__name__.lower() for s in ("quant",))
    )
    if wrapped == 0:
        raise RuntimeError(
            "modelopt_qat.apply: no modules were quantized. Check "
            "quant_config preset and model architecture."
        )
    print(f"[modelopt_qat] quantize({spec.quant_config or 'int8_default'}) applied "
          f"— {wrapped} modules wrapped")


def finalize(yolo: "YOLO", spec: "TrainingSpec", out_pt: Path) -> None:
    import modelopt.torch.opt as mto

    mto.save(yolo.model, str(out_pt))
    print(f"[modelopt_qat] saved → {out_pt}")
