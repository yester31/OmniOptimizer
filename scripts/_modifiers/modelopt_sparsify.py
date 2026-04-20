"""2:4 sparsity via nvidia-modelopt (torch-level).

Wraps ``modelopt.torch.sparsity.sparsify`` with mode ``sparse_magnitude``.
Default search config already uses ``pattern='2:4 sparsity'``, so no extra
config dict is needed.  Finalize uses ``modelopt.torch.opt.save`` to preserve
modelopt metadata alongside the state dict; runner restores via
``modelopt.torch.opt.restore``.
See spec §6.

NOTE: The ``config`` param of ``ms.sparsify`` is for *search* options
(verbose, forward_loop, etc.), not sparsity type.  The 2:4 pattern is the
default for ``sparse_magnitude`` mode (``pattern='2:4 sparsity'``).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO
    from scripts._schemas import TrainingSpec


def apply(yolo: "YOLO", spec: "TrainingSpec") -> None:
    import modelopt.torch.sparsity as ms

    ms.sparsify(
        yolo.model,
        mode="sparse_magnitude",
    )
    print("[modelopt_sparsify] sparsify(sparse_magnitude, 2:4) applied")


def finalize(yolo: "YOLO", spec: "TrainingSpec", out_pt: Path) -> None:
    import modelopt.torch.opt as mto

    mto.save(yolo.model, str(out_pt))
    print(f"[modelopt_sparsify] saved → {out_pt}")
