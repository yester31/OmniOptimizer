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

    ms.sparsify(yolo.model, mode="sparse_magnitude")

    # Verify at least one layer actually got a non-trivial mask. modelopt
    # silently skips layers that don't meet shape constraints (e.g.,
    # in_channels % 16 != 0); if every layer is skipped, _weight_mask ends
    # up all-ones and TRT SPARSE_WEIGHTS becomes a no-op.
    sparsified = 0
    for _, m in yolo.model.named_modules():
        mask = getattr(m, "_weight_mask", None)
        if mask is not None and not bool(mask.all().item()):
            sparsified += 1
    if sparsified == 0:
        raise RuntimeError(
            "modelopt_sparsify.apply: no layers were actually sparsified. "
            "All eligible layers may have been skipped by modelopt due to "
            "shape constraints (e.g., in_channels % 16 != 0). Check "
            "model architecture."
        )
    print(f"[modelopt_sparsify] sparsify(sparse_magnitude, 2:4) applied — "
          f"{sparsified} layers sparsified")


def finalize(yolo: "YOLO", spec: "TrainingSpec", out_pt: Path) -> None:
    import modelopt.torch.opt as mto

    mto.save(yolo.model, str(out_pt))
    print(f"[modelopt_sparsify] saved → {out_pt}")
