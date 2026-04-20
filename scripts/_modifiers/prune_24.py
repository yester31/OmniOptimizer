"""2:4 structured magnitude pruning + SAT (Sparsity-Aware Training).

Flow (spec §5.4 / §5.5):
1. ``apply(yolo, spec)``: for each eligible Conv/Linear in the model, compute
   a mask that keeps top-2 magnitude weights out of every consecutive group
   of 4, then register ``torch.nn.utils.prune.custom_from_mask``. The forward
   pre-hook re-applies the mask each call so ``optimizer.step`` updates
   ``weight_orig`` freely but the effective forward weight always satisfies
   the 2:4 pattern.
2. Training runs normally (60 epochs per recipe).
3. ``finalize(yolo, spec, out_pt)``: call ``prune.remove()`` to bake masks
   into the weight tensor permanently, then ``torch.save`` a plain state
   dict that ``ultralytics.YOLO`` can load via its normal path.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

if TYPE_CHECKING:
    from ultralytics import YOLO
    from scripts._schemas import TrainingSpec


def _compute_2_4_mask(weight: torch.Tensor) -> torch.Tensor:
    """For a weight tensor, return a mask with the 2:4 structured pattern.

    The mask keeps the two largest-magnitude elements in every block of 4
    along the flattened dim and zeros the other two. If the total element
    count is not a multiple of 4, the trailing remainder keeps its
    magnitude-top half (equivalent to 2:4 applied to the padded view).
    """
    flat = weight.detach().reshape(-1)
    n = flat.numel()
    pad = (4 - n % 4) % 4
    if pad:
        flat_padded = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
    else:
        flat_padded = flat
    groups = flat_padded.abs().view(-1, 4)
    # Top-2 magnitude per group
    _, topk_idx = groups.topk(2, dim=-1)
    mask_groups = torch.zeros_like(groups)
    mask_groups.scatter_(1, topk_idx, 1.0)
    mask_flat = mask_groups.view(-1)
    if pad:
        mask_flat = mask_flat[:n]
    return mask_flat.view_as(weight)


def _verify_2_4_pattern(weight: torch.Tensor) -> bool:
    """Check that every 4-element group has <= 2 non-zeros."""
    flat = weight.detach().reshape(-1)
    n = flat.numel()
    pad = (4 - n % 4) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
    groups = flat.view(-1, 4)
    return bool(((groups != 0).sum(dim=-1) <= 2).all().item())


def _is_eligible_module(m: nn.Module) -> bool:
    """Conv2d and Linear layers are eligible for 2:4 pruning.

    Excludes 1x1 pointwise convs with < 4 input channels (too small),
    depthwise convs (groups == in_channels), and the final detect head
    (heuristic: skip if ``_omni_skip_prune`` attribute is set).
    """
    if getattr(m, "_omni_skip_prune", False):
        return False
    if isinstance(m, nn.Conv2d):
        if m.groups == m.in_channels and m.groups > 1:  # depthwise
            return False
        if m.weight.numel() < 16:  # too small to benefit
            return False
        return True
    if isinstance(m, nn.Linear):
        return m.weight.numel() >= 16
    return False


def _apply_2_4_mask_to_module(module: nn.Module) -> None:
    """Attach a 2:4 mask via ``custom_from_mask``.

    Registers ``weight_orig`` (learnable) and ``weight_mask`` (buffer), plus
    a forward pre-hook that computes ``weight = weight_orig * weight_mask``.
    """
    mask = _compute_2_4_mask(module.weight)
    prune.custom_from_mask(module, name="weight", mask=mask)


def _finalize_module(module: nn.Module) -> None:
    """Bake the mask permanently into ``weight`` and remove hook/parametrization."""
    prune.remove(module, "weight")


def apply(yolo: "YOLO", spec: "TrainingSpec") -> None:
    """Mutate ``yolo.model`` in-place to add 2:4 mask parametrizations."""
    applied = 0
    for _, m in yolo.model.named_modules():
        if _is_eligible_module(m):
            _apply_2_4_mask_to_module(m)
            applied += 1
    if applied == 0:
        raise RuntimeError(
            "prune_24.apply: no eligible modules found for 2:4 pruning. "
            "Check model architecture (expected Conv2d/Linear with numel >= 16, "
            "non-depthwise)."
        )
    print(f"[prune_24] applied 2:4 mask to {applied} modules")


def finalize(yolo: "YOLO", spec: "TrainingSpec", out_pt: Path) -> None:
    """Bake masks, verify, and save a plain ultralytics-compatible checkpoint."""
    removed = 0
    for name, m in yolo.model.named_modules():
        if prune.is_pruned(m):
            _finalize_module(m)
            removed += 1
            if not _verify_2_4_pattern(m.weight):
                raise RuntimeError(
                    f"prune_24.finalize: module {name!r} weight does not "
                    f"satisfy 2:4 pattern after prune.remove()"
                )
    print(f"[prune_24] finalized {removed} modules, 2:4 pattern verified")
    torch.save({"model": yolo.model}, str(out_pt))
    print(f"[prune_24] saved → {out_pt}")
