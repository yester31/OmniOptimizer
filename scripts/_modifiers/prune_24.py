"""2:4 structured magnitude pruning + SAT (Sparsity-Aware Training).

Flow (spec §5.4 / §5.5):
1. ``apply(yolo, spec)``: for each eligible Conv/Linear in the model, compute
   a mask that keeps top-2 magnitude weights out of every consecutive group
   of 4, then register a forward pre-hook that applies the mask to
   ``weight.data`` in-place each forward call.  Unlike torch.nn.utils.prune,
   this approach does NOT rename ``weight`` in the state_dict, so ultralytics
   EMA / get_model / load() remain fully compatible.
2. Training runs normally (60 epochs per recipe).
3. ``finalize(yolo, spec, out_pt)``: remove the hooks, bake the mask into
   ``weight.data`` permanently (already sparse at this point), verify the
   2:4 pattern, then ``torch.save`` a plain state dict.

PRE_TRAIN_HOOK = True signals to _train_core that apply() must run *after*
ultralytics constructs trainer.model (via on_train_start callback) so that
pruning is applied to the actual in-training model.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ultralytics import YOLO
    from scripts._schemas import TrainingSpec

# Signal to _train_core: apply() must be deferred to on_train_start callback
# so it acts on trainer.model after ultralytics rebuilds the model from yaml.
PRE_TRAIN_HOOK = True

# Attribute name used to store hook handle and buffer name on each module.
_HOOK_ATTR = "_omni_prune24_hook"
_MASK_ATTR = "_omni_prune24_mask"


def _compute_2_4_mask(weight: torch.Tensor) -> torch.Tensor:
    """Return a 2:4 structured sparsity mask (top-2 magnitude per block of 4).

    The mask keeps the two largest-magnitude elements in every block of 4
    along the flattened dim and zeros the other two. Trailing remainder
    (when numel % 4 != 0) is treated as a padded block.
    """
    flat = weight.detach().reshape(-1)
    n = flat.numel()
    pad = (4 - n % 4) % 4
    if pad:
        flat_padded = torch.cat([flat, torch.zeros(pad, device=flat.device,
                                                   dtype=flat.dtype)])
    else:
        flat_padded = flat
    groups = flat_padded.abs().view(-1, 4)
    _, topk_idx = groups.topk(2, dim=-1)
    mask_groups = torch.zeros_like(groups)
    mask_groups.scatter_(1, topk_idx, 1.0)
    mask_flat = mask_groups.view(-1)
    if pad:
        mask_flat = mask_flat[:n]
    return mask_flat.view_as(weight)


def _verify_2_4_pattern(weight: torch.Tensor) -> bool:
    """Return True iff every 4-element group has at most 2 non-zeros."""
    flat = weight.detach().reshape(-1)
    n = flat.numel()
    pad = (4 - n % 4) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device,
                                            dtype=flat.dtype)])
    groups = flat.view(-1, 4)
    return bool(((groups != 0).sum(dim=-1) <= 2).all().item())


def _is_eligible_module(m: nn.Module) -> bool:
    """Conv2d and Linear layers eligible for 2:4 pruning.

    Skips depthwise convs, layers with < 16 elements, and anything tagged
    with ``_omni_skip_prune``.
    """
    if getattr(m, "_omni_skip_prune", False):
        return False
    if isinstance(m, nn.Conv2d):
        if m.groups == m.in_channels and m.groups > 1:  # depthwise
            return False
        if m.weight.numel() < 16:
            return False
        return True
    if isinstance(m, nn.Linear):
        return m.weight.numel() >= 16
    return False


def _attach_mask_hook(module: nn.Module) -> None:
    """Register a 2:4 mask on *module* using a forward pre-hook.

    The mask is stored as a plain attribute (not a buffer, to avoid
    inflating the parameter count) and is applied to ``weight.data``
    in-place before each forward call.  This keeps the state_dict keys
    unchanged so ultralytics EMA and checkpoint loading remain compatible.
    """
    mask = _compute_2_4_mask(module.weight)
    # Apply once immediately so the very first forward sees sparse weights.
    with torch.no_grad():
        module.weight.data.mul_(mask)
    # Store mask as a plain tensor attribute (non-parameter, non-buffer).
    setattr(module, _MASK_ATTR, mask)

    def _pre_hook(mod, _input):
        m = getattr(mod, _MASK_ATTR, None)
        if m is not None:
            with torch.no_grad():
                mod.weight.data.mul_(m)

    handle = module.register_forward_pre_hook(_pre_hook)
    setattr(module, _HOOK_ATTR, handle)


def apply(yolo: "YOLO", spec: "TrainingSpec") -> None:
    """Mutate ``yolo.model`` in-place to add 2:4 mask forward hooks."""
    applied = 0
    for _, m in yolo.model.named_modules():
        if _is_eligible_module(m):
            _attach_mask_hook(m)
            applied += 1
    if applied == 0:
        raise RuntimeError(
            "prune_24.apply: no eligible modules found for 2:4 pruning. "
            "Check model architecture (expected Conv2d/Linear with numel >= 16, "
            "non-depthwise)."
        )
    print(f"[prune_24] applied 2:4 mask to {applied} modules")


def finalize(yolo: "YOLO", spec: "TrainingSpec", out_pt: Path) -> None:
    """Remove hooks, bake masks, verify 2:4 pattern, and save checkpoint."""
    finalized = 0
    for name, m in yolo.model.named_modules():
        handle = getattr(m, _HOOK_ATTR, None)
        if handle is None:
            continue
        # Remove the forward hook.
        handle.remove()
        delattr(m, _HOOK_ATTR)
        # Bake mask into weight.data one final time and remove the mask attr.
        mask = getattr(m, _MASK_ATTR, None)
        if mask is not None:
            with torch.no_grad():
                m.weight.data.mul_(mask)
            delattr(m, _MASK_ATTR)
        if not _verify_2_4_pattern(m.weight):
            raise RuntimeError(
                f"prune_24.finalize: module {name!r} weight does not satisfy "
                f"2:4 pattern after finalize."
            )
        finalized += 1
    print(f"[prune_24] finalized {finalized} modules, 2:4 pattern verified")
    torch.save({"model": yolo.model}, str(out_pt))
    print(f"[prune_24] saved → {out_pt}")
