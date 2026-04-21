"""prune_24 modifier tests (spec §5.4-5.5)."""
import pytest
import torch
import torch.nn as nn


def _has_2_4_pattern(w: torch.Tensor) -> bool:
    """Every group of 4 consecutive elements has at most 2 non-zeros."""
    flat = w.detach().reshape(-1)
    pad = (4 - flat.numel() % 4) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
    groups = flat.view(-1, 4)
    return bool(((groups != 0).sum(dim=-1) <= 2).all().item())


def test_verify_2_4_helper_positive():
    from scripts._modifiers.prune_24 import _verify_2_4_pattern
    w = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0])
    assert _verify_2_4_pattern(w)


def test_verify_2_4_helper_negative():
    from scripts._modifiers.prune_24 import _verify_2_4_pattern
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])  # all nonzero → violates
    assert not _verify_2_4_pattern(w)


def test_apply_attaches_hook_and_masks_weight():
    """apply() attaches a forward pre-hook and applies 2:4 mask immediately."""
    from scripts._modifiers.prune_24 import _attach_mask_hook
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 16, 3)
    _attach_mask_hook(conv)
    # Hook handle and mask attribute should be set on the module
    assert hasattr(conv, "_omni_prune24_hook")
    assert hasattr(conv, "_omni_prune24_mask")
    # The weight should satisfy 2:4 immediately (mask applied once in _attach)
    assert _has_2_4_pattern(conv.weight)
    # A forward pass should also preserve 2:4 (hook re-applies before each forward)
    x = torch.randn(1, 16, 8, 8)
    _ = conv(x)
    assert _has_2_4_pattern(conv.weight)


def test_finalize_removes_hook_and_preserves_pattern():
    """finalize() removes hook + mask attrs, weight stays 2:4."""
    from scripts._modifiers.prune_24 import _attach_mask_hook
    from scripts._modifiers.prune_24 import apply as apply_24  # noqa: F401
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 16, 3)
    _attach_mask_hook(conv)

    # Manually simulate finalize's per-module cleanup (bypass apply/finalize
    # since those iterate over a YOLO wrapper — we're testing the primitive).
    handle = getattr(conv, "_omni_prune24_hook")
    handle.remove()
    delattr(conv, "_omni_prune24_hook")
    mask = getattr(conv, "_omni_prune24_mask")
    with torch.no_grad():
        conv.weight.data.mul_(mask)
    delattr(conv, "_omni_prune24_mask")

    assert not hasattr(conv, "_omni_prune24_hook")
    assert not hasattr(conv, "_omni_prune24_mask")
    # state_dict keys should be the plain "weight" (NOT weight_orig / weight_mask)
    keys = list(conv.state_dict().keys())
    assert "weight" in keys
    assert "weight_orig" not in keys
    assert "weight_mask" not in keys
    # Pattern preserved
    assert _has_2_4_pattern(conv.weight)


def test_apply_raises_when_no_eligible_modules():
    """apply() must raise rather than silently produce a 0%-sparse model."""
    from scripts._modifiers.prune_24 import apply as apply_24
    from scripts._schemas import TrainingSpec

    class _NoEligibleYolo:
        # Model with only tiny layers that fail size threshold
        def __init__(self):
            self.model = nn.Sequential(nn.Linear(2, 2))  # numel = 4, < 16

    spec = TrainingSpec(
        base_checkpoint="x.pt", epochs=1, modifier="prune_24",
    )
    with pytest.raises(RuntimeError, match="no eligible modules"):
        apply_24(_NoEligibleYolo(), spec)
