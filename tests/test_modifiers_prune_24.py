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


def test_apply_prunes_conv_weight():
    from scripts._modifiers.prune_24 import _apply_2_4_mask_to_module
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 16, 3)
    _apply_2_4_mask_to_module(conv)
    assert hasattr(conv, "weight_orig")
    assert hasattr(conv, "weight_mask")
    assert _has_2_4_pattern(conv.weight)


def test_finalize_removes_pruning_parametrization():
    from scripts._modifiers.prune_24 import _apply_2_4_mask_to_module, _finalize_module
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 16, 3)
    _apply_2_4_mask_to_module(conv)
    _finalize_module(conv)
    assert not hasattr(conv, "weight_orig")
    assert _has_2_4_pattern(conv.weight)
