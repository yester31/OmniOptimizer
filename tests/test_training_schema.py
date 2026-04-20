"""Schema tests for TrainingSpec (spec §4)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from pydantic import ValidationError
from scripts._schemas import TrainingSpec, TechniqueSpec


def test_training_spec_minimal_fields():
    spec = TrainingSpec(
        base_checkpoint="best_qr.pt",
        epochs=30,
        modifier="modelopt_qat",
    )
    assert spec.base_checkpoint == "best_qr.pt"
    assert spec.epochs == 30
    assert spec.modifier == "modelopt_qat"
    assert spec.lr0 == 0.001  # default
    assert spec.batch == 8


def test_training_spec_requires_modifier():
    with pytest.raises(ValidationError):
        TrainingSpec(base_checkpoint="best_qr.pt", epochs=30)  # missing modifier


def test_training_spec_rejects_unknown_modifier():
    with pytest.raises(ValidationError):
        TrainingSpec(
            base_checkpoint="best_qr.pt",
            epochs=30,
            modifier="unknown_algo",
        )


def test_training_spec_accepts_all_three_modifiers():
    for m in ("prune_24", "modelopt_sparsify", "modelopt_qat"):
        spec = TrainingSpec(
            base_checkpoint="best_qr.pt",
            epochs=10,
            modifier=m,
        )
        assert spec.modifier == m


def test_technique_spec_training_is_optional():
    t = TechniqueSpec(name="int8_ptq", source="trt_builtin")
    assert t.training is None


def test_technique_spec_with_training():
    t = TechniqueSpec(
        name="int8_qat",
        source="modelopt",
        training=TrainingSpec(
            base_checkpoint="best_qr.pt",
            epochs=30,
            modifier="modelopt_qat",
        ),
    )
    assert t.training.modifier == "modelopt_qat"
