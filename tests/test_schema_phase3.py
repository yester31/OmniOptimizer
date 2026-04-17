"""Phase 3 schema tests: sparsity_preprocess + nodes_to_exclude fields."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from pydantic import ValidationError

from scripts._schemas import TechniqueSpec


def test_technique_spec_accepts_sparsity_preprocess():
    t = TechniqueSpec(name="x", source="modelopt", sparsity_preprocess="2:4")
    assert t.sparsity_preprocess == "2:4"


def test_technique_spec_defaults_sparsity_preprocess_to_none():
    t = TechniqueSpec(name="x", source="modelopt")
    assert t.sparsity_preprocess is None


def test_technique_spec_accepts_nodes_to_exclude():
    t = TechniqueSpec(
        name="x",
        source="modelopt",
        nodes_to_exclude=["/model.0/conv/Conv", "/model.23/cv2.0/Conv"],
    )
    assert t.nodes_to_exclude == [
        "/model.0/conv/Conv",
        "/model.23/cv2.0/Conv",
    ]


def test_technique_spec_defaults_nodes_to_exclude_to_none():
    t = TechniqueSpec(name="x", source="modelopt")
    assert t.nodes_to_exclude is None


def test_technique_spec_rejects_invalid_sparsity_preprocess():
    with pytest.raises(ValidationError):
        TechniqueSpec(name="x", source="modelopt", sparsity_preprocess="1:2")
