"""Wave 4 schema + dispatch tests for brevitas backend."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._schemas import TechniqueSpec, load_recipe  # noqa: E402


def test_source_literal_accepts_brevitas():
    spec = TechniqueSpec(name="int8_ptq", source="brevitas")
    assert spec.source == "brevitas"


@pytest.mark.parametrize("src", ["trt_builtin", "modelopt", "ort_quant", "brevitas"])
def test_source_literal_accepts_all_backends(src):
    spec = TechniqueSpec(name="int8_ptq", source=src)
    assert spec.source == src


def test_source_literal_rejects_unknown_backend():
    with pytest.raises(Exception):
        TechniqueSpec(name="int8_ptq", source="bogus_backend")


def test_source_tag_includes_brevitas():
    """Engine cache filenames must stay short enough on Windows (MAX_PATH=260)."""
    from scripts.run_trt import _SOURCE_TAG

    assert _SOURCE_TAG["brevitas"] == "_brev"
    assert _SOURCE_TAG["trt_builtin"] == ""
    assert _SOURCE_TAG["modelopt"] == "_modelopt"
    assert _SOURCE_TAG["ort_quant"] == "_ort"
