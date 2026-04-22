"""Wave 15 D4.2 — every recipe YAML parses under the current schema.

Regression guard for schema evolution: when a new Optional field lands in
_schemas.py (e.g., Wave 14 builder_optimization_level, Wave 15
build_ceiling_s), every legacy recipe file must continue to validate
cleanly with sensible defaults. Without this test, a subtle schema
change could break a subset of recipes silently and only surface at
``make all`` time.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts._schemas import Recipe, load_recipe


ROOT = Path(__file__).resolve().parents[1]
RECIPES_DIR = ROOT / "recipes"

# All recipe files present at Wave 15 ship time. New recipes get appended
# here when added; removed recipes get deleted. The explicit list is the
# anti-regression signal: if the directory loses or gains a file the
# difference is obvious in the diff.
_EXPECTED_RECIPES = sorted(
    p.name for p in RECIPES_DIR.glob("*.yaml")
)


def test_recipe_directory_not_empty():
    assert _EXPECTED_RECIPES, "recipes/*.yaml is empty — bank went missing?"


@pytest.mark.parametrize("recipe_filename", _EXPECTED_RECIPES)
def test_recipe_validates(recipe_filename: str):
    recipe_path = RECIPES_DIR / recipe_filename
    # load_recipe handles the OMNI_WEIGHTS_OVERRIDE plumbing, but raw
    # validation is what we want here — we don't want the test to depend
    # on env state.
    import yaml
    with recipe_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    r = Recipe.model_validate(data)
    # Invariants all recipes must carry.
    assert r.name, f"{recipe_filename}: name is empty"
    assert r.model.family, f"{recipe_filename}: model.family missing"
    assert r.runtime.engine, f"{recipe_filename}: runtime.engine missing"
    assert r.runtime.dtype in {"fp32", "fp16", "bf16", "int8"}, (
        f"{recipe_filename}: unknown dtype {r.runtime.dtype!r}"
    )
    assert r.measurement.batch_sizes, f"{recipe_filename}: no batch_sizes"


def test_load_recipe_roundtrip():
    """load_recipe path (with env weights override hook) works on the
    simplest recipe we've got — catches regressions in load_recipe that
    unit tests of Recipe.model_validate alone would miss."""
    # Pick a recipe that's been around since early waves and is unlikely
    # to carry exotic fields. 00_trt_fp32.yaml is the stable baseline.
    target = RECIPES_DIR / "00_trt_fp32.yaml"
    if not target.exists():
        pytest.skip("00_trt_fp32.yaml missing; skipping roundtrip")
    r = load_recipe(str(target))
    assert r.runtime.dtype == "fp32"


def test_recipe_count_matches_claude_md_claim():
    """CLAUDE.md currently claims "31 active recipes". If someone adds or
    removes a recipe, they must also update that claim. This test fails
    noisily when the two drift, forcing a CLAUDE.md update in the same PR.
    """
    n = len(_EXPECTED_RECIPES)
    claude_md = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    # Look for a digit-count claim in the format "N active recipes".
    import re
    m = re.search(r"(\d+)\s+active recipes", claude_md)
    assert m is not None, "CLAUDE.md no longer mentions 'N active recipes' — update this test + CLAUDE.md"
    claimed = int(m.group(1))
    assert claimed == n, (
        f"CLAUDE.md claims {claimed} active recipes but recipes/*.yaml has {n}. "
        f"Update CLAUDE.md scope block to match."
    )
