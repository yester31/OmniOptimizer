"""Tests for scripts/run_trt.py::_modelopt_onnx_tag — Wave 16 T7.

Fixes a cache-key bug where recipes with identical calibrator but different
``nodes_to_exclude`` shared the same ONNX filename. Whichever ran first
poisoned the cache for the other (pre-existing bug flagged during Wave 15
D2 measurement — #09 and #12 share ``best_qr_640_modelopt_entropy_bs1.onnx``).

These tests are pure: no filesystem, no modelopt import required.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _recipe_09_like():
    """Synthesize a recipe matching #09 (entropy, no excludes)."""
    from scripts._schemas import load_recipe
    return load_recipe(str(ROOT / "recipes" / "09_modelopt_int8_entropy.yaml"))


def _recipe_12_like():
    """Synthesize a recipe matching #12 (entropy + 4 excludes)."""
    from scripts._schemas import load_recipe
    return load_recipe(str(ROOT / "recipes" / "12_modelopt_int8_mixed.yaml"))


def test_different_nodes_to_exclude_yield_different_tags():
    """The T7 bug: #09 (no excludes) and #12 (4 excludes) must not share a tag."""
    from scripts.run_trt import _modelopt_onnx_tag

    tag_09 = _modelopt_onnx_tag(_recipe_09_like(), 640, dynamic=False)
    tag_12 = _modelopt_onnx_tag(_recipe_12_like(), 640, dynamic=False)
    assert tag_09 != tag_12, (
        f"#09 and #12 must not share cache keys. Got both = {tag_09!r}"
    )
    # #09 has no excludes → no _ex suffix. #12 does → _ex<sha8> in the tag.
    assert "_ex" not in tag_09, f"#09 tag must not carry _ex token: {tag_09!r}"
    assert "_ex" in tag_12, f"#12 tag must carry _ex token: {tag_12!r}"


def test_empty_nodes_to_exclude_preserves_legacy_tag():
    """No exclusions → tag unchanged from pre-T7 format.

    Protects any recipe with empty nodes_to_exclude from cache invalidation.
    The weights stem comes from recipe.model.weights (``yolo26n.pt`` for #09).
    """
    from scripts.run_trt import _modelopt_onnx_tag

    r = _recipe_09_like()
    tag = _modelopt_onnx_tag(r, 640, dynamic=False)
    stem = Path(r.model.weights).stem
    assert tag == f"{stem}_640_modelopt_entropy_bs1.onnx"
    assert "_ex" not in tag


def test_order_insensitive_hash():
    """nodes_to_exclude is sorted before hashing — same set, different order, same tag."""
    from scripts._schemas import ModelSpec, Recipe, TechniqueSpec, MeasurementSpec
    from scripts.run_trt import _modelopt_onnx_tag

    base = dict(
        model=ModelSpec(family="yolo26", variant="n", weights="yolo26n.pt"),
        runtime={"engine": "tensorrt", "dtype": "int8"},
        measurement=MeasurementSpec(
            dataset="coco_val2017", num_images=10, batch_sizes=[1],
            warmup_iters=10, measure_iters=10, input_size=640,
        ),
    )
    r1 = Recipe(
        name="r1",
        technique=TechniqueSpec(
            name="int8_ptq", source="modelopt", calibrator="entropy",
            nodes_to_exclude=["/a/Conv", "/b/Conv", "/c/Conv"],
        ),
        **base,
    )
    r2 = Recipe(
        name="r2",
        technique=TechniqueSpec(
            name="int8_ptq", source="modelopt", calibrator="entropy",
            nodes_to_exclude=["/c/Conv", "/a/Conv", "/b/Conv"],  # shuffled
        ),
        **base,
    )
    assert _modelopt_onnx_tag(r1, 640, dynamic=False) == _modelopt_onnx_tag(r2, 640, dynamic=False)


def test_asymmetric_still_tagged():
    """Regression: Wave 14 A5 _asym suffix must still appear alongside the new _ex token."""
    from scripts.run_trt import _modelopt_onnx_tag

    r42 = _recipe_42_like()
    tag = _modelopt_onnx_tag(r42, 640, dynamic=False)
    assert "_asym" in tag, f"#42 tag lost _asym suffix: {tag!r}"


def _recipe_42_like():
    from scripts._schemas import load_recipe
    return load_recipe(str(ROOT / "recipes" / "42_modelopt_int8_asymmetric.yaml"))
