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


def _write_min_brevitas_recipe(tmp_path: Path, algo: str) -> Path:
    p = tmp_path / f"brevitas_{algo}.yaml"
    p.write_text(
        "name: smoke\n"
        "model: {family: yolo26, variant: n, weights: yolo26n.pt}\n"
        "runtime: {engine: tensorrt, dtype: int8}\n"
        "technique: {name: int8_ptq, source: brevitas,\n"
        f"            calibrator: {algo}, calibration_samples: 8,\n"
        "            calibration_seed: 42}\n"
        "measurement:\n"
        "  dataset: coco_val2017\n  num_images: 8\n"
        "  warmup_iters: 1\n  measure_iters: 2\n"
        "  batch_sizes: [1]\n  input_size: 640\n"
        "  gpu_clock_lock: false\n  seed: 42\n"
        "constraints: {max_map_drop_pct: 5.0}\n",
        encoding="utf-8",
    )
    return p


def test_brevitas_rejects_unknown_algo(tmp_path):
    """Guardrail: unknown algo must fail fast with a clear ValueError
    (matches the ort_quant precedent)."""
    from scripts.run_trt import _prepare_brevitas_onnx

    recipe = load_recipe(str(_write_min_brevitas_recipe(tmp_path, "bogus_algo")))
    with pytest.raises(ValueError, match="brevitas"):
        _prepare_brevitas_onnx(recipe, 640, tmp_path, dynamic=False)


def test_prepare_onnx_routes_to_brevitas(tmp_path):
    from scripts.run_trt import _prepare_onnx

    recipe = load_recipe(str(_write_min_brevitas_recipe(tmp_path, "percentile")))
    with patch("scripts.run_trt._prepare_brevitas_onnx") as m:
        m.return_value = tmp_path / "fake.onnx"
        path, is_qdq = _prepare_onnx(recipe, 640, tmp_path, bs=1)
    assert is_qdq is True
    assert path == tmp_path / "fake.onnx"
    m.assert_called_once()
