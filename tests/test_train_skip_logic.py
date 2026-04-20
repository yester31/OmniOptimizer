"""train.py skip / --force tests."""
import pytest
from pathlib import Path


def _write_recipe(tmp_path, name: str = "skip_r") -> Path:
    yaml_text = f"""
name: {name}
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
runtime:
  engine: tensorrt
  dtype: int8
technique:
  name: int8_test
  training:
    base_checkpoint: fake_base.pt
    epochs: 1
    modifier: prune_24
measurement:
  dataset: coco
  num_images: 10
  warmup_iters: 1
  measure_iters: 1
  batch_sizes: [1]
"""
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml_text)
    return p


def test_skip_when_output_exists(tmp_path, monkeypatch, capsys):
    from scripts import train, _train_core
    monkeypatch.setattr(_train_core, "ROOT", tmp_path)
    out_dir = tmp_path / "trained_weights"
    out_dir.mkdir()
    (out_dir / "skip_r.pt").write_bytes(b"existing")

    called = {}
    monkeypatch.setattr(_train_core, "train_with_modifier",
                        lambda r: called.setdefault("ran", True))
    recipe_path = _write_recipe(tmp_path)
    train.main([f"--recipe={recipe_path}"])
    captured = capsys.readouterr()
    assert "skip" in captured.out.lower()
    assert "ran" not in called


def test_force_overrides_skip(tmp_path, monkeypatch):
    from scripts import train, _train_core
    monkeypatch.setattr(_train_core, "ROOT", tmp_path)
    out_dir = tmp_path / "trained_weights"
    out_dir.mkdir()
    (out_dir / "skip_r.pt").write_bytes(b"existing")

    called = {}
    def fake_train(recipe):
        called["ran"] = True
        return out_dir / f"{recipe.name}.pt"
    monkeypatch.setattr(_train_core, "train_with_modifier", fake_train)

    recipe_path = _write_recipe(tmp_path)
    train.main([f"--recipe={recipe_path}", "--force"])
    assert called.get("ran") is True
