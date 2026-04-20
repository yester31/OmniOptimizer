"""train.py modifier dispatch tests (spec §12 data flow)."""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock


def _write_recipe(tmp_path, modifier: str, name: str = "test_r") -> Path:
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
    modifier: {modifier}
    data_yaml: fake_data.yaml
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


def test_dispatch_prune_24(tmp_path, monkeypatch):
    """train.py should route prune_24 recipes via PRE_TRAIN_HOOK callback.

    prune_24 sets PRE_TRAIN_HOOK = True so _train_core defers apply() into
    an on_train_start callback (registered via yolo.add_callback).  The fake
    _run_ultralytics_train must fire that callback manually to simulate what
    ultralytics does in real training.
    """
    from scripts import train, _train_core
    recipe_path = _write_recipe(tmp_path, "prune_24")
    called = {}

    def fake_train(yolo, spec, run_name):
        called["trained"] = True
        # Simulate ultralytics firing the on_train_start callback.
        # _train_core registers via yolo.add_callback; ultralytics stores
        # callbacks in yolo.callbacks dict.
        for cb in yolo.callbacks.get("on_train_start", []):
            trainer_stub = MagicMock()
            trainer_stub.model = nn.Linear(4, 4)
            cb(trainer_stub)
        # Also set yolo.trainer so the post-train model restore succeeds.
        yolo.trainer = MagicMock()
        yolo.trainer.model = nn.Linear(4, 4)
        return tmp_path / "dummy_runs" / run_name / "weights" / "last.pt"

    def fake_load_yolo(path):
        m = MagicMock()
        m.model = nn.Linear(4, 4)
        # Provide a real callbacks dict so add_callback works correctly.
        m.callbacks = {}

        def _add_callback(event, fn):
            m.callbacks.setdefault(event, []).append(fn)

        m.add_callback = _add_callback
        return m

    monkeypatch.setattr(_train_core, "_run_ultralytics_train", fake_train)
    monkeypatch.setattr(_train_core, "_load_yolo", fake_load_yolo)
    monkeypatch.setattr(_train_core, "ROOT", tmp_path)
    out = tmp_path / "trained_weights"
    out.mkdir()
    # Ensure base_checkpoint existence check passes
    (tmp_path / "fake_base.pt").write_bytes(b"x")

    import scripts._modifiers.prune_24 as mod
    applied = []
    monkeypatch.setattr(mod, "apply", lambda y, s: applied.append("applied"))
    monkeypatch.setattr(mod, "finalize", lambda y, s, p: p.write_bytes(b"fake"))

    train.main([f"--recipe={recipe_path}", "--force"])
    # apply() is called inside the on_train_start callback, not directly.
    assert applied == ["applied"], (
        "apply() should have been called once via the on_train_start callback"
    )
    assert (out / "test_r.pt").exists()
    assert called.get("trained") is True


def test_dispatch_modelopt_qat(tmp_path, monkeypatch):
    from scripts import train, _train_core
    recipe_path = _write_recipe(tmp_path, "modelopt_qat", name="test_qat")

    def fake_train(yolo, spec, run_name):
        return tmp_path / "dummy" / "last.pt"

    def fake_load_yolo(path):
        m = MagicMock()
        m.model = nn.Linear(4, 4)
        return m

    monkeypatch.setattr(_train_core, "_run_ultralytics_train", fake_train)
    monkeypatch.setattr(_train_core, "_load_yolo", fake_load_yolo)
    monkeypatch.setattr(_train_core, "ROOT", tmp_path)
    (tmp_path / "trained_weights").mkdir()
    (tmp_path / "fake_base.pt").write_bytes(b"x")

    import scripts._modifiers.modelopt_qat as mod
    monkeypatch.setattr(mod, "apply", lambda y, s: None)
    monkeypatch.setattr(mod, "finalize", lambda y, s, p: p.write_bytes(b"fake"))

    train.main([f"--recipe={recipe_path}", "--force"])
    assert (tmp_path / "trained_weights" / "test_qat.pt").exists()


def test_rejects_recipe_without_training(tmp_path):
    from scripts import train
    recipe_path = tmp_path / "no_train.yaml"
    recipe_path.write_text("""
name: no_train
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
runtime:
  engine: tensorrt
  dtype: int8
technique:
  name: int8_test
measurement:
  dataset: coco
  num_images: 10
  warmup_iters: 1
  measure_iters: 1
  batch_sizes: [1]
""")
    with pytest.raises(SystemExit):
        train.main([f"--recipe={recipe_path}"])
