"""CLI entry for QAT / sparsity fine-tuning.

Usage:
    python scripts/train.py --recipe recipes/17_modelopt_int8_qat.yaml
    python scripts/train.py --recipe recipes/07_trt_int8_sparsity.yaml --force

Produces ``trained_weights/{recipe.name}.pt`` + ``.train.json``.
Skips if output already exists (use ``--force`` to retrain).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._schemas import load_recipe  # noqa: E402
from scripts import _train_core  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--force", action="store_true",
                        help="retrain even if trained_weights/{name}.pt exists")
    args = parser.parse_args(argv)

    recipe = load_recipe(args.recipe)
    if recipe.technique.training is None:
        print(f"error: recipe {recipe.name!r} has no technique.training "
              f"section — nothing to train.", file=sys.stderr)
        sys.exit(1)

    out_dir = _train_core.ROOT / "trained_weights"
    out_dir.mkdir(exist_ok=True)
    out_pt = out_dir / f"{recipe.name}.pt"
    if out_pt.exists() and not args.force:
        print(f"[skip] {out_pt} already exists (use --force to retrain)")
        return 0

    trained = _train_core.train_with_modifier(recipe)
    print(f"trained weights -> {trained}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
