"""Read all result JSONs, rank, enforce constraints, emit report.md."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._schemas import Result  # noqa: E402


def _safe(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return value
    except Exception:
        return default


def load_results(results_dir: Path) -> list[Result]:
    items: list[Result] = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            items.append(Result.model_validate(data))
        except Exception as e:
            print(f"[warn] skip {p.name}: {e}", file=sys.stderr)
    return items


def rank(
    results: list[Result],
    baseline_name: str = "pytorch_fp32",
    max_map_drop_pct: Optional[float] = 1.0,
    min_fps_bs1: Optional[float] = None,
    ignore_accuracy: bool = False,
) -> tuple[list[dict], Optional[Result]]:
    baseline = next((r for r in results if r.recipe == baseline_name), None)
    base_map = _safe(baseline.accuracy.map_50) if baseline else None

    rows: list[dict] = []
    for r in results:
        fps_bs1 = _safe(r.throughput_fps.bs1)
        map_50 = _safe(r.accuracy.map_50)
        drop_pp = None
        if base_map is not None and map_50 is not None:
            drop_pp = (base_map - map_50) * 100.0  # mAP50 is 0..1, show in %p

        meets = True
        reasons: list[str] = []
        if not ignore_accuracy and max_map_drop_pct is not None and drop_pp is not None:
            if drop_pp > max_map_drop_pct:
                meets = False
                reasons.append(f"mAP drop {drop_pp:.2f}%p > {max_map_drop_pct}%p")
        if min_fps_bs1 is not None and fps_bs1 is not None:
            if fps_bs1 < min_fps_bs1:
                meets = False
                reasons.append(f"fps {fps_bs1:.1f} < {min_fps_bs1}")
        required_missing = fps_bs1 is None or (not ignore_accuracy and map_50 is None)
        if required_missing:
            meets = False
            reasons.append("missing measurements")

        rows.append(
            {
                "recipe": r.recipe,
                "fps_bs1": fps_bs1,
                "fps_bs8": _safe(r.throughput_fps.bs8),
                "p50_ms": _safe(r.latency_ms.p50),
                "map_50": map_50,
                "map_50_95": _safe(r.accuracy.map_50_95),
                "drop_pp": drop_pp,
                "peak_mem_mb": _safe(r.peak_gpu_mem_mb),
                "meets": meets,
                "reasons": reasons,
                "notes": r.notes,
            }
        )

    # Rank: meets-constraints first, then by fps_bs1 desc.
    rows.sort(
        key=lambda r: (
            0 if r["meets"] else 1,
            -(r["fps_bs1"] or 0.0),
        )
    )
    return rows, baseline


def format_report(rows: list[dict], baseline: Optional[Result]) -> str:
    lines: list[str] = []
    lines.append("# OmniOptimizer Report\n")
    if baseline:
        lines.append(f"Baseline: `{baseline.recipe}`  |  GPU: `{baseline.env.gpu}`  |  CUDA: `{baseline.env.cuda}`\n")

    lines.append("")
    lines.append("| Rank | Recipe | fps(bs1) | fps(bs8) | p50 ms | mAP@0.5 | drop | mem MB | meets? |")
    lines.append("|-----:|--------|---------:|---------:|-------:|--------:|-----:|-------:|:------:|")

    def fmt(v, fmt_str="{:.2f}"):
        if v is None:
            return "—"
        try:
            return fmt_str.format(v)
        except Exception:
            return str(v)

    winner = None
    for i, row in enumerate(rows, 1):
        if winner is None and row["meets"]:
            winner = row
        lines.append(
            "| {rank} | `{recipe}` | {fps1} | {fps8} | {p50} | {map50} | {drop} | {mem} | {ok} |".format(
                rank=i,
                recipe=row["recipe"],
                fps1=fmt(row["fps_bs1"], "{:.1f}"),
                fps8=fmt(row["fps_bs8"], "{:.1f}"),
                p50=fmt(row["p50_ms"], "{:.2f}"),
                map50=fmt(row["map_50"], "{:.3f}"),
                drop=fmt(row["drop_pp"], "{:+.2f}%p") if row["drop_pp"] is not None else "—",
                mem=fmt(row["peak_mem_mb"], "{:.0f}"),
                ok="✔" if row["meets"] else "✘",
            )
        )

    lines.append("")
    if winner:
        lines.append(f"## Recommendation\n")
        lines.append(
            f"**`{winner['recipe']}`** — fps {fmt(winner['fps_bs1'], '{:.1f}')} (bs1), "
            f"mAP@0.5 {fmt(winner['map_50'], '{:.3f}')}, "
            f"drop {fmt(winner['drop_pp'], '{:+.2f}%p')}."
        )
    else:
        lines.append("## Recommendation\n")
        lines.append(
            "_No recipe satisfied all constraints._ Loosen `max_map_drop_pct` or `min_fps_bs1` in "
            "the recipe YAMLs, or investigate the failures listed below."
        )

    # Failure / note dump
    issues = [r for r in rows if not r["meets"] or r["notes"]]
    if issues:
        lines.append("")
        lines.append("## Issues")
        for r in issues:
            parts = []
            if r["reasons"]:
                parts.append("; ".join(r["reasons"]))
            if r["notes"]:
                parts.append(str(r["notes"]))
            lines.append(f"- `{r['recipe']}`: {' | '.join(parts) or 'unknown'}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--baseline", default="pytorch_fp32")
    ap.add_argument("--max-map-drop-pct", type=float, default=1.0)
    ap.add_argument("--min-fps-bs1", type=float, default=None)
    ap.add_argument("--ignore-accuracy", action="store_true",
                    help="rank by latency/fps only; use when mAP wasn't measured")
    args = ap.parse_args()

    results = load_results(Path(args.results_dir))
    if not results:
        print(f"no results found in {args.results_dir}", file=sys.stderr)
        return 1

    rows, baseline = rank(
        results,
        baseline_name=args.baseline,
        max_map_drop_pct=args.max_map_drop_pct,
        min_fps_bs1=args.min_fps_bs1,
        ignore_accuracy=args.ignore_accuracy,
    )
    report = format_report(rows, baseline)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
