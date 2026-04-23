"""Wave 15 helper - diff baseline vs new Result JSON for the D2 A/B table.

Not a general tool. Specific to Wave 15's "archive + rerun + compare" flow.
Remove (or fold into recommend.py) once Wave 15 ships.

Usage:
    python scripts/_compare_wave15.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results_qr" / "_pre_wave15"
NEW = ROOT / "results_qr"

RECIPES = [
    "09_modelopt_int8_entropy",
    "12_modelopt_int8_mixed",
    "42_modelopt_int8_asymmetric",
]


def _load(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _row(name: str, d: dict) -> dict:
    t = d.get("throughput_fps") or {}
    a = d.get("accuracy") or {}
    return {
        "name": name,
        "bs1": t.get("bs1"),
        "bs8": t.get("bs8"),
        "map50": a.get("map_50"),
        "build_time_s": d.get("build_time_s"),
        "meets": d.get("meets_constraints"),
    }


def _fmt(v, d=1):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "✔" if v else "✘"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def _delta(new, base, pct=False):
    if new is None or base is None or base == 0:
        return "-"
    if pct:
        return f"{(new - base) / base * 100:+.1f}%"
    return f"{(new - base):+.4f}"


def main():
    print("=" * 100)
    print(f"{'Recipe':<30} {'opt':<4} {'bs1':>9} {'bs8':>9} {'mAP50':>8} {'build_s':>9} {'Δfps%':>8} {'ΔmAP':>9}")
    print("=" * 100)
    for r in RECIPES:
        base = _row(r, _load(BASE / f"{r}.json"))
        new = _row(r, _load(NEW / f"{r}.json"))
        print(
            f"{r:<30} {'3':<4} {_fmt(base['bs1']):>9} {_fmt(base['bs8']):>9} "
            f"{_fmt(base['map50'], 4):>8} {_fmt(base['build_time_s'], 1):>9} {'-':>8} {'-':>9}"
        )
        print(
            f"{'':<30} {'5':<4} {_fmt(new['bs1']):>9} {_fmt(new['bs8']):>9} "
            f"{_fmt(new['map50'], 4):>8} {_fmt(new['build_time_s'], 1):>9} "
            f"{_delta(new['bs1'], base['bs1'], pct=True):>8} "
            f"{_delta(new['map50'], base['map50']):>9}"
        )
    print("=" * 100)
    print("\nAccept criteria: fps delta ≥ +3% AND mAP delta ≥ -0.003 (−0.3%p) AND build_time_s ≤ 1200")
    print()
    for r in RECIPES:
        base = _row(r, _load(BASE / f"{r}.json"))
        new = _row(r, _load(NEW / f"{r}.json"))
        if new["bs1"] is None:
            print(f"{r}: PENDING (new JSON not yet produced)")
            continue
        fps_pass = base["bs1"] and new["bs1"] >= base["bs1"] * 1.03
        map_pass = (new["map50"] or 0) >= (base["map50"] or 0) - 0.003
        build_pass = (new["build_time_s"] is None) or (new["build_time_s"] <= 1200)
        verdict = "ACCEPT" if (fps_pass and map_pass and build_pass) else "ROLLBACK"
        print(
            f"{r}: fps={'PASS' if fps_pass else 'FAIL'} "
            f"mAP={'PASS' if map_pass else 'FAIL'} "
            f"build={'PASS' if build_pass else 'FAIL'} → {verdict}"
        )


if __name__ == "__main__":
    main()
