"""Phase 9 — Wave 10 재오픈 ship: mAP 측정 + results/*.json 생성.

입력:  Phase 7/8 의 engine + bench JSON.
출력:  results/23_modelopt_fastnas_int8.json
      results/24_modelopt_fastnas_sp_int8.json
      → report_qr.md 재생성 대상.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._schemas import (  # noqa: E402
    AccuracyStats, EnvInfo, LatencyStats, Result, ThroughputStats,
)


# ---------- engine bench (Phase 8 결과 재사용) ----------
P8_JSON = REPO_ROOT / "logs" / "wave10_p8_modelopt_onnx.json"
DATA_YAML = REPO_ROOT / "qr_barcode.yaml"


RECIPES = {
    # recipe name → (phase-8 key, engine path, weights label)
    "modelopt_fastnas_int8": {
        "p8_key": "E_fastnas_modelopt_int8",
        "engine": REPO_ROOT / "trained_weights" / "23_fair_bench_engines" / "E_fastnas_modelopt_int8_bs1.engine",
    },
    "modelopt_fastnas_sp_int8": {
        "p8_key": "F_fastnas_sp_modelopt_int8",
        "engine": REPO_ROOT / "trained_weights" / "23_fair_bench_engines" / "F_fastnas_sp_modelopt_int8_bs1.engine",
    },
}


def _env() -> EnvInfo:
    import torch
    try:
        import tensorrt
        trt_v = tensorrt.__version__
    except Exception:
        trt_v = None
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return EnvInfo(
        gpu=gpu_name,
        cuda=torch.version.cuda if torch.cuda.is_available() else None,
        tensorrt=trt_v,
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        torch=torch.__version__,
    )


def measure_map(engine_path: Path) -> tuple[float, float]:
    """Use ultralytics to val the TRT engine and return (map50, map50_95)."""
    from ultralytics import YOLO
    yolo = YOLO(str(engine_path))
    metrics = yolo.val(
        data=str(DATA_YAML), imgsz=640, batch=1, device=0,
        half=False, int8=False,
        verbose=False, plots=False, save=False, workers=0,
    )
    return float(metrics.box.map50), float(metrics.box.map)


def main() -> int:
    assert P8_JSON.exists(), f"missing {P8_JSON}"
    p8 = json.loads(P8_JSON.read_text())
    env = _env()
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    baseline_map = 0.988  # ultralytics val of plain YOLO26n QR baseline (report_cpu_qr)
    baseline_fps = {"modelopt_int8_entropy": 763.9}

    produced = []
    for name, spec in RECIPES.items():
        p8_key = spec["p8_key"]
        engine = spec["engine"]
        r = p8.get(p8_key, {})
        if r.get("status") != "OK":
            print(f"[skip] {name}: Phase 8 status={r.get('status')}")
            continue
        print(f"\n=== {name} ===")
        print(f"  engine: {engine.name}  ({engine.stat().st_size/1e6:.2f}MB)")
        print(f"  p50={r['p50_ms']:.3f}  fps={r['fps_bs1']:.1f}")

        # mAP 측정 (engine 기반, Phase 1B 패턴)
        print(f"  measuring mAP...")
        t0 = time.time()
        map50, map50_95 = measure_map(engine)
        print(f"  mAP@0.5 = {map50:.4f}   mAP@0.5:0.95 = {map50_95:.4f}   ({time.time()-t0:.1f}s)")

        drop_pct = (baseline_map - map50) * 100.0
        constraints_ok = drop_pct <= 5.0 and r["fps_bs1"] >= 30.0

        started = finished = datetime.now(timezone.utc).isoformat()
        result = Result(
            recipe=name,
            started_at=started,
            finished_at=finished,
            env=env,
            model_size_mb=r["engine_size_mb"],
            latency_ms=LatencyStats(
                p50=r["p50_ms"],
                p95=r.get("p95_ms"),
                p99=r.get("p99_ms"),
                p50_gpu=r.get("p50_gpu_ms"),
                stddev_ms=r.get("stddev_ms"),
            ),
            throughput_fps=ThroughputStats(bs1=r["fps_bs1"], bs8=None),
            peak_gpu_mem_mb=r.get("peak_gpu_mem_mb_torch"),
            peak_gpu_mem_mb_nvml_delta=r.get("peak_gpu_mem_mb_nvml_delta"),
            cold_start_ms=None,
            accuracy=AccuracyStats(map_50=map50, map_50_95=map50_95),
            meets_constraints=constraints_ok,
            notes=(
                "Wave 10 reopened 2026-04-22. FastNAS pruning (15.7% FLOPs) pre-applied "
                "+ QR fine-tuned. modelopt.onnx.quantize INT8 entropy QDQ (223 Q / 223 DQ, "
                "Detect head 42 Q/DQ pairs). Engine 4.72MB (-88% vs baseline 38MB). "
                f"mAP drop vs baseline 0.988: {drop_pct:.2f}%p. "
                "bs=8 not measured (engine built bs=1 only)."
            ),
        )
        # write results
        idx = 23 if name == "modelopt_fastnas_int8" else 24
        out_path = results_dir / f"{idx:02d}_{name}.json"
        out_path.write_text(json.dumps(json.loads(result.model_dump_json()), indent=2))
        print(f"  saved → {out_path}")
        produced.append(out_path)

    print("\n=== Summary ===")
    for p in produced:
        print(f"  {p}")
    print("\nNext: python scripts/recommend.py --results results/ --out report_qr.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
