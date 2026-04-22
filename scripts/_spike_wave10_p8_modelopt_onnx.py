"""Phase 8 — FastNAS ONNX 에 modelopt.onnx.quantize 적용 (baseline 과 동일 전략).

Phase 7 에서 FastNAS + TRT native INT8 calibrator 는 fps 502 로 baseline
modelopt_int8_entropy(fps 763.9) 대비 -34%. 원인: build log 의 Detect head
(`model.23`) 19+ tensor 가 "Missing scale and zero-point" → INT8 fallback 불가.

해결 전략 (scripts/run_trt.py::_prepare_modelopt_onnx 재사용):
  - clean FP32 ONNX 를 modelopt.onnx.quantization.quantize 에 넘겨 **QDQ 노드 주입**
  - 결과 ONNX 는 activation / weight scale 모두 포함 → TRT 가 quant_preapplied 로 읽음
  - _build_engine(dtype="int8", quant_preapplied=True)  → pure INT8 실행

실측 대상:
  (E) FastNAS + modelopt INT8 entropy           best.onnx → QDQ → engine
  (F) FastNAS + sparsity + modelopt INT8 entropy B_final.onnx → QDQ → engine + sparsity="2:4"

비교 (baseline QDQ 커버리지):
  - best_qr_640_modelopt_entropy_bs1.onnx (modelopt_int8_entropy) 의 QuantizeLinear / DequantizeLinear 노드 개수
  - FastNAS QDQ ONNX 의 Q/DQ 개수
  - 차이 = Detect head / pruned layer 에 대한 scale 주입 완성도
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OMNI_COCO_YAML", str(REPO_ROOT / "qr_barcode.yaml"))

from scripts._weights_io import _build_calib_numpy, _export_onnx  # noqa: E402
from scripts.measure import measure_latency, throughput_from_latency  # noqa: E402
from scripts.run_trt import _build_engine, _make_trt_forward  # noqa: E402


CACHE_DIR = REPO_ROOT / "trained_weights" / "23_fair_bench_engines"
ONNX_CACHE = CACHE_DIR / "onnx"
QDQ_CACHE = CACHE_DIR / "qdq_onnx"
ENGINE_CACHE = CACHE_DIR

VARIANTS = {
    "E_fastnas_modelopt_int8": {
        "pt": REPO_ROOT / "trained_weights" / "23_fastnas_p1_finetune" / "weights" / "best.pt",
        "sparsity": None,
        "calibrator": "entropy",
    },
    "F_fastnas_sp_modelopt_int8": {
        "pt": REPO_ROOT / "trained_weights" / "23_fastnas_chain_ft" / "B_final.pt",
        "sparsity": "2:4",
        "calibrator": "entropy",
    },
}


def count_qdq_nodes(onnx_path: Path) -> dict:
    import onnx
    m = onnx.load(str(onnx_path))
    counts = {"Quantize": 0, "Dequantize": 0, "total_nodes": len(m.graph.node)}
    detect_q = 0
    detect_dq = 0
    for n in m.graph.node:
        if n.op_type == "QuantizeLinear":
            counts["Quantize"] += 1
            if "model.23" in (n.name or ""):
                detect_q += 1
        elif n.op_type == "DequantizeLinear":
            counts["Dequantize"] += 1
            if "model.23" in (n.name or ""):
                detect_dq += 1
    counts["detect_head_quantize"] = detect_q
    counts["detect_head_dequantize"] = detect_dq
    return counts


def build_qdq_onnx(label: str, spec: dict, imgsz: int = 640) -> Path:
    print(f"\n=== [{label}] step 1: export clean FP32 ONNX + modelopt.onnx.quantize ===")
    ONNX_CACHE.mkdir(parents=True, exist_ok=True)
    QDQ_CACHE.mkdir(parents=True, exist_ok=True)

    # dynamic FP32 ONNX 재export
    clean_onnx = _export_onnx(
        str(spec["pt"]), imgsz=imgsz, half=False,
        cache_dir=ONNX_CACHE, dynamic=True,
    )
    print(f"  clean ONNX: {clean_onnx.name}  ({clean_onnx.stat().st_size/1e6:.2f}MB)")

    # modelopt.onnx.quantize 로 QDQ 주입
    from modelopt.onnx.quantization import quantize as moq_quantize
    qdq_path = QDQ_CACHE / f"{label}_qdq.onnx"
    if qdq_path.exists():
        qdq_path.unlink()
    calib = _build_calib_numpy(
        str(REPO_ROOT / "qr_barcode.yaml"),
        n_samples=512,
        imgsz=imgsz,
        seed=42,
    )
    print(f"  calib shape: {calib.shape}  dtype={calib.dtype}")
    t0 = time.time()
    moq_quantize(
        onnx_path=str(clean_onnx),
        quantize_mode="int8",
        calibration_method=spec["calibrator"],
        calibration_data=calib,
        output_path=str(qdq_path),
        log_level="WARNING",
    )
    print(f"  QDQ ONNX: {qdq_path.name}  ({qdq_path.stat().st_size/1e6:.2f}MB)  built {time.time()-t0:.1f}s")

    # QDQ 노드 수 audit
    counts = count_qdq_nodes(qdq_path)
    print(f"  QDQ nodes: Q={counts['Quantize']}  DQ={counts['Dequantize']}  "
          f"model.23 Q={counts['detect_head_quantize']} DQ={counts['detect_head_dequantize']}")
    return qdq_path, counts


def build_engine_and_bench(label: str, qdq_onnx: Path, sparsity: str | None,
                           batch_size: int = 1, imgsz: int = 640) -> dict:
    print(f"\n=== [{label}] step 2: TRT engine build + bench ===")
    engine_path = ENGINE_CACHE / f"{label}_bs{batch_size}.engine"
    if engine_path.exists():
        engine_path.unlink()

    # "Missing scale" 경고 수 카운트 위해 stderr 캡처 — _build_engine 내부 TRT warning
    # 을 수집하려면 subprocess 실행 + stderr grep 으로 우회.
    # 간단히 직접 호출 후 return.
    t0 = time.time()
    built, err = _build_engine(
        onnx_path=qdq_onnx, engine_path=engine_path,
        dtype="int8", sparsity=sparsity,
        batch_size=batch_size, imgsz=imgsz,
        calib_samples=0, calib_seed=42,
        quant_preapplied=True, enable_tf32=False,
    )
    build_dur = time.time() - t0
    if built is None:
        print(f"  BUILD FAIL: {err}")
        return {"label": label, "status": "CRASH_BUILD", "error": (err or "")[:400]}
    print(f"  engine built: {engine_path.name}  ({engine_path.stat().st_size/1e6:.2f}MB)  {build_dur:.1f}s")

    # bench
    fwd, _engine = _make_trt_forward(engine_path, batch_size, imgsz)
    stats = measure_latency(fwd, warmup_iters=100, measure_iters=100)
    fps = throughput_from_latency(stats["p50"], batch_size)
    print(f"  p50={stats['p50']:.3f}ms  p50_gpu={(stats.get('p50_gpu') or 0):.3f}ms  fps={fps:.1f}  "
          f"peak_torch={(stats.get('peak_gpu_mem_mb') or 0):.1f}MB  "
          f"NVML delta={(stats.get('peak_gpu_mem_mb_nvml_delta') or 0):.1f}MB")
    return {
        "label": label, "status": "OK",
        "engine_file": engine_path.name,
        "engine_size_mb": engine_path.stat().st_size / 1e6,
        "build_duration_s": build_dur,
        "p50_ms": stats["p50"], "p95_ms": stats.get("p95"), "p99_ms": stats.get("p99"),
        "stddev_ms": stats.get("stddev_ms"),
        "p50_gpu_ms": stats.get("p50_gpu"),
        "fps_bs1": fps,
        "peak_gpu_mem_mb_torch": stats.get("peak_gpu_mem_mb"),
        "peak_gpu_mem_mb_nvml_delta": stats.get("peak_gpu_mem_mb_nvml_delta"),
    }


def audit_baseline_qdq_coverage() -> dict:
    """기존 report_qr.md 의 QDQ ONNX 에 대한 QDQ 커버리지 감사."""
    print("\n" + "=" * 70)
    print("BASELINE QDQ coverage audit (existing cached ONNX)")
    print("=" * 70)
    baselines = {
        "modelopt_int8_entropy": REPO_ROOT / "results" / "_onnx" / "best_qr_640_modelopt_entropy_bs1.onnx",
        "modelopt_int8_max": REPO_ROOT / "results" / "_onnx" / "best_qr_640_modelopt_max_bs1.onnx",
        "modelopt_int8_percentile": REPO_ROOT / "results" / "_onnx" / "best_qr_640_modelopt_percentile_bs1.onnx",
        "brevitas_int8_percentile": REPO_ROOT / "results" / "_onnx" / "best_qr_640_brev_percentile_512_s42_bs1.qdq.onnx",
        "brevitas_int8_mse": REPO_ROOT / "results" / "_onnx" / "best_qr_640_brev_mse_512_s42_bs1.qdq.onnx",
    }
    out = {}
    for name, path in baselines.items():
        if not path.exists():
            print(f"  {name:<30} MISSING: {path.name}")
            out[name] = {"status": "MISSING"}
            continue
        c = count_qdq_nodes(path)
        print(f"  {name:<30} Q={c['Quantize']:<4} DQ={c['Dequantize']:<4} "
              f"detect_Q={c['detect_head_quantize']:<3} detect_DQ={c['detect_head_dequantize']:<3} "
              f"total_nodes={c['total_nodes']}")
        out[name] = c
    return out


def main() -> int:
    # Step 0: baseline QDQ 커버리지 audit
    baseline_audit = audit_baseline_qdq_coverage()

    # Step 1: FastNAS ONNX 에 modelopt.onnx.quantize 적용 + bench
    results = {"baseline_qdq_audit": baseline_audit}
    for label, spec in VARIANTS.items():
        qdq_onnx, qdq_counts = build_qdq_onnx(label, spec)
        bench = build_engine_and_bench(label, qdq_onnx, spec["sparsity"])
        bench["qdq_counts"] = qdq_counts
        results[label] = bench

    # Phase 7 과 비교
    print("\n" + "=" * 110)
    print("PHASE 8 SUMMARY - modelopt.onnx.quantize 전략 적용")
    print("=" * 110)
    print(f"{'variant':<32} {'p50 ms':<9} {'p50_gpu':<10} {'fps':<8} {'engine MB':<11} "
          f"{'Q nodes':<9} {'DQ':<6} {'det_Q':<7} {'det_DQ':<7}")
    for k, r in results.items():
        if k == "baseline_qdq_audit":
            continue
        if r.get("status") == "OK":
            qc = r.get("qdq_counts", {})
            print(f"{k:<32} {r['p50_ms']:<9.3f} "
                  f"{(r['p50_gpu_ms'] or 0):<10.3f} "
                  f"{r['fps_bs1']:<8.1f} "
                  f"{r['engine_size_mb']:<11.2f} "
                  f"{qc.get('Quantize', 0):<9} "
                  f"{qc.get('Dequantize', 0):<6} "
                  f"{qc.get('detect_head_quantize', 0):<7} "
                  f"{qc.get('detect_head_dequantize', 0):<7}")
        else:
            print(f"{k:<32} {r.get('status', '?')}  {r.get('error', '')[:70]}")

    print("\n--- vs Phase 7 (TRT native calibrator) ---")
    p7 = REPO_ROOT / "logs" / "wave10_p7_fair_bench.json"
    if p7.exists():
        p7_data = json.loads(p7.read_text())
        for key in ("B_fastnas_int8", "D_fastnas_sp_int8"):
            r = p7_data.get(key, {})
            if r.get("status") == "OK":
                print(f"  Phase7 {key:<27} fps={r['fps_bs1']:.1f}  p50={r['p50_ms']:.3f}ms")

    print("\nREFERENCE (report_qr.md):")
    print(f"  modelopt_int8_entropy    fps=763.9   mAP=0.987")
    print(f"  modelopt_int8_sparsity   fps=439.4   mAP=0.987")

    out_json = REPO_ROOT / "logs" / "wave10_p8_modelopt_onnx.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[p8] saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
