"""Brevitas QDQ 커버리지 감사 — engine build 중 발생한 "Missing scale" 경고 수집.

가설: brevitas는 weight에만 DequantizeLinear 주입 (Q=92, DQ=204 비대칭).
TRT는 activation tensor에 scale 없을 때 INT8 fallback 불가 → FP16 혼합 실행 → fps 손실.

검증:
  1. brevitas QDQ ONNX → 새 engine build (stderr 에서 "Missing scale" grep)
  2. 동일 FP32 base로 modelopt.onnx.quantize → engine build → warning 수 비교
  3. 결과 비교 + brevitas 개선 가능성 판단
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OMNI_COCO_YAML", str(REPO_ROOT / "qr_barcode.yaml"))

TARGETS = {
    "brev_percentile": REPO_ROOT / "results" / "_onnx" / "best_qr_640_brev_percentile_512_s42_bs1.qdq.onnx",
    "brev_mse": REPO_ROOT / "results" / "_onnx" / "best_qr_640_brev_mse_512_s42_bs1.qdq.onnx",
    "modelopt_entropy": REPO_ROOT / "results" / "_onnx" / "best_qr_640_modelopt_entropy_bs1.onnx",
}
TMP_ENGINE_DIR = REPO_ROOT / "trained_weights" / "23_fair_bench_engines" / "audit"


def build_and_count_warnings(label: str, onnx_path: Path) -> dict:
    TMP_ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    engine_path = TMP_ENGINE_DIR / f"{label}_audit.engine"
    if engine_path.exists():
        engine_path.unlink()

    # subprocess 에서 _build_engine 호출, stderr 캡처
    runner = f"""
import sys
sys.path.insert(0, r'{REPO_ROOT}')
from pathlib import Path
from scripts.run_trt import _build_engine
built, err = _build_engine(
    onnx_path=Path(r'{onnx_path}'),
    engine_path=Path(r'{engine_path}'),
    dtype='int8',
    sparsity=None,
    batch_size=1,
    imgsz=640,
    calib_samples=0,
    calib_seed=42,
    quant_preapplied=True,
    enable_tf32=False,
)
print('BUILT:' + ('OK' if built else f'FAIL({{err}})'))
"""
    proc = subprocess.run(
        [sys.executable, "-c", runner], capture_output=True, text=True, timeout=600,
    )
    stderr = proc.stderr
    # Warning 개수
    missing_scale = re.findall(r"Missing scale and zero-point for tensor (\S+)", stderr)
    missing_set = set(missing_scale)
    # Detect head 관련 (model.23.*)
    detect_missing = [m for m in missing_set if "model.23" in m or "23_" in m]
    built_ok = "BUILT:OK" in proc.stdout
    size_mb = engine_path.stat().st_size / 1e6 if engine_path.exists() else 0
    print(f"\n=== {label} ===")
    print(f"  onnx: {onnx_path.name}")
    print(f"  built: {built_ok}  size: {size_mb:.2f}MB")
    print(f"  Missing scale warnings: {len(missing_set)} unique tensors")
    print(f"  Detect-head 관련 ({'model.23' in str(missing_set)}): {len(detect_missing)}")
    if detect_missing[:5]:
        print(f"  sample: {detect_missing[:5]}")
    return {
        "label": label,
        "onnx_file": onnx_path.name,
        "built": built_ok,
        "engine_size_mb": size_mb,
        "missing_scale_warnings": len(missing_set),
        "detect_head_missing_scale": len(detect_missing),
        "missing_sample": sorted(missing_set)[:10],
    }


def main() -> int:
    results = {}
    for k, p in TARGETS.items():
        results[k] = build_and_count_warnings(k, p)

    print("\n" + "=" * 80)
    print("SUMMARY - QDQ coverage audit")
    print("=" * 80)
    print(f"{'target':<25} {'missing_scale':<15} {'detect_head_missing':<22} {'engine MB':<11}")
    for k, r in results.items():
        print(f"{k:<25} {r['missing_scale_warnings']:<15} "
              f"{r['detect_head_missing_scale']:<22} {r['engine_size_mb']:<11.2f}")

    out = REPO_ROOT / "logs" / "brevitas_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[audit] saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
