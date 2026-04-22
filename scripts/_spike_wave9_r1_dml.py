"""Wave 9 Task 0 spike — ORT DirectML EP 환경 검증.

실행: .venv_dml/Scripts/python.exe scripts/_spike_wave9_r1_dml.py

검증:
  Step 1. onnxruntime / DmlExecutionProvider 감지
  Step 2. best_qr ONNX (FP32 bs=1) DML 추론 성공
  Step 3. warmup/measure fps (simple timing)
  Step 4. FP16 ONNX 시도 (있으면)

비교: results_qr/05_trt_fp16.json, results_cpu_qr/30_ort_cpu_fp32.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parent.parent
ONNX_FP32 = REPO_ROOT / "results" / "_onnx" / "best_qr_640_fp32_bs1.onnx"
ONNX_FP16 = REPO_ROOT / "results" / "_onnx" / "best_qr_640_fp16_bs1.onnx"


def probe_providers() -> list[str]:
    providers = ort.get_available_providers()
    print(f"[step1] ort version: {ort.__version__}")
    print(f"[step1] available providers: {providers}")
    return providers


def bench_session(onnx_path: Path, provider: str, *, warmup: int = 30, measure: int = 100) -> dict:
    print(f"\n[step2] loading {onnx_path.name} with {provider}")
    sess = ort.InferenceSession(str(onnx_path), providers=[provider, "CPUExecutionProvider"])
    active = sess.get_providers()
    print(f"  active providers (ordered): {active}")
    if active[0] != provider:
        print(f"  WARN: {provider} not primary — fell back to {active[0]}")

    input_name = sess.get_inputs()[0].name
    input_shape = tuple(sess.get_inputs()[0].shape)
    print(f"  input: {input_name} shape={input_shape}")

    # static shape 가정 (bs=1)
    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
    outs = sess.run(None, {input_name: dummy})
    print(f"  outputs: {[o.shape for o in outs]}")

    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    samples = []
    for _ in range(measure):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95)]
    p99 = samples[int(len(samples) * 0.99)]
    fps = 1000.0 / p50 if p50 > 0 else 0
    print(f"  p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms  fps_bs1={fps:.1f}")
    return {
        "onnx": onnx_path.name, "provider": provider, "active": active,
        "p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "fps_bs1": fps,
        "warmup": warmup, "measure": measure,
    }


def main() -> int:
    providers = probe_providers()

    has_dml = "DmlExecutionProvider" in providers
    if not has_dml:
        print("[FAIL] DmlExecutionProvider not available. venv_dml 설치 확인 필요.")
        return 1
    print("[ok] DmlExecutionProvider detected")

    assert ONNX_FP32.exists(), f"missing {ONNX_FP32}"

    results = {"providers": providers}

    # Step 2-3: DML FP32
    results["dml_fp32"] = bench_session(ONNX_FP32, "DmlExecutionProvider")

    # 비교: CPU FP32
    results["cpu_fp32"] = bench_session(ONNX_FP32, "CPUExecutionProvider")

    # Step 4: FP16 (있으면)
    if ONNX_FP16.exists():
        try:
            results["dml_fp16"] = bench_session(ONNX_FP16, "DmlExecutionProvider")
        except Exception as e:
            print(f"[fp16] CRASH: {type(e).__name__}: {e}")
            results["dml_fp16"] = {"status": "CRASH", "error": f"{type(e).__name__}: {str(e)[:200]}"}

    # Summary
    print("\n" + "=" * 70)
    print("WAVE 9 TASK 0 SPIKE SUMMARY")
    print("=" * 70)
    print(f"{'run':<18} {'p50 ms':<10} {'fps_bs1':<10} {'active':<40}")
    for k, v in results.items():
        if k == "providers" or not isinstance(v, dict) or "p50_ms" not in v:
            continue
        print(f"{k:<18} {v['p50_ms']:<10.2f} {v['fps_bs1']:<10.1f} {str(v['active']):<40}")

    print("\nREFERENCE (기존 reports):")
    print("  trt_fp16 (GPU):      fps_bs1=435.1 (report_qr.md)")
    print("  ort_cpu_fp32 (CPU):  fps_bs1=14.4  (report_cpu_qr.md)")

    out = REPO_ROOT / "logs" / "wave9_r1_dml.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[spike] saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
