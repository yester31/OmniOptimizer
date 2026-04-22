"""Wave 9 Task 0 Step 2 — DML device_id sweep.

이 랩탑은 RTX 3060 Laptop (NVIDIA) + Intel Xe iGPU 조합.
DML 는 DirectX 12 어댑터 열거 순서로 device_id 배정.
RTX 가 primary (0) 면 Intel iGPU 는 1 일 가능성.
각 device 로 load + benchmark.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parent.parent
ONNX_FP32 = REPO_ROOT / "results" / "_onnx" / "best_qr_640_fp32_bs1.onnx"


def bench_device(device_id: int, *, warmup: int = 20, measure: int = 60) -> dict:
    print(f"\n=== device_id={device_id} ===")
    try:
        sess = ort.InferenceSession(
            str(ONNX_FP32),
            providers=[
                ("DmlExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ],
        )
    except Exception as e:  # noqa: BLE001
        print(f"  CRASH: {type(e).__name__}: {e}")
        return {"device_id": device_id, "status": "CRASH", "error": f"{type(e).__name__}: {str(e)[:150]}"}

    input_name = sess.get_inputs()[0].name
    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {input_name: dummy})
    samples = []
    for _ in range(measure):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p50 = samples[len(samples) // 2]
    fps = 1000.0 / p50 if p50 > 0 else 0
    print(f"  p50={p50:.2f}ms  fps={fps:.1f}  providers={sess.get_providers()}")
    return {"device_id": device_id, "status": "OK", "p50_ms": p50, "fps_bs1": fps}


def main() -> int:
    print(f"ort {ort.__version__}")
    results = []
    for dev in [0, 1, 2]:
        r = bench_device(dev)
        results.append(r)
        if r.get("status") == "CRASH":
            break  # 더 큰 device_id 도 없을 것

    print("\n=== SUMMARY ===")
    print(f"{'device_id':<12} {'status':<10} {'p50 ms':<10} {'fps_bs1':<10}")
    for r in results:
        if r.get("status") == "OK":
            print(f"{r['device_id']:<12} {r['status']:<10} {r['p50_ms']:<10.2f} {r['fps_bs1']:<10.1f}")
        else:
            print(f"{r['device_id']:<12} {r.get('status', '?'):<10}  {r.get('error', '')[:60]}")
    return 0


if __name__ == "__main__":
    main()
