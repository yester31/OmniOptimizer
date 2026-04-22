"""Phase 6 - FastNAS variants 의 GPU VRAM 측정.

각 TRT engine 을 별도 subprocess 에서 로드하고 1회 inference 수행.
pynvml 로 process 메모리 델타 측정.

측정 대상:
  (A) FastNAS FP16       best.engine                       (Phase 1B)
  (B) FastNAS INT8       best_int8.engine                  (Phase 1B)
  (C) FastNAS + sparsity FT FP16   B_final_fp16.engine    (Phase 4C)
  (D) FastNAS + sparsity + INT8 chain  B_final_int8.engine (Phase 4C)

baseline (report_qr.md): 38 MB (공통; report는 torch process memory)

측정 방법:
  - subprocess Python process 1 start
  - load engine via ultralytics YOLO + 1 forward
  - query nvml process mem used by PID
  - 종료 후 total subprocess peak
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path


# Windows cp949 콘솔 회피 — UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


REPO_ROOT = Path(__file__).resolve().parent.parent

TARGETS = {
    "A_fastnas_fp16": REPO_ROOT / "trained_weights" / "23_fastnas_p1_finetune" / "weights" / "best.engine",
    "B_fastnas_int8": REPO_ROOT / "trained_weights" / "23_fastnas_p1_finetune" / "weights" / "best_int8.engine",
    "C_fastnas_sp_fp16": REPO_ROOT / "trained_weights" / "23_fastnas_chain_ft" / "B_final_fp16.engine",
    "D_fastnas_sp_int8_chain": REPO_ROOT / "trained_weights" / "23_fastnas_chain_ft" / "B_final_int8.engine",
}


# subprocess 로 실행할 mini-script - pynvml 로 자기 PID 메모리 측정
RUNNER_SCRIPT = r"""
import os, sys, time, json
import pynvml
from ultralytics import YOLO
import torch

engine = sys.argv[1]
label = sys.argv[2]

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
pid = os.getpid()

def proc_mem_mb():
    # process-specific mem (Windows 에서 None 나오는 드라이버 있음)
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for p in procs:
            if p.pid == pid and p.usedGpuMemory is not None:
                return p.usedGpuMemory / (1024*1024)
    except Exception:
        pass
    return None

def device_mem_mb():
    # device-wide used memory — 다른 프로세스 간섭 가능하나 robust
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024*1024)

t0 = time.time()
mem_start_proc = proc_mem_mb()
mem_start_dev = device_mem_mb()

yolo = YOLO(engine)
# warmup 3 + inference 10 frames
import numpy as np
dummy = (np.random.rand(640, 640, 3) * 255).astype("uint8")
for _ in range(3):
    yolo(dummy, verbose=False)
torch.cuda.synchronize()
mem_warmup_proc = proc_mem_mb()
mem_warmup_dev = device_mem_mb()

peak_proc = mem_warmup_proc or 0.0
peak_dev = mem_warmup_dev
for _ in range(10):
    yolo(dummy, verbose=False)
    torch.cuda.synchronize()
    p = proc_mem_mb()
    if p is not None:
        peak_proc = max(peak_proc, p)
    peak_dev = max(peak_dev, device_mem_mb())

dur = time.time() - t0

out = {
    "label": label,
    "engine_file": engine,
    "engine_size_mb": os.path.getsize(engine) / (1024*1024),
    "gpu_mem_start_proc_mb": mem_start_proc,
    "gpu_mem_start_device_mb": mem_start_dev,
    "gpu_mem_warmup_proc_mb": mem_warmup_proc,
    "gpu_mem_warmup_device_mb": mem_warmup_dev,
    "gpu_mem_peak_proc_mb": peak_proc,
    "gpu_mem_peak_device_mb": peak_dev,
    "gpu_mem_delta_device_mb": peak_dev - mem_start_dev,
    "duration_s": dur,
    "cuda_memory_allocated_mb": torch.cuda.memory_allocated() / (1024*1024),
    "cuda_memory_reserved_mb": torch.cuda.memory_reserved() / (1024*1024),
    "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024*1024),
}
print("RESULT_JSON:" + json.dumps(out))
"""


def run_one(engine_path: Path, label: str) -> dict:
    print(f"\n=== {label} ===  ({engine_path.name})")
    assert engine_path.exists(), f"missing {engine_path}"
    # subprocess 로 실행
    try:
        proc = subprocess.run(
            [sys.executable, "-c", RUNNER_SCRIPT, str(engine_path), label],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"label": label, "status": "TIMEOUT"}
    # stdout 에서 RESULT_JSON 추출
    out_line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON:")), None)
    if out_line is None:
        print(f"  CRASH - no RESULT_JSON\n  stdout tail:\n{proc.stdout[-1000:]}\n  stderr tail:\n{proc.stderr[-1000:]}")
        return {"label": label, "status": "CRASH",
                "stdout_tail": proc.stdout[-500:], "stderr_tail": proc.stderr[-500:]}
    result = json.loads(out_line[len("RESULT_JSON:"):])
    result["status"] = "OK"
    print(f"  engine {result['engine_size_mb']:.2f}MB  "
          f"nvml device delta={result['gpu_mem_delta_device_mb']:.1f}MB  "
          f"peak_device={result['gpu_mem_peak_device_mb']:.1f}MB  "
          f"torch.cuda max_alloc={result['cuda_max_memory_allocated_mb']:.1f}MB")
    return result


def main() -> int:
    results = {}
    for key, engine in TARGETS.items():
        if not engine.exists():
            print(f"SKIP {key}: missing {engine}")
            results[key] = {"status": "MISSING", "engine_file": str(engine)}
            continue
        results[key] = run_one(engine, key)

    print("\n" + "=" * 70)
    print("PHASE 6 SUMMARY - GPU memory per variant")
    print("=" * 70)
    print(f"{'variant':<30} {'engine MB':<11} {'dev delta MB':<13} {'dev peak MB':<13} {'cuda max MB':<12}")
    for k, r in results.items():
        if r.get("status") == "OK":
            print(f"{k:<30} {r['engine_size_mb']:<11.2f} "
                  f"{r['gpu_mem_delta_device_mb']:<13.1f} "
                  f"{r['gpu_mem_peak_device_mb']:<13.1f} "
                  f"{r['cuda_max_memory_allocated_mb']:<12.1f}")
        else:
            print(f"{k:<30} {r.get('status', '?')}  "
                  f"{r.get('engine_file', '')}")

    print("\nREFERENCE - report_qr.md mem MB 는 torch process memory (38MB):")
    print("  modelopt_int8_entropy  mem=38  fps=763.9  mAP=0.987")
    print("  trt_fp16              mem=38  fps=435.1  mAP=0.989")

    out_json = REPO_ROOT / "logs" / "wave10_p6_gpu_mem.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[p6] saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
