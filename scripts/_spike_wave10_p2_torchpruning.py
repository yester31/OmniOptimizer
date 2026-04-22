"""Phase 2 spike — torch-pruning DepGraph 기반 구조적 pruning.

FastNAS 대비 장점: torch.fx 의존 없음. DependencyGraph로 YOLO backbone 도 prunable.

실험:
  - ratio sweep: [0.20, 0.30, 0.50]  (channel 감축 비율; global pruning)
  - Detect head 보호 (nc + reg_max*4 출력 shape 불변)
  - 결과: pruned 모델 ultralytics pickle 저장 (finetune 가능 포맷)

출력:
  trained_weights/_spike_wave10_tp_{ratio}.pt  (ultralytics pickle)
  logs/wave10_p2_tp_sweep.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch_pruning as tp
from thop import profile as thop_profile
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect


REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_PT = REPO_ROOT / "best_qr.pt"
RATIOS = [0.20, 0.30, 0.50]


def prune_one(ratio: float) -> dict:
    print(f"\n========== ratio = {ratio:.2f} ==========")
    t0 = time.time()
    yolo = YOLO(str(BASE_PT))
    model = yolo.model.cuda().eval()
    dummy = torch.randn(1, 3, 640, 640, device="cuda")

    # warmup — Detect.shape 세팅 (FastNAS spike 와 동일 이유)
    with torch.no_grad():
        _ = model(dummy)

    base_macs, base_params = thop_profile(model, inputs=(dummy,), verbose=False)
    print(f"  original MACs = {base_macs:.3e}  params = {base_params:.3e}")

    # Detect head 출력 Conv 는 pruning 금지 (output shape 고정)
    ignored_layers: list[torch.nn.Module] = []
    for m in model.modules():
        if isinstance(m, Detect):
            ignored_layers.append(m)

    # MagnitudePruner (L2 norm) + global pruning
    imp = tp.importance.MagnitudeImportance(p=2)
    try:
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=dummy,
            importance=imp,
            pruning_ratio=ratio,
            ignored_layers=ignored_layers,
            global_pruning=True,
        )
        pruner.step()
    except Exception as e:  # noqa: BLE001
        print(f"  CRASH during pruner.step(): {type(e).__name__}: {e}")
        return {
            "ratio_target": ratio,
            "status": "CRASH_STEP",
            "error": f"{type(e).__name__}: {str(e)[:300]}",
            "duration_s": time.time() - t0,
        }

    # Pruned 모델 forward 검증
    try:
        with torch.no_grad():
            _ = model(dummy)
    except Exception as e:  # noqa: BLE001
        print(f"  CRASH post-prune forward: {type(e).__name__}: {e}")
        return {
            "ratio_target": ratio,
            "status": "CRASH_FORWARD",
            "error": f"{type(e).__name__}: {str(e)[:300]}",
            "duration_s": time.time() - t0,
        }

    pruned_macs, pruned_params = thop_profile(model, inputs=(dummy,), verbose=False)
    ratio_macs = pruned_macs / base_macs
    ratio_params = pruned_params / base_params
    print(f"  pruned MACs   = {pruned_macs:.3e}  (ratio = {ratio_macs:.3f})")
    print(f"  pruned params = {pruned_params:.3e}  (ratio = {ratio_params:.3f})")

    # ultralytics 호환 pickle 저장
    out_pt = REPO_ROOT / "trained_weights" / f"_spike_wave10_tp_{int(ratio*100)}.pt"
    ckpt = {
        "model": model.cpu(),
        "train_args": dict(getattr(model, "args", {})) if hasattr(model, "args") else {},
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, str(out_pt))
    size_mb = out_pt.stat().st_size / 1e6
    print(f"  saved to {out_pt.name}  ({size_mb:.2f}MB)")

    return {
        "ratio_target": ratio,
        "status": "PASS",
        "macs_orig": base_macs,
        "macs_pruned": pruned_macs,
        "macs_ratio": ratio_macs,
        "params_orig": base_params,
        "params_pruned": pruned_params,
        "params_ratio": ratio_params,
        "saved_pt": str(out_pt.name),
        "saved_size_mb": size_mb,
        "duration_s": time.time() - t0,
    }


def main() -> int:
    print(f"[p2] torch-pruning {tp.__version__}")
    print(f"[p2] targets: {RATIOS}")
    results = []
    for r in RATIOS:
        results.append(prune_one(r))
        torch.cuda.empty_cache()

    print("\n========== SUMMARY ==========")
    print(f"{'target':<8} {'status':<14} {'macs_ratio':<11} {'params_ratio':<13} {'size(MB)':<10} {'dur(s)':<8}")
    for r in results:
        if r["status"] == "PASS":
            print(f"{r['ratio_target']:<8.2f} {r['status']:<14} {r['macs_ratio']:<11.3f} "
                  f"{r['params_ratio']:<13.3f} {r['saved_size_mb']:<10.2f} {r['duration_s']:<8.1f}")
        else:
            print(f"{r['ratio_target']:<8.2f} {r['status']:<14} {'-':<11} {'-':<13} {'-':<10} "
                  f"{r['duration_s']:<8.1f}  {r.get('error', '')[:80]}")

    out_json = REPO_ROOT / "logs" / "wave10_p2_tp_sweep.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[p2] results saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
