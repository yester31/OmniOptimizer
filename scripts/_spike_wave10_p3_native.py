"""Phase 3 spike — PyTorch native torch.nn.utils.prune.

제한: unstructured weight zero-out. Shape 불변이라 TRT에서 실제 속도 이득 없음
(2:4 Ampere sparse pattern 일치 없으면 일반 dense kernel). **참고용 측정만**.

측정:
  - L1 unstructured amount=0.3 후 actual sparsity
  - 2:4 structured pruning (Ampere) 시도 — Wave 1 "trt_int8_sparsity" 와 비교 관점
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
from thop import profile as thop_profile
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_PT = REPO_ROOT / "best_qr.pt"


def count_sparsity(model: torch.nn.Module) -> tuple[int, int, float]:
    total = 0
    zeros = 0
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() >= 2:  # Conv/Linear weights only
            total += p.numel()
            zeros += (p == 0).sum().item()
    return total, zeros, (zeros / total if total else 0.0)


def apply_l1_unstructured(model: torch.nn.Module, amount: float) -> None:
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)
            prune.remove(m, "weight")  # 영구 제거 (mask 풀고 zero 고정)


def apply_2x4_structured(model: torch.nn.Module) -> int:
    """2:4 structured sparsity — 4개 weight 중 2개 0. Ampere tensor core 패턴."""
    from torch.sparse import to_sparse_semi_structured
    count = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) and m.weight.shape[1] >= 4:
            # L1-magnitude 기반 2:4 mask — 매 4개 채널마다 L1 작은 2개 0
            w = m.weight.data
            shape = w.shape
            w_flat = w.reshape(-1, 4)
            _, idx = w_flat.abs().topk(2, dim=1, largest=False)
            mask = torch.ones_like(w_flat)
            mask.scatter_(1, idx, 0)
            m.weight.data = (w_flat * mask).reshape(shape)
            count += 1
    return count


def main() -> int:
    yolo = YOLO(str(BASE_PT))
    model = yolo.model.cuda().eval()
    dummy = torch.randn(1, 3, 640, 640, device="cuda")

    with torch.no_grad():
        _ = model(dummy)

    base_macs, base_params = thop_profile(model, inputs=(dummy,), verbose=False)
    total_w, zeros_w, sp = count_sparsity(model)
    print(f"[p3] baseline: MACs={base_macs:.3e} params={base_params:.3e} conv_weight_sparsity={sp:.3f}")

    results = {}

    # 1) L1 unstructured 30%
    yolo_a = YOLO(str(BASE_PT))
    model_a = yolo_a.model.cuda().eval()
    with torch.no_grad():
        _ = model_a(dummy)
    apply_l1_unstructured(model_a, amount=0.30)
    total, zeros, sp = count_sparsity(model_a)
    macs_a, params_a = thop_profile(model_a, inputs=(dummy,), verbose=False)
    print(f"[p3] L1 30% unstructured: sparsity={sp:.3f}  MACs={macs_a:.3e} (ratio={macs_a/base_macs:.3f})")
    print(f"       note: shape 불변, TRT dense kernel — 실 속도 이득 없음")
    results["l1_unstructured_30"] = {
        "sparsity": sp, "macs_ratio": macs_a / base_macs,
        "params_unchanged": True, "trt_speedup_expected": False,
    }

    # 2) L1 unstructured 50%
    yolo_b = YOLO(str(BASE_PT))
    model_b = yolo_b.model.cuda().eval()
    with torch.no_grad():
        _ = model_b(dummy)
    apply_l1_unstructured(model_b, amount=0.50)
    total, zeros, sp = count_sparsity(model_b)
    macs_b, params_b = thop_profile(model_b, inputs=(dummy,), verbose=False)
    print(f"[p3] L1 50% unstructured: sparsity={sp:.3f}  MACs={macs_b:.3e} (ratio={macs_b/base_macs:.3f})")
    results["l1_unstructured_50"] = {
        "sparsity": sp, "macs_ratio": macs_b / base_macs,
        "params_unchanged": True, "trt_speedup_expected": False,
    }

    # 3) 2:4 structured (Ampere 호환 패턴)
    yolo_c = YOLO(str(BASE_PT))
    model_c = yolo_c.model.cuda().eval()
    with torch.no_grad():
        _ = model_c(dummy)
    count = apply_2x4_structured(model_c)
    total, zeros, sp = count_sparsity(model_c)
    macs_c, params_c = thop_profile(model_c, inputs=(dummy,), verbose=False)
    print(f"[p3] 2:4 structured ({count} Conv): sparsity={sp:.3f}  MACs={macs_c:.3e} (ratio={macs_c/base_macs:.3f})")
    print(f"       note: Ampere sparse tensor core 활용 시 INT8에서 ~1.5x (TRT 지원)")
    results["2x4_structured"] = {
        "sparsity": sp, "macs_ratio": macs_c / base_macs,
        "layers_patched": count, "trt_speedup_expected": True,
        "reference": "기존 trt_int8_sparsity recipe 이미 동일 패턴",
    }

    # 요약
    print("\n========== SUMMARY ==========")
    print(f"{'method':<25} {'sparsity':<10} {'macs_ratio':<11} {'trt_speedup':<12}")
    for name, r in results.items():
        print(f"{name:<25} {r['sparsity']:<10.3f} {r['macs_ratio']:<11.3f} "
              f"{'YES' if r['trt_speedup_expected'] else 'NO':<12}")

    out_json = REPO_ROOT / "logs" / "wave10_p3_native.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[p3] results saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
