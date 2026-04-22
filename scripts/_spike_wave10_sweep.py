"""Wave 10 Task 0 spike 확장 — constraint sweep.

R1 결과 ("95%" constraint → 84.3% pruned)를 일반화. 각 target constraint 마다
- FastNAS prune() 가 satisfiable 한지
- 실제 달성한 FLOPs / params ratio
- search time
을 측정해, recipe #23/#24/#25 의 FLOPs 타깃을 **실측 기반**으로 확정.

각 iteration 마다 fresh YOLO 로드 — 이전 prune()이 모델 architecture를 mutate하므로.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

import modelopt.torch.prune as mtp
from thop import profile as thop_profile
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules.head import Detect


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = REPO_ROOT / "qr_barcode.yaml"
BASE_PT = REPO_ROOT / "best_qr.pt"


CONSTRAINTS_TO_TRY = ["95%", "85%", "80%", "70%", "50%"]


class SimpleImgLoader:
    def __init__(self, img_dir: Path, *, imgsz: int = 640, batch: int = 8, n: int = 32):
        paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        self.paths = paths[:n]
        self.batch = batch
        self.tfm = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])

    def __iter__(self):
        buf: list[torch.Tensor] = []
        for p in self.paths:
            buf.append(self.tfm(Image.open(p).convert("RGB")).float())
            if len(buf) == self.batch:
                yield torch.stack(buf).cuda()
                buf = []
        if buf:
            yield torch.stack(buf).cuda()

    def __len__(self) -> int:
        return (len(self.paths) + self.batch - 1) // self.batch


def run_one(constraint: str, *, loader: SimpleImgLoader, data_yaml: str) -> dict:
    print(f"\n========== constraint = {constraint} ==========")
    t0 = time.time()
    yolo = YOLO(str(BASE_PT))
    model = yolo.model.cuda().eval()
    dummy = torch.randn(1, 3, 640, 640, device="cuda")
    with torch.no_grad():
        _ = model(dummy)

    flops_before, params_before = thop_profile(model, inputs=(dummy,), verbose=False)

    def collect_func(batch: torch.Tensor) -> torch.Tensor:
        return batch

    def score_func(m: torch.nn.Module) -> float:
        yolo.model = m
        try:
            r = yolo.val(
                data=data_yaml, imgsz=640, device="cuda",
                fraction=0.01, verbose=False, plots=False, save=False, workers=0,
            )
            return float(r.box.map50)
        except Exception:  # noqa: BLE001
            return 0.0

    result: dict = {"constraint": constraint, "status": "UNKNOWN"}
    try:
        pruned, _ = mtp.prune(
            model=model,
            mode="fastnas",
            constraints={"flops": constraint},
            dummy_input=dummy,
            config={
                "data_loader": loader,
                "score_func": score_func,
                "collect_func": collect_func,
                "checkpoint": str(REPO_ROOT / f"modelopt_search_checkpoint_{constraint.rstrip('%')}.pth"),
                "max_iter_data_loader": 2,
            },
        )
        with torch.no_grad():
            _ = pruned.cuda().eval()(dummy)
        flops_after, params_after = thop_profile(pruned.cuda().eval(), inputs=(dummy,), verbose=False)
        ratio_f = flops_after / flops_before
        ratio_p = params_after / params_before
        result.update({
            "status": "PASS",
            "flops_orig_macs": flops_before,
            "flops_pruned_macs": flops_after,
            "flops_ratio": ratio_f,
            "params_orig": params_before,
            "params_pruned": params_after,
            "params_ratio": ratio_p,
            "duration_s": time.time() - t0,
        })
        print(f"  PASS  flops ratio={ratio_f:.3f}  params ratio={ratio_p:.3f}  dur={result['duration_s']:.1f}s")
    except ValueError as e:
        result.update({
            "status": "UNSAT",
            "error": str(e).splitlines()[0][:200],
            "duration_s": time.time() - t0,
        })
        print(f"  UNSAT  {result['error']}")
    except Exception as e:  # noqa: BLE001
        result.update({
            "status": "CRASH",
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "duration_s": time.time() - t0,
        })
        print(f"  CRASH  {result['error']}")
    finally:
        del yolo
        torch.cuda.empty_cache()

    return result


def main() -> int:
    print(f"[sweep] torch {torch.__version__}  GPU={torch.cuda.get_device_name(0)}")
    print(f"[sweep] targets: {CONSTRAINTS_TO_TRY}")

    # Detect.forward 카운터는 sweep 동안 단일 원본 유지 — 반복 측정 불필요
    data = check_det_dataset(str(DATA_YAML))
    loader = SimpleImgLoader(Path(data["train"]), imgsz=640, batch=8, n=32)

    results = []
    for c in CONSTRAINTS_TO_TRY:
        results.append(run_one(c, loader=loader, data_yaml=str(DATA_YAML)))

    print("\n========== SUMMARY ==========")
    print(f"{'constraint':<12} {'status':<8} {'flops_ratio':<12} {'params_ratio':<13} {'dur(s)':<8}")
    for r in results:
        if r["status"] == "PASS":
            print(f"{r['constraint']:<12} {r['status']:<8} {r['flops_ratio']:<12.3f} "
                  f"{r['params_ratio']:<13.3f} {r['duration_s']:<8.1f}")
        else:
            print(f"{r['constraint']:<12} {r['status']:<8} {'-':<12} {'-':<13} {r.get('duration_s', 0):<8.1f}  "
                  f"{r.get('error', '')}")

    # JSON 덤프
    import json
    out_json = REPO_ROOT / "logs" / "wave10_sweep_results.json"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[sweep] results saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
