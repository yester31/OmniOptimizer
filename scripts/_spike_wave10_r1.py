"""Wave 10 Task 0 spike — FastNAS on YOLO26n.

공식 modelopt.torch.prune API:
    mtp.prune(model, mode="fastnas", constraints={"flops": "<N>%"},
              dummy_input=..., config={"data_loader": ..., "score_func": ...,
              "collect_func": ..., "checkpoint": ..., "max_iter_data_loader": N})

검증 대상 (4-way decision tree):
  (A) PASS  — prune() 완주 + Detect.forward 호출 카운터 > 0
  (B) anchor mutation AssertionError (Wave 7 재현)
  (C) 기타 crash (dataloader / collect_func / compute_flops 오류)
  (D) Step 1 PASS but restore/export 단계 실패 — _spike_wave10_r1_restore.py 에서 판정

이 스크립트는 (A/B/C)만 다룬다. (D)는 별도 프로세스.

Plan: docs/plans/2026-04-21-wave10-modelopt-fastnas-pruning.md (Task 0 Step 1)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

import modelopt.torch.opt as mto
import modelopt.torch.prune as mtp
from thop import profile as thop_profile
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules.head import Detect


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = REPO_ROOT / "qr_barcode.yaml"
BASE_PT = REPO_ROOT / "best_qr.pt"
CHECKPOINT_PATH = REPO_ROOT / "modelopt_search_checkpoint_spike.pth"
PRUNED_PT = REPO_ROOT / "trained_weights" / "_spike_wave10_pruned.pt"


class SimpleImgLoader:
    """Lightweight tensor loader for FastNAS BN-recalibration pass.

    FastNAS needs only input tensors to re-calibrate BN running stats; the
    full ultralytics batch dict is not required.  Returns `torch.Tensor`
    of shape (B, 3, imgsz, imgsz) directly so `collect_func = lambda x: x`.
    """

    def __init__(self, img_dir: Path, *, imgsz: int = 640, batch: int = 8, n: int = 32):
        paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        self.paths = paths[:n]
        if not self.paths:
            raise RuntimeError(f"No images in {img_dir}")
        self.batch = batch
        self.tfm = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])

    def __iter__(self):
        buf: list[torch.Tensor] = []
        for p in self.paths:
            img = self.tfm(Image.open(p).convert("RGB")).float()
            buf.append(img)
            if len(buf) == self.batch:
                yield torch.stack(buf).cuda()
                buf = []
        if buf:
            yield torch.stack(buf).cuda()

    def __len__(self) -> int:
        return (len(self.paths) + self.batch - 1) // self.batch


def main() -> int:
    print(f"[spike] torch {torch.__version__}  cuda.available={torch.cuda.is_available()}")
    print(f"[spike] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[spike] base checkpoint: {BASE_PT}")
    assert BASE_PT.exists(), f"missing {BASE_PT}"
    PRUNED_PT.parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(BASE_PT))
    model = yolo.model.cuda().eval()
    dummy = torch.randn(1, 3, 640, 640, device="cuda")

    # W4 우회: Detect.shape 세팅용 warmup forward.
    # 이후 subnet sampling 시 H/W 불변 → head.py:178의 shape != self.shape 분기 재트리거 안 됨.
    with torch.no_grad():
        _ = model(dummy)
    print("[spike] warmup forward OK")

    # 원본 FLOPs (thop) — modelopt.torch.utils.count_flops 부재로 thop 사용
    flops_before, params_before = thop_profile(model, inputs=(dummy,), verbose=False)
    print(f"[spike] original FLOPs = {flops_before:.3e}  params = {params_before:.3e}")

    # BN recalib용 dataloader — ultralytics build_dataloader는 Dataset 객체 요구,
    # check_det_dataset["train"]은 문자열 경로라 우회용 custom loader 사용.
    data = check_det_dataset(str(DATA_YAML))
    train_img_dir = Path(data["train"])
    print(f"[spike] train dir: {train_img_dir}")
    loader = SimpleImgLoader(train_img_dir, imgsz=640, batch=8, n=32)
    print(f"[spike] dataloader built, batches = {len(loader)}")

    # collect_func — SimpleImgLoader가 이미 tensor를 yield하므로 identity
    def collect_func(batch: torch.Tensor) -> torch.Tensor:
        return batch

    # Detect.forward 카운터 (B3 가드)
    forward_count = {"n": 0}
    orig_forward = Detect.forward

    def counting_forward(self, *args, **kwargs):
        forward_count["n"] += 1
        return orig_forward(self, *args, **kwargs)

    Detect.forward = counting_forward  # type: ignore[assignment]

    # score_func — yolo.val(fraction=0.01) 빠른 mAP 측정
    def score_func(m: torch.nn.Module) -> float:
        yolo.model = m
        try:
            results = yolo.val(
                data=str(DATA_YAML),
                imgsz=640,
                device="cuda",
                fraction=0.01,
                verbose=False,
                plots=False,
                save=False,
                workers=0,
            )
            return float(results.box.map50)
        except Exception as e:  # noqa: BLE001
            print(f"[spike] score_func error: {type(e).__name__}: {e}")
            return 0.0

    # NOTE: yolo26n에서 FastNAS search space max/min ratio = 1.11 (max=5.74B, min=5.19B).
    # 즉 얻을 수 있는 최대 FLOPs 감축은 ~10%뿐. plan 목표(30~50%)는 구조적 불가.
    # "95%" 로 재시도해 인프라(search→save→restore→ONNX)만 검증.
    print("[spike] calling mtp.prune(mode='fastnas', constraints={'flops': '95%'})...")
    t0 = time.time()
    try:
        pruned, search_meta = mtp.prune(
            model=model,
            mode="fastnas",
            constraints={"flops": "95%"},
            dummy_input=dummy,
            config={
                "data_loader": loader,
                "score_func": score_func,
                "collect_func": collect_func,
                "checkpoint": str(CHECKPOINT_PATH),
                "max_iter_data_loader": 2,
            },
        )
    except AssertionError as e:
        print(f"[spike] SCENARIO B (AssertionError — anchor mutation 의심):")
        print(f"        {e}")
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[spike] SCENARIO C (other crash):")
        print(f"        {type(e).__name__}: {e}")
        raise
    finally:
        Detect.forward = orig_forward  # type: ignore[assignment]

    elapsed = time.time() - t0
    print(f"[spike] prune() OK in {elapsed:.1f}s")
    print(f"[spike] search_meta keys: {list(search_meta.keys()) if search_meta else 'None'}")

    # 성공 게이트 — Detect.forward 가 실제로 호출됐는가
    assert forward_count["n"] > 0, (
        "Detect.forward 호출 0회 — FastNAS가 forward를 돌리지 않음. "
        "score_func/collect_func 구성 오류 의심."
    )
    print(f"[spike] Detect.forward calls = {forward_count['n']}")

    # Pruned FLOPs
    pruned_cuda = pruned.cuda().eval()
    with torch.no_grad():
        _ = pruned_cuda(dummy)  # shape warmup
    flops_after, params_after = thop_profile(pruned_cuda, inputs=(dummy,), verbose=False)
    ratio_flops = flops_after / flops_before
    ratio_params = params_after / params_before
    print(f"[spike] pruned FLOPs  = {flops_after:.3e}  (ratio = {ratio_flops:.3f})")
    print(f"[spike] pruned params = {params_after:.3e}  (ratio = {ratio_params:.3f})")

    # save/restore 체인 검증용 — R1.5 가 이걸 restore
    mto.save(pruned_cuda, str(PRUNED_PT))
    print(f"[spike] saved pruned (modelopt) to {PRUNED_PT}  size={PRUNED_PT.stat().st_size/1e6:.2f}MB")

    # ultralytics 호환 full-model pickle — restore_from_modelopt_state 의 strict
    # 제약을 우회. YOLO(path) 가 load_checkpoint 경로로 읽음.
    pruned_pt_ult = REPO_ROOT / "trained_weights" / "_spike_wave10_pruned_ult.pt"
    ckpt_ult = {
        "model": pruned_cuda.cpu(),
        "train_args": dict(getattr(yolo.model, "args", {})) if hasattr(yolo.model, "args") else {},
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt_ult, str(pruned_pt_ult))
    print(f"[spike] saved pruned (ultralytics) to {pruned_pt_ult}  size={pruned_pt_ult.stat().st_size/1e6:.2f}MB")

    print("[spike] SCENARIO A partial PASS (R1 단독). R1.5 restore 검증 필요.")
    print("[spike] next: python scripts/_spike_wave10_r1_restore.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
