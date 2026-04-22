"""Phase 4 — FastNAS pruned 모델에 modelopt INT8 / sparsity / chain 3-variant 적용.

입력:  trained_weights/23_fastnas_p1_finetune/weights/best.pt (fine-tuned FastNAS)
출력:  각 variant별 ultralytics pickle + ONNX + TRT engine + bench JSON

variants:
  (A) fastnas_int8        — modelopt INT8 entropy PTQ (forward_loop calibration)
  (B) fastnas_sparsity    — modelopt 2:4 sparse_magnitude
  (C) fastnas_sp_int8     — sparsity 먼저 + INT8 PTQ (chain)

비교 대상: Phase 1B 기준 fine-tuned pytorch=0.9473, trt_fp16=0.9492, trt_int8=0.9465.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

import modelopt.torch.quantization as mtq
import modelopt.torch.sparsity as ms
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent.parent
FASTNAS_BEST = REPO_ROOT / "trained_weights" / "23_fastnas_p1_finetune" / "weights" / "best.pt"
DATA_YAML = REPO_ROOT / "qr_barcode.yaml"
OUT_DIR = REPO_ROOT / "trained_weights" / "23_fastnas_chain"


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


def apply_int8_ptq(yolo: YOLO, loader: SimpleImgLoader) -> None:
    """INT8 PTQ with entropy calibration via forward_loop."""
    def forward_loop(m: torch.nn.Module) -> None:
        m.eval()
        with torch.no_grad():
            for batch in loader:
                _ = m(batch)

    cfg = mtq.INT8_DEFAULT_CFG
    mtq.quantize(yolo.model, cfg, forward_loop=forward_loop)
    wrapped = sum(1 for _, m in yolo.model.named_modules()
                  if any(s in type(m).__name__.lower() for s in ("quant",)))
    print(f"  [int8] {wrapped} modules quantized")
    if wrapped == 0:
        raise RuntimeError("INT8: 0 modules quantized")


def apply_sparsity(yolo: YOLO) -> None:
    """2:4 sparse_magnitude."""
    ms.sparsify(yolo.model, mode="sparse_magnitude")
    sparsified = 0
    for _, m in yolo.model.named_modules():
        mask = getattr(m, "_weight_mask", None)
        if mask is not None and not bool(mask.all().item()):
            sparsified += 1
    print(f"  [sparsity] {sparsified} layers sparsified (2:4)")
    if sparsified == 0:
        print("  WARN: 0 layers sparsified — in_channels % 16 != 0 constraints")


def save_ultralytics_pickle(yolo: YOLO, out_pt: Path) -> None:
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": yolo.model.cpu(),
        "train_args": dict(getattr(yolo.model, "args", {})) if hasattr(yolo.model, "args") else {},
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, str(out_pt))
    # 원래 cuda 로 되돌리기
    yolo.model.cuda()


def bench_trt_engine(yolo: YOLO, label: str, stem: str, *,
                     use_int8: bool = False, use_sparsity: bool = False) -> dict:
    """in-process YOLO 객체 직접 engine export — modelopt-wrapped 모듈은 pickle 불가라 pickle 단계 생략."""
    print(f"\n=== bench {label} ===  (in-memory model)")

    # modelopt QDQ 가 삽입된 모델은 half=True + int8=False 로 engine build.
    # TRT 가 ONNX 의 Q/DQ 노드를 보고 자동 INT8 mode 로 해당 layer 실행.
    # sparsity 는 TRT가 weight tensor 의 2:4 pattern 을 자동 감지 (default enabled in 10.x).
    try:
        engine_path = yolo.export(
            format="engine",
            imgsz=640,
            half=True,
            int8=False,            # modelopt QDQ 있으면 TRT 가 자동 INT8
            dynamic=False,
            batch=1,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  engine build CRASH: {type(e).__name__}: {e}")
        return {"label": label, "status": "CRASH_ENGINE", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    engine_pt = Path(engine_path)
    engine_renamed = engine_pt.with_name(f"{stem}.engine")
    if engine_renamed.exists():
        engine_renamed.unlink()
    shutil.move(str(engine_pt), str(engine_renamed))
    print(f"  engine: {engine_renamed.name}  ({engine_renamed.stat().st_size/1e6:.2f}MB)")

    # val + speed
    yolo = YOLO(str(engine_renamed))
    try:
        metrics = yolo.val(
            data=str(DATA_YAML),
            imgsz=640,
            batch=1,
            device=0,
            half=not use_int8,
            int8=use_int8,
            verbose=False,
            plots=False,
            save=False,
            workers=0,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  val CRASH: {type(e).__name__}: {e}")
        return {"label": label, "status": "CRASH_VAL", "error": f"{type(e).__name__}: {str(e)[:200]}",
                "engine_size_mb": engine_renamed.stat().st_size / 1e6}

    speed = metrics.speed
    inf_ms = speed.get("inference", 0.0)
    fps = (1000.0 / inf_ms) if inf_ms > 0 else 0.0
    total_ms = sum([speed.get(k, 0) for k in ("preprocess", "inference", "postprocess")])
    fps_e2e = (1000.0 / total_ms) if total_ms > 0 else 0.0
    print(f"  mAP50={float(metrics.box.map50):.4f}  inf={inf_ms:.2f}ms  fps={fps:.1f}  "
          f"e2e={total_ms:.2f}ms/{fps_e2e:.1f}fps")
    return {
        "label": label,
        "status": "OK",
        "engine_file": engine_renamed.name,
        "engine_size_mb": engine_renamed.stat().st_size / 1e6,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "inference_ms": inf_ms,
        "e2e_ms": total_ms,
        "fps_inference": fps,
        "fps_e2e": fps_e2e,
    }


def variant(name: str, *, do_sparsity: bool, do_int8: bool) -> dict:
    print(f"\n========== variant {name} ==========")
    t0 = time.time()
    yolo = YOLO(str(FASTNAS_BEST))
    yolo.model.cuda().eval()

    # calibration loader
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(str(DATA_YAML))
    loader = SimpleImgLoader(Path(data["train"]), imgsz=640, batch=8, n=32)

    # warmup forward
    dummy = torch.randn(1, 3, 640, 640, device="cuda")
    with torch.no_grad():
        _ = yolo.model(dummy)

    if do_sparsity:
        apply_sparsity(yolo)
    if do_int8:
        apply_int8_ptq(yolo, loader)

    # modelopt wrapped 모듈은 pickle 불가 — pickle 단계 생략하고 바로 engine export.
    engine_stem = f"{OUT_DIR.name}/{name}"
    # 실제 저장 경로: trained_weights/23_fastnas_chain/{name}.engine
    bench = bench_trt_engine(yolo, name, stem=str(OUT_DIR / name), use_int8=do_int8, use_sparsity=do_sparsity)
    bench["apply_duration_s"] = time.time() - t0
    return bench


def main() -> int:
    assert FASTNAS_BEST.exists(), f"missing {FASTNAS_BEST}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    results["A_fastnas_int8"] = variant("fastnas_int8", do_sparsity=False, do_int8=True)
    torch.cuda.empty_cache()
    results["B_fastnas_sparsity"] = variant("fastnas_sparsity", do_sparsity=True, do_int8=False)
    torch.cuda.empty_cache()
    results["C_fastnas_sp_int8"] = variant("fastnas_sp_int8", do_sparsity=True, do_int8=True)

    print("\n========== PHASE 4 SUMMARY ==========")
    print(f"{'variant':<25} {'status':<12} {'mAP50':<8} {'inf ms':<9} {'fps (inf)':<11} {'size MB':<8}")
    for k, r in results.items():
        if r.get("status") == "OK":
            print(f"{k:<25} {r['status']:<12} {r['map50']:<8.4f} {r['inference_ms']:<9.2f} "
                  f"{r['fps_inference']:<11.1f} {r['engine_size_mb']:<8.2f}")
        else:
            print(f"{k:<25} {r.get('status', '?'):<12} CRASH — {r.get('error', '')[:60]}")

    print("\nREFERENCE — Phase 1B fine-tuned:")
    print("  pytorch_fp32  mAP50=0.9473  inf=26.59ms  fps=37.6")
    print("  trt_fp16      mAP50=0.9492  inf=9.79ms   fps=102.1")
    print("  trt_int8      mAP50=0.9465  inf=9.34ms   fps=107.0  (ultralytics int8 export, TRT native calibrator)")

    out_json = REPO_ROOT / "logs" / "wave10_p4_chain.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[p4] saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
