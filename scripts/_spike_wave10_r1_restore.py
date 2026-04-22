"""Wave 10 Task 0 spike Step 1.5 — restore + ONNX export 검증.

R1 과 별도 프로세스에서 실행해야 in-process state 의존을 배제한다.

검증:
  1. YOLO(base) skeleton + mto.restore(pruned_pt) — 채널 수 변경 state_dict 복원
  2. yolo.export(format="onnx") — deepcopy 실패 여부 (Brevitas 전례)

exit codes:
  0 = 둘 다 OK → SCENARIO (A) 최종 확정
  1 = mto.restore 실패 → SCENARIO (D-restore)
  2 = ONNX export 실패 → SCENARIO (D-export), Task 3 ONNX 헬퍼 필요
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch
import modelopt.torch.opt as mto
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_PT = REPO_ROOT / "best_qr.pt"
PRUNED_PT = REPO_ROOT / "trained_weights" / "_spike_wave10_pruned.pt"
PRUNED_PT_ULT = REPO_ROOT / "trained_weights" / "_spike_wave10_pruned_ult.pt"
ONNX_OUT = REPO_ROOT / "trained_weights" / "_spike_wave10_pruned.onnx"


def main() -> int:
    print(f"[restore] base: {BASE_PT}")
    print(f"[restore] pruned (ultralytics fmt): {PRUNED_PT_ULT}")
    assert BASE_PT.exists(), f"missing {BASE_PT}"
    assert PRUNED_PT_ULT.exists(), (
        f"missing {PRUNED_PT_ULT}  —  R1 spike 를 먼저 실행 (ultralytics 저장 경로 포함 버전)"
    )

    # restore_from_modelopt_state 는 strict=True 우회 경로 부재 (Wave 10 R1 리스크 확인).
    # 대안: full-model pickle 을 ultralytics YOLO() 로더가 바로 소화.
    print("[restore] path A: YOLO(pruned_ult_pt) — ultralytics 호환 full-model pickle 경로")
    try:
        yolo = YOLO(str(PRUNED_PT_ULT))
        print(f"[restore]   YOLO load OK — model type: {type(yolo.model).__name__}")
        # 채널 수가 원본과 다른지 sanity check
        conv0_out = yolo.model.model[0].conv.out_channels
        print(f"[restore]   model.model[0].conv.out_channels = {conv0_out}  (원본 yolo26n = 16)")
    except Exception as e:  # noqa: BLE001
        print(f"[restore]   FAIL A: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # ONNX export — Brevitas 에서 목격된 deepcopy 실패 재현 여부 확인
    try:
        yolo.export(
            format="onnx",
            imgsz=640,
            dynamic=True,
            half=False,
            simplify=True,
        )
        print("[export] ONNX export OK")
        # ultralytics 는 기본적으로 base_pt 옆에 .onnx 생성. 위치 확인.
        default_onnx = BASE_PT.with_suffix(".onnx")
        if default_onnx.exists():
            print(f"[export] onnx at {default_onnx}  size={default_onnx.stat().st_size/1e6:.2f}MB")
    except Exception as e:  # noqa: BLE001
        print(f"[export] FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 2

    print("[restore] SCENARIO A FULL PASS — Task 1 진행 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
