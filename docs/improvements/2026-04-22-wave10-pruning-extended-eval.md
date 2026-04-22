# Wave 10 확장 실험 — pruning 3-framework 비교

**Date**: 2026-04-22
**Status**: 실측 완료. Wave 10 archive 결정 재확인. 차기 방향은 **Wave 9 DirectML EP** 유지.
**Scope**: 사용자 요청에 따라 (a) FastNAS pruned 모델 fine-tune/bench, (b) torch-pruning spike, (c) PyTorch native `prune` spike 3개를 순차 실행.

## TL;DR

| Framework | 실측 결과 | Wave 10 가치 |
|---|---|---|
| **FastNAS** (modelopt) | 15.7% FLOPs ↓, fine-tune 후 **mAP50 0.9475** (−4.05%p drop) | ❌ mAP 제약(1.5%p) 위반 |
| **torch-pruning** (DepGraph) | `ignored_layers=[Detect]` 로 전체 네트워크 coupled protected → **0% pruning** | ⚠️ 재설계 필요 (Detect output Conv만 개별 보호) |
| **PyTorch native prune** (unstructured + 2:4) | weight zero-out only, shape 불변 → **TRT dense kernel**, 속도 이득 0. 2:4 sparsity는 기존 `trt_int8_sparsity` recipe로 이미 측정 완료 (fps 649.2, mAP drop 1.50%p ✘) | ❌ 새 정보 없음 |

**결론**: Wave 10 archive 결정 유지. 후속 방향은 **Wave 9 DirectML EP**.

---

## Phase 1 — FastNAS pruned 모델 fine-tune + bench

### 1A: 60 epoch QR fine-tune (lr0=5e-4, AdamW)

베이스: `_spike_wave10_pruned_ult.pt` (FastNAS constraint=`"95%"`, ratio 0.843)

mAP50 recovery 추세 (주요 epoch):
```
Epoch 1:  0.111    Epoch 15: 0.858
Epoch 2:  0.450    Epoch 22: 0.909
Epoch 3:  0.536    Epoch 30: 0.929
Epoch 5:  0.661    Epoch 35: 0.936
Epoch 8:  0.770    Epoch 40: 0.938
Epoch 10: 0.808    Epoch 45: 0.953  ← peak
Epoch 12: 0.829    Epoch 50: 0.943
Epoch 14: 0.852    Epoch 58: 0.947
                   Epoch 60: 0.948
Best mAP50: 0.9475
```

Original baseline `best_qr.pt` mAP50 ≈ 0.987. Drop = **−3.95%p ~ −4.05%p** (epoch 45 best vs final).

### 1B: TRT FP16 / INT8 engine build + bench

| run | mAP@0.5 | mAP@0.5:0.95 | inference (ms) | fps (inf only)¹ | size MB |
|---|---:|---:|---:|---:|---:|
| pytorch_fp32 (fine-tuned) | 0.9473 | 0.8207 | 26.59 | 37.6 | 5.37 |
| **trt_fp16** | **0.9492** | 0.8241 | 9.79 | 102.1 | 7.61 |
| trt_int8 | 0.9465 | 0.8173 | 9.34 | 107.0 | 5.19 |
| *(참고) baseline `modelopt_int8_entropy`* | 0.987 | — | — | 763.9² | 38 |

¹ ultralytics val 프로토콜 (preprocess 3~4ms + postprocess 0.9ms 포함)
² `scripts/run_trt.py` warmup 100 + measure 100 pure inference — **직접 비교 불가**

### 1C: 해석

- 크기 감축: 38 MB → 5~8 MB (**−79~−86%**)
- 상대 속도: pytorch 26.6ms → trt_int8 9.3ms (**2.8× 가속**)
- **mAP 제약 초과**: baseline 0.987 → 0.949 (trt_fp16 기준 **−3.78%p**)
- **Plan의 max_map_drop_pct=1.5%p 조건 2.5× 초과**. `meets_constraints=False`

### 1D: 왜 fine-tune 후에도 mAP 복구 안 됨?

- QR val set이 133 이미지로 **매우 작음** → overfitting 여지 제한
- 원본 baseline mAP 0.987 자체가 학습/val 분포 정합으로 만들어진 peak
- Pruning이 neck/head 일부만 건드렸지만 (backbone 불변), 그 부분의 capacity loss도 복구 불가

---

## Phase 2 — torch-pruning DepGraph

### 시도

- 패키지: `torch-pruning 1.6.0` (DepGraph + MagnitudePruner, L2 norm, global_pruning=True)
- `pruning_ratio = [0.20, 0.30, 0.50]` sweep
- `ignored_layers = [모든 Detect module]` (YOLO 출력 shape 보호)

### 결과

| ratio_target | status | MACs ratio | params ratio | duration |
|---|---|---:|---:|---:|
| 0.20 | PASS | **1.000** (변화 없음) | 1.000 | 2.8s |
| 0.30 | PASS | **1.000** | 1.000 | 0.6s |
| 0.50 | PASS | **1.000** | 1.000 | 0.6s |

### 원인

torch-pruning DepGraph가 `Detect` 통째를 `ignored_layers`로 취급하면, Detect에 입력되는 **모든 upstream tensor의 채널들이 coupled groups으로 protected** 처리됨. YOLO는 neck/head가 backbone과 dense하게 연결되어 있어 결과적으로 **전체 network가 frozen**.

### 수정 방향 (파킹)

공식 `torch-pruning/examples/yolov8/yolov8_pruning.py` 패턴 모방:

```python
ignored_layers = []
for m in model.modules():
    # Detect head 의 output conv (nc*reg_max, reg_max*4 등 shape 고정) 만 개별 보호
    if isinstance(m, Detect):
        for branch_list in (m.cv2, m.cv3, m.one2one_cv2, m.one2one_cv3):
            for seq in branch_list:
                last = seq[-1]
                if hasattr(last, "conv"):
                    ignored_layers.append(last.conv)
                elif isinstance(last, torch.nn.Conv2d):
                    ignored_layers.append(last)
```

Wave 10 archive 결정으로 **재시도는 하지 않음**. 향후 재평가 시 위 패턴 + `round_to=8` + `unwrapped_parameters`(anchors/strides) 조합 필요.

---

## Phase 3 — PyTorch native `torch.nn.utils.prune`

### 시도

- L1 unstructured 30% / 50% (`prune.l1_unstructured(m, name="weight", amount=...)` + `prune.remove`)
- 2:4 structured sparsity (4개 weight 중 L1 작은 2개 = 0, Ampere sparse tensor core 패턴)

### 결과

| method | sparsity | MACs ratio | TRT speedup 가능? |
|---|---:|---:|:-:|
| L1 30% unstructured | 0.300 | 1.000 (shape 불변) | ❌ NO |
| L1 50% unstructured | 0.500 | 1.000 | ❌ NO |
| 2:4 structured (111 Conv) | 0.497 | 1.000 | ✅ YES (Ampere) |

### 해석

- `torch.nn.utils.prune` 은 mask 기반 weight zero-out. **shape / architecture 불변**
- dense GEMM/Conv kernel이 그대로 돌아 0 가중치도 곱해지므로 **TRT에서 연산량 감소 없음**
- 2:4 structured는 Ampere sparse tensor core 로 이론상 ~1.5× 가속이나, **기존 `trt_int8_sparsity` recipe (report_qr.md rank 21)가 동일 패턴으로 이미 측정됨**: fps 649.2, mAP drop 1.50%p → `meets_constraints=False`

→ **새 정보 제공 없음**. 본 Phase는 "PyTorch native prune이 TRT 실 속도에 기여하지 못한다"는 근거를 문서화하는 역할만 함.

---

## 전체 판정 요약

Wave 10 archive 결정을 **재확인**. 근거 3중:

1. **FastNAS**: infra 작동 확인했으나 15.7% 감축이 structural 상한이고, fine-tune 후 mAP drop 4%p로 제약 위반
2. **torch-pruning**: `ignored_layers` 재설계 필요 (30~60분 추가 R&D 비용). ship 전엔 여전히 fine-tune 비용(수 시간) 필요
3. **PyTorch native prune**: 기존 sparsity recipe 2개 (`modelopt_int8_sparsity`, `trt_int8_sparsity`)와 동일 패턴, 추가 가치 없음

→ **Wave 9 DirectML EP** 로 이동 결정 유지.

## 산출물

- `scripts/_spike_wave10_r1.py`, `_spike_wave10_r1_restore.py`, `_spike_wave10_sweep.py` (기존 FastNAS spike)
- `scripts/_spike_wave10_p1_finetune.py`, `_spike_wave10_p1b_bench.py` (Phase 1/1B)
- `scripts/_spike_wave10_p2_torchpruning.py` (Phase 2)
- `scripts/_spike_wave10_p3_native.py` (Phase 3)
- `logs/wave10_p1_finetune.log`, `wave10_p1b_bench.{log,json}`, `wave10_p2_tp_sweep.{log,json}`, `wave10_p3_native.{log,json}`
- `trained_weights/23_fastnas_p1_finetune/weights/{best,last}.pt` (5.37MB 각)
- `trained_weights/23_fastnas_p1_finetune/weights/best.{onnx,engine}` (FP16), `best_int8.engine`
- `trained_weights/_spike_wave10_pruned_ult.pt`, `_spike_wave10_pruned_ult.onnx` (FastNAS 원본 pruned, fine-tune 전)
