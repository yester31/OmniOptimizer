# Wave 10 R1 spike 결과 — FastNAS × YOLO26n PASS (패치 필요)

**Date**: 2026-04-22
**Status**: Scenario A FULL PASS (인프라 작동) + constraint sweep으로 **plan의 30~50% pruning 목표 구조적 불가능** 확정.
**Decision (2026-04-22)**: **옵션 B (Wave 10 archive → Wave 9 DirectML EP)** 선택 확정. Plan 파일에 ARCHIVED 스탬프 적용, CLAUDE.md 에 근거 반영.

## TL;DR

`scripts/_spike_wave10_r1.py` + `_spike_wave10_r1_restore.py` 두 스크립트로 자동 검증:

1. ✅ FastNAS `mtp.prune(mode="fastnas", constraints={"flops": "95%"})` **58.8s → 32.3s** (재실행) 완주
2. ✅ Detect.forward **180회/30회 호출** — Wave 7 anchor mutation 블로커 **재현 없음**
3. ✅ Pruned FLOPs **2.43B (16% 감축)** / params **2.06M (18% 감축)**
4. ❌ `mto.restore` / `restore_from_modelopt_state` — `select(..., strict=True)` 경로 **우회 API 부재**
5. ✅ **ultralytics full-model pickle 우회**: `torch.save({"model": pruned_cuda.cpu(), "train_args": ...})` → `YOLO(path)` 로 별도 프로세스 복원
6. ✅ `yolo.export(format="onnx", dynamic=True)` — **deepcopy 실패 없음** (Brevitas 전례 미재현), 8.3MB ONNX 생성

4-way decision tree: **Scenario A** 확정.

## 실측 수치

### R1 — FastNAS search

| 지표 | 값 |
|---|---|
| Search duration | 32.3s ~ 58.8s (2회 실행) |
| Detect.forward 호출 | 30 ~ 180회 |
| Original FLOPs (thop) | 2.887e+09 MACs (= ~5.77B FLOPs modelopt 정의) |
| Pruned FLOPs (thop) | 2.433e+09 MACs (= ~4.87B FLOPs) |
| FLOPs 감축률 | **15.7%** |
| Original params | 2.505M |
| Pruned params | 2.058M |
| Params 감축률 | **17.8%** |
| Best subnet constraint | `{'params': '2.06M', 'flops': '4.99B'}` |
| Model size | 38.2MB (baseline, fp32) → **8.48MB (modelopt)** / **8.60MB (ultralytics)** |
| Total search space | 2.52e+07 subnets |

### Search space 분석

307개 hparam entry 중 대부분 single-choice (prunable X). 실제 prunable 계층:

| 영역 | choices | 비고 |
|---|---|---|
| `model.8.m.0.*` | [32, 64] | 50% or 100% |
| `model.10.m.0.ffn.*` | [32, 64, 96, 128, 160, 192, 224, 256] | 8-way |
| `model.22.m.0.*` | [32, 64] / [32, ..., 256] | ffn & conv |
| `model.23.cv2/cv3/one2one_cv*` | 혼합 | Detect head 일부 |

Backbone 대부분 (`model.0~6`, `model.12/15/18/21`) 은 **FX trace 실패**로 search space에서 제외됨:
- `TraceError: Proxy object cannot be iterated` — `model.2/4/6/8/13/16/19/22/23`
- `NotImplementedError: ModuleList missing forward` — `model.2.m/4.m/6.m/...`
- `TypeError: cat() received Proxy, int` — `model.12/15/18/21`

**실질 prunable 구조는 neck/head + 일부 middle block**. Backbone에 손대지 못하므로 감축 상한이 제한적.

### Profiling Results (constraint="90%" 실패 케이스)

FastNAS 내부 3-subnet profiling:

| Constraint | min | centroid | max | max/min |
|---|---|---|---|---|
| flops | 5.19B | 5.26B | 5.74B | 1.11 |
| params | 2.00M | 2.10M | 2.49M | 1.24 |

`"90%"` upper bound = 5.17B < min(5.19B) → Satisfiable=False로 `ValueError: NOT all constraints can be satisfied`.
`"95%"` upper bound = 5.45B > min(5.19B) → OK, 84.3%로 내려감.

## Constraint sweep — 2026-04-22 `scripts/_spike_wave10_sweep.py`

**실측 결과** (`logs/wave10_sweep_results.json`):

| constraint | status | flops ratio | params ratio | duration |
|---|---|---|---|---|
| `"95%"` | **PASS** | 0.843 | 0.822 | 68.6s |
| `"85%"` | UNSAT | — | — | 21.6s |
| `"80%"` | UNSAT | — | — | 20.9s |
| `"70%"` | UNSAT | — | — | 20.5s |
| `"50%"` | UNSAT | — | — | 18.7s |

**결론**: FastNAS × YOLO26n 조합에서 **`"95%"` 만 feasible**. 85% 이상 타이트한 constraint는 모두 `ValueError: NOT all constraints can be satisfied` — search space 내 subnet이 upper bound를 만족 못함.

즉 plan의 recipe 목표 재계산:

| 원 plan | 실제 가능 여부 |
|---|---|
| `#23 prune_flops_ratio=0.70` (30% 감축) | ❌ 불가능 |
| `#24 prune_flops_ratio=0.50` (50% 감축) | ❌ 불가능 |
| `#25 prune_flops_ratio=0.70 + INT8` | ❌ 불가능 |
| 실현 가능 최대 | **15.7% 감축** (ratio 0.843, constraint `"95%"`) |

## 판정 (사용자 결정 대기)

### 옵션 A — 축소 ship

- Recipe #23만 유지: `prune_flops_ratio=0.85` (실측 0.843 반영, ±2%p 마진)
- #24 archive (50% target 구조적 불가)
- #25 pruning+INT8 chain 유지 — pruned base (15% 감축) + INT8 entropy 체이닝. `modelopt_int8_entropy` fps 763.9 대비 **+10~15%** 기대 (추정 820~880 fps)
- Task 1~7 실행 비용: 8~12h (recipe 2개로 축소)
- 실익: medium (INT8 chain 에서 차별화 가능)

### 옵션 B — 완전 archive

- Wave 10 전체 archive
- Wave 9 DirectML EP (ORT-native, AMD/Intel GPU/NPU 커버 확장) 로 이동
- 이유: 15% 감축만으로는 recipe bank에 추가할 차별성 낮음, 학습 비용(fine-tune 60 epoch) 대비 얻는 정보 제한적
- 실익: high (Wave 9 가 새 HW 경로 확보)

### 권장

**옵션 B (archive)** — 근거:
1. 15% pruning 은 GPU rank 변동에 marginal
2. `modelopt_int8_entropy` 이미 pruning 없이 #1 (fps 763.9). Fine-tune 드리프트로 QAT base 약화 위험
3. Wave 7/8/12/13 모두 archive된 시점에서 Wave 9 (DirectML) 이 다양성 확보에 더 가치 있음
4. FastNAS FX trace 제약(`model.0~6` backbone 제외)은 **ultralytics 모델 구조 수정** 없이 해결 불가 — R&D 비용 과대

사용자가 옵션 A 선택 시 Task 1~7 실행 가능.

즉 **min-3-sample 관찰이 실제 search space의 상한과 일치**. Plan 30% target은 **추가 검증으로 확정 실패**.

## R1.5 — restore + ONNX export

### Path A (공식 modelopt 경로) — FAIL

```python
mto.restore(yolo_base.model, "_spike_wave10_pruned.pt")
# or
mto.restore_from_modelopt_state(yolo_base.model, modelopt_state=objs["modelopt_state"])
```

둘 다 내부 `modelopt.torch.opt.dynamic.select(config, strict=True)` 에서:

```
RuntimeError: Missing keys in config for:
    model.23.cv3.0.0.1.conv.out_channels
    model.23.cv3.0.1.1.conv.out_channels
    ... (50+ keys)
Make sure all keys are present in config or set strict=False.
```

**원인**: FastNAS가 `strict=True` 로 modelopt_state를 복원할 때, `YOLO("best_qr.pt")`로 만든 **원본 skeleton이 modelopt dynamic hook 으로 wrap되지 않은 상태**에서 저장된 config의 일부 keys를 매치할 수 없음. `restore` / `restore_from_modelopt_state` 둘 다 **strict 옵션 공개 파라미터 없음** (kwargs는 torch.load 전용). 공식 우회 경로 부재.

### Path B (ultralytics full-model pickle) — PASS

```python
# R1 저장:
torch.save({
    "model": pruned_cuda.cpu(),           # 전체 nn.Module pickle
    "train_args": dict(yolo.model.args),  # ultralytics 호환
    "date": "...",
}, "_spike_wave10_pruned_ult.pt")

# R1.5 복원 (별도 프로세스):
yolo = YOLO("_spike_wave10_pruned_ult.pt")
# → ultralytics load_checkpoint 경로가 자동 호출
# → yolo.model == pruned DetectionModel (architecture 보존)
yolo.export(format="onnx", imgsz=640, dynamic=True)  # 8.3MB 생성
```

**장점**: modelopt 공식 API 제약 우회, ultralytics fine-tune 경로(`trainer.resume`)와 동일 포맷.
**트레이드오프**: `modelopt_state` (search history, sensitivity_map 등) 휘발. 학습 재개 후 재-pruning 필요 시 재실행. Wave 10 워크플로우는 prune → fine-tune → export 단방향이라 무관.

## 영향 (Plan 수정 사항)

### Wave 10 plan 수정 (docs/plans/2026-04-21-wave10-modelopt-fastnas-pruning.md)

1. **Task 2 (`_modifiers/modelopt_prune.py`) `finalize()` 변경**:
   ```python
   # Before (plan):
   mto.save(yolo.model, str(out_pt))
   # After:
   torch.save({
       "model": yolo.model.cpu(),
       "train_args": dict(getattr(yolo.model, "args", {})),
       "date": time.strftime("%Y-%m-%d %H:%M:%S"),
   }, str(out_pt))
   ```

2. **Task 3 (`_weights_io.py::_resolve_weights`) 변경**:
   - `modelopt_prune` 분기에서 `mto.restore` 호출 **삭제**
   - 대신 기본 ultralytics `YOLO(pruned_pt)` 경로 사용 (추가 로직 불필요)
   - `qat-nvidia` fork 도입 **불필요**

3. **Task 4 (recipe YAML) FLOPs ratio 재조정 후보**:
   - `#23 70pct_fp16` → **추가 spike 필요** — `constraints={"flops": "70%"}`가 search space 수용 확인 후 결정
   - `#24 50pct_fp16` → 50%는 search space 상 **거의 불가능** (min subnet ≈ 90% 기준). 번호 재할당 또는 `"85%"` 로 보수적 목표
   - `#25 70pct_int8` → #23 의존 (동일)

4. **Task 0 spike의 4-way decision tree 결과**: **Scenario A**. Task 1~7 진행 권고.

### 추가 조사 권장 (Task 0 확장)

`_spike_wave10_r1.py` 에 constraint 스윕 루프 추가 시도:

```python
for target in ["70%", "50%", "80%", "85%"]:
    pruned, _ = mtp.prune(..., constraints={"flops": target}, ...)
    # each constraint 이 satisfiable한지 실측
```

이게 plan의 실제 recipe 타깃을 결정하는 근거 수치를 제공. 예상:
- `"80%"` / `"85%"` : PASS (search space 내 충분히 공격적 subnet 존재 가능)
- `"70%"` / `"50%"` : **FAIL 가능성 높음** — backbone prune 안 되는 구조적 제약

## 교훈 (CLAUDE.md 후보)

- `mto.save` / `mto.restore` 는 **architecture 불변 모델 (QAT/sparsity)** 에 최적화. Pruning처럼 architecture 가 변하는 경우 **ultralytics full-model pickle이 더 robust**.
- FastNAS search space 는 `torch.fx` trace 가능성에 심하게 의존. YOLO-family 처럼 dynamic control flow + ModuleList 를 많이 쓰는 모델은 **backbone 자체가 search space에서 제외**되어 상한이 제한됨.
- `min/centroid/max` 3-sample profiling 결과의 `max/min ratio` 는 **탐색 상한의 lower bound** 로만 해석 — 실제 search space엔 더 공격적인 subnet이 존재 가능. 최종 판단은 실 spike의 pruned FLOPs로.
- Plan 에서 공식 API를 그대로 옮길 때 **API 제약 (strict=True hard-coded)** 은 spike 전엔 발견 어려움. `/gsd-plan-phase` cross-verify 에서 이런 세부 제약까지 잡긴 어려우니 **spike-first 원칙**이 중요.

## 참고 링크

- NVIDIA ModelOpt — [pytorch prune guide](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_model_optimizer.html#pruning)
- modelopt 0.43.0 conversion.py — [restore_from_modelopt_state 소스](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/opt/conversion.py)
- ultralytics `load_checkpoint` 로더 — `ultralytics/nn/tasks.py`
- Wave 7/8 archive (anchor mutation / external converter) — `docs/improvements/2026-04-21-wave7-r3-r5-spike-results.md`, `-wave8-r1-spike-results.md`

## 산출물

- `scripts/_spike_wave10_r1.py` — FastNAS search + ultralytics pickle 저장
- `scripts/_spike_wave10_r1_restore.py` — ultralytics pickle 복원 + ONNX export 검증
- `trained_weights/_spike_wave10_pruned.pt` (modelopt 8.48MB) — 참고용
- `trained_weights/_spike_wave10_pruned_ult.pt` (ultralytics 8.60MB) — restore 가능
- `trained_weights/_spike_wave10_pruned_ult.onnx` (8.3MB) — export 산출물
- `logs/wave10_r1_spike.log`, `logs/wave10_r1_restore.log` — 실행 기록
