# QR Training Pipeline — QAT & Sparsity 레시피 활성화

**날짜**: 2026-04-20
**상태**: Design (user review pending)
**대상 브랜치**: main
**목표**: 파킹된 레시피 #07, #11 을 학습 파이프라인과 함께 활성화하고, 신규 #17 modelopt_int8_qat 추가. QR/Barcode fine-tuned 체크포인트 기반.

---

## 1. Goal & Non-Goals

### Goal
- 3개 레시피 일괄 구축: `#07 trt_int8_sparsity` (prune_24), `#11 modelopt_int8_sparsity` (modelopt sparsify), `#17 modelopt_int8_qat` (신규 QAT)
- 공통 학습 파이프라인 (`scripts/train.py` + `_train_core` + `_modifiers/*`)
- `best_qr.pt` 기반 recovery fine-tuning (QAT 30ep, sparsity 60ep)
- 결과: `trained_weights/{recipe}.pt` 생성 → 기존 `run_trt.py`가 자동 로드하여 평가

### Non-Goals
- ultralytics trainer fallback 실제 구현 (스캐폴드만, 호환성 문제 발생 시 구현)
- Multi-GPU 학습
- QAT 초기 scale calibration (forward_loop) — MVP 이후 고려
- Brevitas QAT 레시피 — 별도 작업
- COCO baseline 재학습

---

## 2. Architecture

OmniOptimizer의 recipe-driven 원칙 확장: 학습은 별도 entry point, 레시피 YAML이 single source of truth.

```
              recipe YAML (technique.training 섹션 추가)
              └── modifier: prune_24 | modelopt_sparsify | modelopt_qat
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
   scripts/train.py                 scripts/run_trt.py (최소 변경)
        │                               │
        │ produces                      │ auto-loads
        ▼                               ▼
   trained_weights/{recipe}.pt     TRT engine + results_qr/{recipe}.json
```

**원칙**:
- 학습 스크립트가 1 recipe → 1 `.pt`. modifier별 전후 훅.
- runner 수정 최소화 — 10 LOC 이내 trained_weights 조회 로직만 추가.
- 학습/평가 분리: `run_qr_train_batch.sh` → `run_qr_batch.sh`.
- modelopt-wrapped 모델은 `modelopt.torch.opt.save/restore` 로 직렬화.

---

## 3. File Changes

### 신규 파일

| 파일 | LOC (예상) | 설명 |
|---|---|---|
| `scripts/train.py` | ~150 | CLI entry, recipe 로드, modifier dispatch, skip/force 로직 |
| `scripts/_train_core.py` | ~150 | 공통 학습 루프 (ultralytics model.train 래퍼 + fallback 스캐폴드) |
| `scripts/_modifiers/__init__.py` | 5 | package marker |
| `scripts/_modifiers/prune_24.py` | ~100 | 2:4 magnitude pruning + custom_from_mask hook + finalize 검증 |
| `scripts/_modifiers/modelopt_sparsify.py` | ~60 | modelopt.torch.sparsity.sparsify 래퍼 |
| `scripts/_modifiers/modelopt_qat.py` | ~70 | modelopt.torch.quantization.quantize fake quant 삽입 |
| `scripts/run_qr_train_batch.sh` | ~30 | 3개 recipe 순차 학습 batch |
| `recipes/17_modelopt_int8_qat.yaml` | ~35 | 신규 recipe |
| `tests/test_training_schema.py` | ~60 | TrainingSpec pydantic 검증 |
| `tests/test_train_dispatch.py` | ~80 | modifier dispatch (mock) |
| `tests/test_modifiers_prune_24.py` | ~50 | 더미 Conv2d에 적용 후 2:4 패턴 확인 |
| `tests/test_modifiers_modelopt.py` | ~60 | modelopt apply (importorskip) |
| `tests/test_run_trt_trained_weights.py` | ~40 | `_resolve_weights` trained path 선택 |
| `tests/test_train_skip_logic.py` | ~40 | skip/force |

### 수정 파일

| 파일 | 변경 | 설명 |
|---|---|---|
| `scripts/_schemas.py` | +`TrainingSpec` class, `TechniqueSpec.training: Optional` | 스키마 확장 |
| `scripts/run_trt.py` | `_resolve_weights` 내 ~10 LOC | trained_weights/ 자동 조회 |
| `scripts/run_ort.py` | 동일 | (선택적) ORT 경로 확장 — 현재 scope에선 불필요, 미적용 |
| `recipes/07_trt_int8_sparsity.yaml` | `technique.training:` 섹션 추가 | 파킹 해제 |
| `recipes/11_modelopt_int8_sparsity.yaml` | 동일 | 파킹 해제 |
| `recipes/08_modelopt_int8_ptq.yaml` | 변경 없음 | calibration_samples 등 기존 유지 |
| `scripts/run_qr_batch.sh` | ORDER/RECIPES 맵에 #07/#11/#17 추가 | 평가 배치 확장 |
| `Makefile` | `PARKED`에서 #07/#11 제거, `train-qr` 타겟 신규 | 활성화 반영 |
| `.gitignore` | `trained_weights/`, `runs/`, `best_qr.pt` 추가 | 학습 산출물 + fine-tuned checkpoint 제외 |
| `pyproject.toml` | extras `modelopt = ["nvidia-modelopt>=0.15"]` | 의존성 명시 |
| `docs/architecture.md` | recipe 개수 20 active + 1 parked, Wave 5 섹션 추가 | 상태 갱신 |
| `docs/qr_barcode_eval_v2.md` | "Training-based recipes" 하위 섹션 append | 결과 추가 (학습 후) |
| `README.md` | best_qr.pt 복사 지시 섹션 신설 | 재현 절차 |

---

## 4. Schema Extension (`scripts/_schemas.py`)

```python
class TrainingSpec(BaseModel):
    base_checkpoint: str              # 프로젝트 루트 기준 상대경로 또는 절대경로
    epochs: int
    batch: int = 8
    workers: int = 4
    imgsz: int = 640
    lr0: float = 0.001                # QAT recipe는 0.0001로 override
    optimizer: str = "AdamW"
    seed: int = 42
    data_yaml: Optional[str] = None   # None이면 OMNI_TRAIN_YAML → qr_barcode.yaml
    modifier: Literal["prune_24", "modelopt_sparsify", "modelopt_qat"]
    prune_amount: Optional[float] = None
    quant_config: Optional[str] = "int8_default"


class TechniqueSpec(BaseModel):
    # ... 기존 필드 ...
    training: Optional[TrainingSpec] = None
```

### `load_recipe()` 확장
- `OMNI_WEIGHTS_OVERRIDE` 환경변수는 `recipe.model.weights` **와** `recipe.technique.training.base_checkpoint` 둘 다 덮어씀 (의미 일관성).
- `training.base_checkpoint` 경로는 프로젝트 루트 기준. `Path(__file__).parent.parent / base_checkpoint` 로 resolve.

---

## 5. Training Dynamics (HIGH 이슈 반영)

### 5.1 Learning Rate
- `prune_24`, `modelopt_sparsify`: `lr0: 0.001` (기본)
- `modelopt_qat`: `lr0: 0.0001` — fake quant scale 민감도 때문

### 5.2 AMP (Automatic Mixed Precision)
- `prune_24`: ultralytics 기본 `amp=True` 유지 (AMP 호환)
- `modelopt_sparsify`, `modelopt_qat`: `amp=False` **강제** — modelopt fake quant scale은 fp32 기대, AMP 활성 시 silent scale corruption 가능성

### 5.3 Best.pt / EMA 회피 (val leak 방지)
- ultralytics `model.train()` 후 **in-memory `yolo.model`** 을 직접 직렬화
- `last.pt`/`best.pt` 파일 경유하지 않음
- 근거: ultralytics trainer는 매 epoch val mAP 측정 + best checkpoint selection → QR val 133장이 eval에도 사용되므로 overlap bias. in-memory 참조는 val-based selection 개입 없음.
- 부작용: ultralytics EMA는 unconditional 생성되어 학습 중 보고되는 val mAP 은 EMA 가중치 기반 (modelopt wrapped 모델에서는 신뢰도 낮음). 우리는 별도 `run_trt.py` 평가 결과를 정본으로 쓰므로 무시 가능.

### 5.4 Sparsity Mask 수명
- `prune_24.apply()`: `torch.nn.utils.prune.custom_from_mask(module, 'weight', mask)` 호출 → forward_pre_hook 등록
- 학습 중: optimizer는 `weight_orig` 를 업데이트, forward 시 `weight = weight_orig * mask` 로 2:4 패턴 유지
- `prune_24.finalize()`: 모든 pruned module에 `prune.remove(module, 'weight')` → mask 영구 적용 + hook 제거 → clean state_dict

### 5.5 2:4 Pattern Verification
`prune_24.finalize()` 및 `modelopt_sparsify.finalize()` 공통:
```python
def _verify_2_4_pattern(w: torch.Tensor) -> bool:
    # weight를 4개 단위로 reshape해 각 group의 non-zero 수가 2 이하인지
    flat = w.reshape(-1, 4)
    return ((flat != 0).sum(dim=-1) <= 2).all().item()
```
위반 시 `RuntimeError` — 잘못된 checkpoint 저장 방지.

---

## 6. Model Serialization (ISSUE-1 해결)

modelopt-wrapped 모델은 단순 `state_dict` 저장/로드로 구조 복원 불가. `QuantModule` 등 내부 모듈이 생성돼야 하므로:

### Save (finalize)
```python
# scripts/_modifiers/modelopt_qat.py::finalize
import modelopt.torch.opt as mto
mto.save(yolo.model, str(out_pt))
# 파일 내용: modelopt_state + model state_dict 결합
```

### Load (run_trt.py)
```python
# recipe.technique.training.modifier == "modelopt_*" 일 때
from ultralytics import YOLO
import modelopt.torch.opt as mto
# 원형 YOLO26n architecture 로드용 plain checkpoint (recipe.model.weights 사용 — OMNI_WEIGHTS_OVERRIDE 적용된 best_qr.pt)
yolo = YOLO(recipe.model.weights)
mto.restore(yolo.model, str(trained_pt))  # modelopt 구조 복원 + trained weights 로드
```
`mto.restore()`는 구조를 `trained_pt`에 저장된 modelopt_state로 재구성하고, 같은 파일의 state_dict를 붙임. 초기 `YOLO(...)` 의 weights는 restore 시 덮어써짐 — architecture 출발점 역할만.

### `prune_24`는 예외
- `prune.remove()` 후 결과는 plain state_dict (mask가 weight에 영구 적용돼 0으로 박힘)
- 일반 `torch.save({'model': yolo.model.state_dict()}, ...)` 로 저장
- `run_trt.py`는 `prune_24` modifier에 대해선 기존 경로 그대로 (plain load)

### Runner 로드 로직 (`scripts/run_trt.py`)
```python
def _resolve_weights(recipe: Recipe) -> str:
    if recipe.technique.training:
        trained = Path("trained_weights") / f"{recipe.name}.pt"
        if not trained.exists():
            raise RuntimeError(
                f"{recipe.name} requires training. Run: "
                f"python scripts/train.py --recipe recipes/{recipe.name}.yaml"
            )
        return str(trained)
    # ... 기존 경로 ...
```

`run_trt.py`의 modelopt export 경로는 wrapped model을 인식하고 ONNX export 시 QDQ 노드로 변환 (modelopt 기본 동작). 추가 작업 없음.

---

## 7. Data Flow

### 학습 (예: #17 modelopt_int8_qat)

```
$ python scripts/train.py --recipe recipes/17_modelopt_int8_qat.yaml

[1] load_recipe → TrainingSpec.base_checkpoint = "best_qr.pt"
[2] ultralytics.YOLO("best_qr.pt") 로드
[3] modelopt_qat.apply(yolo, spec):
    - modelopt.torch.quantization.quantize(yolo.model, INT8_DEFAULT_CFG)
    - yolo.model 내부 Conv/Linear → QuantConv/QuantLinear 치환
[4] yolo.train(data=qr_barcode.yaml, epochs=30, lr0=0.0001, amp=False, name="17_modelopt_int8_qat")
    - 30 epochs QAT, gradient가 fake quant scale에도 흐름
    - runs/train/17_modelopt_int8_qat/ 에 ultralytics 로그 (무시)
[5] modelopt_qat.finalize(yolo, spec, out_pt):
    - mto.save(yolo.model, "trained_weights/17_modelopt_int8_qat.pt")
    - *.train.json 메타데이터 기록
```

### 평가

```
$ bash scripts/run_qr_batch.sh   # #17 포함 전체
  └── python scripts/run_trt.py --recipe recipes/17_*.yaml --out results_qr/17_*.json
       └── _resolve_weights() → trained_weights/17_*.pt 반환
       └── YOLO(base_arch).model에 mto.restore() → structure 복원
       └── 기존 modelopt export 경로 (ONNX with QDQ → TRT engine)
       └── 측정 + results_qr/17_*.json
```

### Batch 순서

```bash
# 학습 (한 번, 약 2시간)
bash scripts/run_qr_train_batch.sh  # #07 #11 #17 순차, 기존에 있으면 skip

# 평가 (기존 플로우)
bash scripts/run_qr_batch.sh        # 전체 21 recipes
```

---

## 8. Recipe YAML

### `recipes/17_modelopt_int8_qat.yaml` (신규)

```yaml
name: modelopt_int8_qat
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
  ultralytics_version: null
runtime:
  engine: tensorrt
  version: null
  dtype: int8
technique:
  name: int8_qat
  source: modelopt
  calibrator: null
  training:
    base_checkpoint: best_qr.pt
    epochs: 30
    batch: 8
    workers: 4
    imgsz: 640
    lr0: 0.0001          # QAT 전용 낮은 lr
    optimizer: AdamW
    seed: 42
    data_yaml: qr_barcode.yaml
    modifier: modelopt_qat
    quant_config: int8_default
hardware:
  gpu: null
  cuda: null
  driver: null
measurement:
  dataset: coco_val2017
  num_images: 500
  warmup_iters: 100
  measure_iters: 100
  batch_sizes: [1, 8]
  input_size: 640
  gpu_clock_lock: true
  seed: 42
constraints:
  max_map_drop_pct: 1.0
  min_fps_bs1: 30
```

### `recipes/07_trt_int8_sparsity.yaml` 패치

기존 `technique:` 블록에 추가:
```yaml
  training:
    base_checkpoint: best_qr.pt
    epochs: 60
    batch: 8
    workers: 4
    imgsz: 640
    lr0: 0.001
    optimizer: AdamW
    seed: 42
    data_yaml: qr_barcode.yaml
    modifier: prune_24
    prune_amount: 0.5
```

### `recipes/11_modelopt_int8_sparsity.yaml` 패치
동일 `training:` 블록, `modifier: modelopt_sparsify`.

---

## 9. Error Handling

| 단계 | 실패 | 처리 |
|---|---|---|
| base_checkpoint 부재 | FileNotFoundError | 메시지: "see README for best_qr.pt placement" |
| modelopt 미설치 | ImportError | "pip install nvidia-modelopt" hint |
| nc mismatch | 학습 시작 전 체크 | base_checkpoint 의 nc != data_yaml 의 nc → 명확한 에러 |
| 2:4 적용 불가 layer | 경고 + skip | Conv pattern 맞지 않는 layer는 skip, 로그 |
| `model.train()` OOM | 에러 캐치 | "lower `training.batch`" hint, exit 2 |
| `model.train()` wrapped model 거부 | **Hybrid fallback** | MVP에선 `NotImplementedError` + TODO 로그 |
| finalize 시 디스크 부족 | 에러 전파 | partial `.pt` 정리 |
| 2:4 패턴 검증 실패 | RuntimeError | finalize 시 assert |
| `run_trt.py`에서 trained_weights 없음 | RuntimeError | "run scripts/train.py first" |

### Batch 스크립트
- `run_qr_train_batch.sh`: 학습 실패 recipe는 `.pt` 생성 안 됨 → batch 계속 진행, exit 요약에 실패 목록 보고
- `run_qr_batch.sh`: trained_weights 없어서 평가 실패 시 기존 "Degrade, don't crash" 원칙 (meets_constraints=False JSON 저장)

---

## 10. Logging

### `trained_weights/{recipe_name}.train.json`
```json
{
  "recipe": "17_modelopt_int8_qat",
  "started_at": "2026-04-20T12:00:00Z",
  "finished_at": "2026-04-20T12:25:00Z",
  "duration_s": 1500,
  "base_checkpoint": "best_qr.pt",
  "epochs": 30,
  "modifier": "modelopt_qat",
  "training_config_hash": "sha256:...",
  "final_train_loss": 0.0321,
  "final_val_map_50_95_ema": 0.9180,
  "notes": "val mAP is EMA-based, use run_trt.py for authoritative eval"
}
```

### Skip 시
- `trained_weights/{recipe}.pt` + `.train.json` 있으면 기본 skip
- `--force` 플래그: 덮어쓰기
- stale 검출: `.train.json`의 `training_config_hash`와 현재 recipe 해시 불일치 → 경고만 (자동 재학습은 안 함)

---

## 11. Testing

### 단위 테스트 (CI 포함)

| 파일 | 검증 |
|---|---|
| `test_training_schema.py` | TrainingSpec 필드 검증, Literal 제약, YAML round-trip |
| `test_train_dispatch.py` | recipe.modifier → 올바른 `_modifiers/*` import (mock) |
| `test_modifiers_prune_24.py` | 더미 Conv2d(64,64,3) 에 적용 후 weight가 2:4 패턴 |
| `test_modifiers_modelopt.py` | `pytest.importorskip("modelopt")`, apply 후 QuantModule 치환 확인 |
| `test_run_trt_trained_weights.py` | `_resolve_weights`가 trained path 반환 (파일 있을 때) / RuntimeError (없을 때) |
| `test_train_skip_logic.py` | 기존 .pt 있으면 skip, `--force`로 재학습 |

### 통합 스모크 (수동, CI 제외)
- `OMNI_TRAIN_SMOKE=1 python scripts/train.py --recipe recipes/17_*.yaml` → `epochs=1, fraction=0.1` 으로 ~3분 dry-run
- `trained_weights/17_*.pt` 생성 확인
- 후속 `run_trt.py` → `.json` 생성 확인 (mAP 품질은 무시)

### Mock 전략
실제 ultralytics `model.train()` 은 mock. dispatch + schema 경로만 검증.

```python
def test_dispatch_modelopt_qat(monkeypatch, tmp_path):
    recipe = _build_recipe(modifier="modelopt_qat")
    called = {}
    monkeypatch.setattr(
        "scripts._modifiers.modelopt_qat.apply",
        lambda yolo, spec: called.setdefault("applied", True)
    )
    monkeypatch.setattr(
        "scripts._train_core._run_ultralytics_train",
        lambda *a, **k: None
    )
    train.main_with_recipe(recipe, force=True, out_dir=tmp_path)
    assert called["applied"]
```

---

## 12. Makefile + Scripts

### `Makefile`
```makefile
PARKED := brevitas_int8_entropy  # #22만 남음

train-qr:
	bash scripts/run_qr_train_batch.sh

all: (기존 그대로 — #07 #11 #17 자동 포함)
```

### `scripts/run_qr_train_batch.sh`
```bash
#!/usr/bin/env bash
set -u
export OMNI_COCO_YAML="$PWD/qr_barcode.yaml"
export OMNI_WEIGHTS_OVERRIDE="$PWD/best_qr.pt"

mkdir -p trained_weights

TRAINING_RECIPES=(
    07_trt_int8_sparsity
    11_modelopt_int8_sparsity
    17_modelopt_int8_qat
)

for r in "${TRAINING_RECIPES[@]}"; do
    out="trained_weights/${r}.pt"
    if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        continue
    fi
    echo "=== [$(date +%H:%M:%S)] training $r ==="
    python scripts/train.py --recipe "recipes/${r}.yaml" 2>&1 | tail -10
done

echo "all training done."
```

### `scripts/run_qr_batch.sh` 패치
`ORDER`, `RECIPES` 맵에 `07_trt_int8_sparsity`, `11_modelopt_int8_sparsity`, `17_modelopt_int8_qat` 추가 (모두 `run_trt`).

---

## 13. Known Limitations

1. **QAT 초기 scale calibration 생략** — `mtq.quantize(model, cfg)` 만 호출, `forward_loop` 없음. scale init 기본값으로 시작, 학습이 scale도 학습. 수렴 느릴 가능성. 미래 개선: 512장 COCO val calibration forward loop 추가.
2. **ultralytics EMA unconditional** — EMA kwarg로 끌 수 없음. 학습 중 보고되는 val mAP는 EMA 기반 (modelopt wrapped 모델에서 부정확). 해결: in-memory `yolo.model` 직접 직렬화로 사이드스텝. `run_trt.py` 최종 평가가 정본.
3. **fallback 미구현** — ultralytics `model.train()` 이 wrapped 모델을 거부하는 경우 MVP에선 `NotImplementedError`. 실제 실패 관찰 시 구현.
4. **Multi-GPU 미지원** — single GPU (`device="0"`) 가정.
5. **학습 결정성 제약** — torch CUDA ops는 기본적으로 non-deterministic. seed 고정해도 bit-exact 재현 불보장. 결과 JSON에 seed + base_checkpoint hash 기록.

---

## 14. Rollout Plan

1. Schema + modifier 스캐폴드 (단위 테스트 포함)
2. `train.py` + `_train_core` 구현 + skip 로직
3. prune_24 modifier 실제 구현 + 2:4 검증
4. modelopt_sparsify modifier
5. modelopt_qat modifier
6. `run_trt.py` `_resolve_weights` 확장 + modelopt restore 통합
7. `recipes/17_*.yaml` 신규, #07/#11 training 섹션 추가
8. Batch 스크립트 + Makefile
9. `.gitignore`, `pyproject.toml`, `README.md` 업데이트
10. Smoke 검증 (`OMNI_TRAIN_SMOKE=1`) → 실제 학습 실행
11. `run_qr_batch.sh` 실행 → 3개 새 결과 JSON
12. `docs/qr_barcode_eval_v2.md` 에 "Training-based recipes" 섹션 append
13. `docs/architecture.md` 갱신
14. 단일 커밋 또는 여러 waypoint 커밋

각 단계는 다음 writing-plans 단계에서 bite-size task로 분해.

---

## 15. README.md 추가 섹션 (best_qr.pt 복사 지시)

```markdown
## QR/Barcode fine-tuned checkpoint

재현에 `best_qr.pt` 필요 (QR/Barcode 2-class fine-tuned YOLO26n). 로컬 복사:

\`\`\`bash
cp "C:/Users/yeste/OneDrive/Desktop/QR_Barcode/QR_Barcode_detection/yolo26n_qrcode_barcode_bg/weights/best.pt" ./best_qr.pt
\`\`\`

외부 사용자는 자체 fine-tuned checkpoint (동일 nc=2 구조)로 대체 가능.
`.gitignore`에 추가되어 있음 — 커밋 대상 아님.
```

---

## 16. Documentation Updates

### `docs/qr_barcode_eval_v2.md` 에 추가될 섹션 (학습 완료 후)

```markdown
## Training-based recipes (신규)

baseline: `00 trt_fp32` mAP@50=0.9893, mAP@50-95=0.9328.

| # | Recipe | base | epochs | p50 ms | bs1 fps | mAP@50 | ΔmAP@50-95 |
|---|---|---|---|---|---|---|---|
| 07 | trt_int8_sparsity | best_qr.pt | 60 | ... | ... | ... | ... |
| 11 | modelopt_int8_sparsity | best_qr.pt | 60 | ... | ... | ... | ... |
| 17 | modelopt_int8_qat | best_qr.pt | 30 | ... | ... | ... | ... |

관찰:
- (학습 완료 후 기술)
```

### `docs/architecture.md`
- "18 active recipes" → "20 active + 1 parked" (브레비타스 #22만 파킹)
- Wave 5 섹션 신설 (QAT/Sparsity training)
- `trained_weights/` 디렉토리 설명

---

## 17. Dependencies

`pyproject.toml` 추가:
```toml
[project.optional-dependencies]
modelopt = ["nvidia-modelopt>=0.15"]
# all = [...]에도 modelopt 포함
```

현재 설치: `modelopt 0.43.0`, `ultralytics 8.4.27`, `torch>=2.3`. 이 조합에서 동작 검증됨.

---

## 18. Risks & Mitigations

| 위험 | 확률 | 영향 | 완화 |
|---|---|---|---|
| ultralytics `model.train()`이 modelopt wrapped 모델 거부 | M | H | Hybrid fallback 스캐폴드, 실패 시 custom trainer 작성 |
| 2:4 pattern이 일부 layer에서 불가 | H | L | 경고 후 skip, 전체 학습 계속 |
| QAT scale 발산 (lr0 문제) | L | H | lr0=1e-4 설정, 관찰 후 필요시 1e-5 |
| Windows workers>0 hang | L | M | `OMNI_TRAIN_WORKERS=0` 환경변수 fallback 제공 |
| 학습 결과 mAP 기대 이하 | M | M | QAT calibration 추가 (Known Limitation 1) 또는 epoch 증가 |
| modelopt state 직렬화 실패 | L | H | 설계에 명시적 `mto.save/restore` 경로, 테스트 포함 |

---

## 19. Acceptance Criteria

- [ ] `pytest tests/` 전체 pass (신규 6개 테스트 포함)
- [ ] `OMNI_TRAIN_SMOKE=1 bash scripts/run_qr_train_batch.sh` 3개 모두 성공
- [ ] 실제 학습 (non-smoke) 3개 모두 완료, `trained_weights/*.pt` 생성
- [ ] `bash scripts/run_qr_batch.sh` 후 `results_qr/{07,11,17}_*.json` 생성
- [ ] `docs/qr_barcode_eval_v2.md` Training-based recipes 섹션 채워짐
- [ ] 3개 추가 recipe 중 최소 1개가 `meets_constraints=True` (현실적으로 QAT가 후보)
- [ ] `make all` 여전히 완주 (기존 원칙 "degrade, don't crash")
