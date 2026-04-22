# Wave 10: ModelOpt FastNAS 구조적 채널 Pruning (#23–#24) — **REOPENED & SHIPPED 2026-04-22**

> **✅ SHIPPED 2026-04-22 (reopened after archive)**:
>
> 초기 archive 결정은 측정 프로토콜 오류(ultralytics val vs `scripts/run_trt.py`) 때문이었고, Phase 7/8에서 공정 비교로 뒤집힘. **modelopt.onnx.quantize** 전략을 FastNAS pruned ONNX에 적용해 `#11 modelopt_int8_sparsity` 와 동일한 QDQ 커버리지(Q=223 / DQ=223 / Detect head 42쌍)를 확보하자 fps 502 → **716** (+43%) 달성.
>
> **report_qr.md 반영 (2026-04-22)**:
> - Rank 4: `modelopt_fastnas_int8` — fps **716.3**, mAP 0.947 (−4.10%p), engine **5 MB** (baseline 38 MB, −88%)
> - Rank 5: `modelopt_fastnas_sp_int8` — fps **697.4**, mAP 0.948 (−4.02%p), engine **5 MB**
> - Baseline #1 `modelopt_int8_entropy` (fps 763.9) 대비 **93.8%** 유지하면서 **크기 12.4%** 수준
> - GPU recipe bank: 21 → **23 recipes**
>
> **해결 전략 핵심**: FastNAS 자체 pruning 폭은 15.7%에 한정되지만, **크기 / VRAM 절감은 압도적(-88%)**. 엣지/임베디드/메모리 제약 시나리오에서 실용 가치. constraints 는 baseline 1.5%p 대신 **5.0%p**로 완화해 ship.
>
> **산출물**:
> - `recipes/23_modelopt_fastnas_int8.yaml`, `recipes/24_modelopt_fastnas_sp_int8.yaml`
> - `results_qr/23_modelopt_fastnas_int8.json`, `results_qr/24_modelopt_fastnas_sp_int8.json`
> - `trained_weights/23_fastnas_p1_finetune/weights/best.pt` (FastNAS fine-tuned, 5.4 MB)
> - `trained_weights/23_fastnas_chain_ft/B_final.pt` (+ 2:4 sparsity FT, 10.5 MB)
> - `trained_weights/23_fair_bench_engines/{E,F}_*_bs1.engine` (최종 TRT engines)
> - `scripts/_spike_wave10_p1...p9_*.py` (재현 스크립트 9개)
> - `docs/improvements/2026-04-21-wave10-r1-spike-results.md` (spike 결과)
> - `docs/improvements/2026-04-22-wave10-pruning-extended-eval.md` (확장 실험)
>
> **초기 archive 기록 (참고용)**: Phase 1B/4C 측정이 ultralytics val 기반이라 fps가 실제보다 4배 낮게 나왔고, 이를 baseline (run_trt.py 기반 fps 763)과 직접 비교해 "속도 이득 없음"으로 오판. Phase 7 fair bench(`scripts/measure.py::measure_latency` + CUDA events)로 bias 제거 후 Phase 8 modelopt.onnx.quantize 로 Detect head QDQ 주입이 해결책임 확인.
>
> **Wave 10 교훈**:
> - **측정 프로토콜을 baseline과 일치시켜 비교하라** — ultralytics val은 preprocess/postprocess 포함 e2e, run_trt.py는 pure CUDA kernel.
> - **`modelopt.onnx.quantization.quantize`는 torch-level `mtq.quantize`를 우회하는 핵심 경로** — `QuantConv2d` ONNX export segfault 회피. 이미 학습된 ONNX에 QDQ 주입 방식이 더 robust.
> - FastNAS의 FX trace 한계(backbone 제외)는 여전히 존재 — pruning 폭 15.7%가 상한. 하지만 크기/VRAM 절감이 속도 이득과 **독립적**으로 실용 가치 제공.

> **⚠ 이하 원본 plan 보존용. 현재 ship 경로는 `recipes/23_*.yaml` 와 `scripts/_spike_wave10_p*.py` 참조.**

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (권장) 또는 `superpowers:executing-plans`. 체크박스(`- [ ]`)가 진행 트래커. Task 0 spike 통과 전에는 Task 1 이후로 진행하지 않는다.

**Goal:** GPU recipe bank를 ModelOpt의 **구조적 채널 pruning**으로 확장한다. 3 recipes (#23–#25) — FLOPs 30% / 50% 삭감 + (#25) pruning → INT8 entropy 체이닝. Wave 5 QR 학습 파이프라인을 재사용해 fine-tune 단계까지 자동화. 성공 시 현 GPU 최고(`modelopt_int8_entropy`, bs1 fps 763.9) 대비 **연산량 자체를 깎은** 후보군 확보.

**Architecture:** 변경 지점 7곳. (1) `scripts/_schemas.py::TrainingSpec.modifier` Literal에 `"modelopt_prune"` 추가 + `prune_flops_ratio` / `prune_search_iters` 신규 필드 + `prune_amount` ↔ `prune_flops_ratio` 상호배타 `@model_validator`. (2) `scripts/_modifiers/modelopt_prune.py` **신규** — `apply()`가 `mtp.prune(model, mode="fastnas", constraints={"flops": "<N>%"}, dummy_input, config={"data_loader": ..., "score_func": ..., "collect_func": ..., "checkpoint": ..., "max_iter_data_loader": N})` 호출(공식 API), `finalize()`는 기존 `modelopt_qat`/`sparsify` 와 동일하게 `mto.save`. (3) `scripts/_train_core.py`는 수정 없음 (modifier 디스패치는 기존 hook 재사용). (4) `scripts/_weights_io.py::_resolve_weights` 분기에 `modelopt_prune` 경로 + **pruned 전용 ONNX cache key tag** (`tag_suffix=f"_fastnas{ratio:.2f}"`) 추가 — baseline ONNX와 stem 충돌 방지. (5) `scripts/run_trt.py` ONNX export는 pruned 모델 별도 파일(`_prune{ratio}`)로 강제 재-export. (6) `recipes/23,24,25_*.yaml` 신규 + `Makefile` 타깃. (7) 필요 시 `qat-nvidia` ultralytics fork 도입 결정 (Task 0 Step 2에서 실측). `run_cpu.py` / `run_ort.py`는 **건드리지 않는다**.

**중요 리스크 (교차 검증에서 확인됨)**:
- (R1) FastNAS가 채널 수를 바꾸는 구조적 pruning이라 `YOLO(base_ckpt)` skeleton과 pruned state_dict shape 불일치 → `mto.restore` 실패 가능성 → `qat-nvidia` fork의 `save_model/load_checkpoint` 패턴 도입 필요할 수 있음 (Task 0 Step 2 (D) 시나리오에서 판정).
- (R2) ultralytics `YOLO.export(format="onnx")` 내부 `deepcopy`가 modelopt-wrapped 모델에서 실패한 전례 (Brevitas 경로 `run_trt.py:558-568`) — pruned 모델도 동일 경로라 deepcopy 실패 시 **ONNX export 전용 헬퍼** 작성 필요.
- (R3) FastNAS `prune()` 호출 자체가 subnet sampling × data_loader iter × score_func (매 subnet val) 을 수행. 단일 `prune()` 호출이 30분~3h 소요 가능 — 학습 시간 견적 상향 필요.

**Tech Stack:** `nvidia-modelopt>=0.15`, `torch>=2.8`, `ultralytics==8.4.27`, `tensorrt>=10.x`. Wave 5 `scripts/train.py` / `_train_core.py` / `_modifiers/` 파이프라인 재사용.

---

## Scope Boundaries

### In-scope (Wave 10)
- **알고리즘**: FastNAS (`mtp.prune(mode=[("fastnas", cfg)])`)
- **제약**: `flops` 기반 (`params`도 추가 기록하되 primary는 flops)
- **타겟 모델**: YOLO26n (기존)
- **타겟 HW**: RTX 3060 Laptop (Ampere SM 8.6) — pruning은 HW-agnostic, TRT FP16/INT8 실행 단계에서 HW 의존
- **Fine-tune**: Wave 5 QR dataset + QAT recipe `#17`의 lr 스케줄 반 (lr0=5e-4)

### Out-of-scope (future waves)
- **GradNAS** — BERT/QA 전용, CNN 가이던스 부재
- **Minitron** — LLM transformer depth 제거 기반
- **Latency-constraint** FastNAS — TRT build latency를 search loop 안에서 측정해야 해서 인프라 무거움. `flops` proxy로 충분히 유용
- **다중 teacher distillation** — 사용자가 Wave 11을 명시적으로 drop

### Assumptions (착수 전 Task 0에서 확인)
- FastNAS는 forward 중 `self.anchors`/`self.strides` 재할당을 건드리지 않아야 한다. **우회 전략**: `prune()` 진입 전에 `yolo.model(dummy_input)` warmup forward 1회 실행해 `Detect.shape`를 세팅 → 이후 subnet sampling 시 채널만 변하고 H/W는 불변이므로 `shape != self.shape` 분기(`head.py:178`)가 **재트리거되지 않음**. 이 가정이 Task 0 Step 1의 핵심 검증 대상.
- ultralytics `build_dataloader`가 modelopt `data_loader` 인자(BN 재-보정용)에 직접 feed 가능하다. `collect_func=lambda batch: batch["img"]` 어댑터로 ultralytics `batch` dict → tensor 변환 필요 (공식 y-t-g YOLO 튜토리얼 패턴).
- Pruned 모델이 `mto.save`/`mto.restore`만으로 일반 ultralytics `YOLO(base).export(onnx)` 파이프라인을 통과한다 **(가정, 실측 필요)**. 실패 시 `qat-nvidia` fork 또는 custom save/restore 로직 도입.
- 기존 ONNX export cache (`scripts/_weights_io.py::_export_onnx`) 가 stem 기반이라 **pruned vs baseline 충돌** 위험 있음 — Task 3에서 `tag_suffix` 추가로 분리.

---

## Recipe Map (#23–#25)

| # | name | source | dtype | flops 비율 | fine-tune | 후속 최적화 |
|---|---|---|---|---|---|---|
| 23 | `modelopt_fastnas_70pct_fp16` | `modelopt` | fp16 | 0.70× | 60 epochs | TRT FP16 |
| 24 | `modelopt_fastnas_50pct_fp16` | `modelopt` | fp16 | 0.50× | 80 epochs | TRT FP16 |
| 25 | `modelopt_fastnas_70pct_int8` | `modelopt` | int8 | 0.70× | 60 epochs | TRT INT8 entropy (PTQ 재호출) |

**번호 정책**: #23–#25은 Wave 1~5에서 예약만 해두었던 GPU 여유 공간. #26은 Wave 12 INT4, #27–28은 Wave 13 autocast/autotune, #29 parked, #30–#35 기존 CPU.

**Parked 후보** (Task 7 결과에 따라):
- `#23b` `modelopt_fastnas_70pct_int8_qat` — #25는 PTQ, QAT 체이닝은 학습 비용 큼
- `#24b` `modelopt_fastnas_30pct_fp16` — 너무 공격적, #24 결과 보고 판단

---

## Measurement Hygiene

기존 GPU 프로토콜(warmup 100 / measure 100, p50/p95/p99, NVML 메모리 델타) 그대로. 추가 필드:
- `Result.notes`에 `fastnas_flops_ratio=<achieved>`, `fastnas_params_ratio=<achieved>`, `search_steps=<N>`, `pruned_state_dict_mb=<size>` 기록.
- `model_size_mb`는 **pruned** PT 파일 크기(baseline 대비 축소 여부 검증용).

스키마 변경 없음 (`notes` 문자열로 충분).

---

## Task Dependency Graph

```
Task 0 (spike — BLOCKING gate)
   │
   ├─ Task 0 pass ──> Task 1 (schema) ──> Task 2 (modifier)
   │                                       │
   │                                       v
   │                     Task 3 (_weights_io mto.restore 분기)
   │                                       │
   │                                       v
   │                     Task 4 (recipe YAML × 3)
   │                                       │
   │                                       v
   │                     Task 5 (Makefile targets + batch script)
   │                                       │
   │                                       v
   │                     Task 6 (E2E eval + report update)
   │                                       │
   │                                       v
   │                     Task 7 (docs + CLAUDE.md 갱신)
   │
   └─ Task 0 fail ──> archive Wave 10, Wave 12로 이동
```

---

## Task 0 — FastNAS on YOLO26n spike (BLOCKING)

**목표**: FastNAS search가 YOLO26n에서 **실제 forward pass + 실 dataloader + 실 score** 로 완주하는지 확인. Wave 7 torch.export 블로커(anchors 재할당), `mto.save`/`restore` 체인 실패, ONNX re-export 실패 **3가지**를 단일 spike에서 관찰.

**산출물**: `scripts/_spike_wave10_r1.py`, `scripts/_spike_wave10_r1_restore.py`, `docs/improvements/2026-04-21-wave10-r1-spike-results.md`.

### Step 1 — spike 스크립트 작성 (`scripts/_spike_wave10_r1.py`)

공식 `modelopt.torch.prune` API (`mode="fastnas"` 문자열 + `max_iter_data_loader` + `collect_func`) 사용. `score_func`은 `yolo.val(..., fraction=0.01)` 실측. Detect head 우회는 `prune()` 진입 전 `model(dummy)` warmup forward로 `self.shape` 세팅.

- [ ] spike 스크립트 작성:
  ```python
  import torch, modelopt.torch.prune as mtp, modelopt.torch.opt as mto
  from ultralytics import YOLO
  from ultralytics.data import build_dataloader
  from ultralytics.data.utils import check_det_dataset

  yolo = YOLO("best_qr.pt")
  model = yolo.model.cuda().eval()
  dummy = torch.randn(1, 3, 640, 640, device="cuda")

  # (W4 우회) Detect.shape 세팅을 위한 warmup — 이후 subnet sampling에서
  # shape != self.shape 분기가 재트리거되지 않도록 anchors/strides 확정.
  with torch.no_grad():
      _ = model(dummy)

  # 실 dataloader (BN 재-보정용). max_iter_data_loader로 짧게 제한.
  data = check_det_dataset("qr_barcode.yaml")
  loader = build_dataloader(
      dataset=data["train"], batch=8, workers=2, shuffle=False, rank=-1,
  )

  # collect_func: ultralytics batch dict -> tensor (공식 y-t-g 튜토리얼 패턴)
  def collect_func(batch):
      return batch["img"].cuda().float() / 255.0

  # Detect forward 가 실제로 실행되는지 counter로 검증 (B3 재현 가드)
  from ultralytics.nn.modules.head import Detect
  _forward_count = {"n": 0}
  _orig_forward = Detect.forward
  def _counting_forward(self, *args, **kwargs):
      _forward_count["n"] += 1
      return _orig_forward(self, *args, **kwargs)
  Detect.forward = _counting_forward

  # score_func: 빠른 val (QR train의 1%) — dataloader와 별도
  def score_func(m):
      yolo.model = m
      results = yolo.val(data="qr_barcode.yaml", imgsz=640, device="cuda",
                         fraction=0.01, verbose=False, plots=False, save=False)
      return float(results.box.map50)

  # FastNAS — 공식 API. max_iter_data_loader=2 로 BN 재보정 짧게
  pruned, _ = mtp.prune(
      model=model,
      mode="fastnas",                          # 문자열 (B1)
      constraints={"flops": "90%"},            # string percent (B1)
      dummy_input=dummy,
      config={
          "data_loader": loader,
          "score_func": score_func,
          "collect_func": collect_func,         # B1: 필수
          "checkpoint": "modelopt_search_checkpoint_spike.pth",
          "max_iter_data_loader": 2,           # B1: 짧은 BN 재보정
      },
  )

  # 성공 게이트 — Detect.forward 가 실제로 호출됐는가
  assert _forward_count["n"] > 0, "Detect.forward 호출 0회 — FastNAS가 " \
      "forward를 돌리지 않음. score_func/collect_func 구성 오류 의심."

  print(f"[spike] search OK, Detect.forward calls = {_forward_count['n']}")
  print(f"[spike] original FLOPs  = {mtp.utils.compute_flops(model, dummy):.3e}")
  print(f"[spike] pruned   FLOPs  = {mtp.utils.compute_flops(pruned, dummy):.3e}")

  # save/restore 체인 검증용 save
  mto.save(pruned, "trained_weights/_spike_wave10_pruned.pt")
  ```

### Step 1.5 — save/restore 체인 검증 (`scripts/_spike_wave10_r1_restore.py`)

B2 리스크 격리. Step 1과 **별도 프로세스**에서 실행해야 in-process state 의존을 배제.

- [ ] 스크립트 작성:
  ```python
  import torch, modelopt.torch.opt as mto
  from ultralytics import YOLO

  # base skeleton 로드 — pruned 아님
  yolo = YOLO("best_qr.pt")
  # 채널 수가 다른 state_dict를 restore — modelopt이 구조 복원 지원하는지
  try:
      mto.restore(yolo.model, "trained_weights/_spike_wave10_pruned.pt")
      print("[restore] mto.restore OK")
  except (RuntimeError, KeyError, AssertionError) as e:
      print(f"[restore] FAIL: {type(e).__name__}: {e}")
      raise SystemExit(1)

  # ONNX re-export — Brevitas에서 목격된 deepcopy 실패 재현 여부
  try:
      yolo.export(format="onnx", imgsz=640, dynamic=True,
                  half=False, simplify=True)
      print("[export] ONNX export OK")
  except Exception as e:
      print(f"[export] FAIL: {type(e).__name__}: {e}")
      raise SystemExit(2)
  ```

### Step 2 — 4가지 시나리오 분기

실행 결과에 따라:

- [ ] **(A) Both PASS** — Step 1 exit 0 + Step 1.5 exit 0. `fastnas_converter: native` 로 기록 후 Task 1 착수.
- [ ] **(B) Anchor mutation AssertionError** (Wave 7 재현) — Step 1의 `Detect.forward` 중 assertion. → 별도 monkey-patch R&D 2~4h 판단. 해결 가능하면 Step 3, 아니면 archive.
- [ ] **(C) Step 1 other crash** — stacktrace 분석. 일반적으로 `collect_func`/`data_loader` shape 불일치, `mtp.utils.compute_flops` 내부 에러. 수리 후 재실행.
- [ ] **(D) Step 1 PASS but Step 1.5 FAIL** — `mto.restore` 또는 `YOLO.export` 실패. **qat-nvidia fork 도입** 또는 **custom save/restore 구현** 결정 필요. fork 도입 시 `pyproject.toml`에 git dependency 추가 + 2~4h 추가 예산.

### Step 3 (조건부, B 시나리오) — Detect head 우회 강화

Step 1의 warmup forward로 `self.shape` 세팅이 충분치 않은 경우:

- [ ] `ultralytics.nn.modules.head.Detect._get_decode_boxes`를 monkey-patch — forward 진입 시 `self.anchors is None` 또는 `shape != self.shape` 조건을 **무조건 False로 단락**하고 anchors/strides를 사전 계산 dict에서 lookup. 구체 패턴:
  ```python
  # Detect.forward 호출 전 _cached_anchors, _cached_strides 세팅
  _ANCHORS_CACHE = {}
  def _patched_get_decode_boxes(self, ...):
      key = (self.nl, self.stride[0].item())
      if key not in _ANCHORS_CACHE:
          _ANCHORS_CACHE[key] = make_anchors(...)   # 1회만
      self.anchors, self.strides = _ANCHORS_CACHE[key]
      return orig_decode(self, ...)
  ```

### Step 4 — spike 결과 문서화

- [ ] `docs/improvements/2026-04-21-wave10-r1-spike-results.md` 작성 (Wave 7/8 spike 문서 포맷 참조). 4개 시나리오 중 어느 것이 재현됐는지 + 측정된 원본/pruned FLOPs + search 소요 시간 기록.

**Gate**:
- (A) → Task 1 진행, 견적 확정.
- (B) Step 3 성공 → Task 1 진행 + `scripts/_modifiers/modelopt_prune.py`에 monkey-patch 포함.
- (D) qat-nvidia fork 선택 → Task 1 진행 + `pyproject.toml` 수정 (Task 1에 하위 체크박스 추가).
- (B+Step 3 실패) 또는 (C resolved 후 재현) 또는 (D fork 불가) → Wave 10 ARCHIVED, Wave 9 DirectML로 이동.

---

## Task 1 — Schema 확장

- [ ] `scripts/_schemas.py::TrainingSpec.modifier` Literal에 `"modelopt_prune"` 추가:
  ```python
  modifier: Literal["prune_24", "modelopt_sparsify", "modelopt_qat", "modelopt_prune"]
  ```
- [ ] `TrainingSpec`에 pruning 전용 필드 추가:
  ```python
  prune_flops_ratio: Optional[float] = Field(default=None, gt=0, le=1)  # 0.70 = 70% 유지
  prune_search_iters: Optional[int] = Field(default=None, gt=0)         # FastNAS max_iter_data_loader
  ```
- [ ] **(W1)** `prune_amount` ↔ `prune_flops_ratio` 상호배타 `@model_validator` 추가 — `modifier=="prune_24"` 이면 `prune_amount` 필수 / `prune_flops_ratio=None`, `modifier=="modelopt_prune"` 이면 그 반대. 위반 시 `ValueError` raise:
  ```python
  @model_validator(mode="after")
  def _validate_prune_fields(self):
      if self.modifier == "modelopt_prune":
          if self.prune_flops_ratio is None:
              raise ValueError("modelopt_prune requires prune_flops_ratio")
          if self.prune_amount is not None:
              raise ValueError("prune_amount is prune_24-only; use prune_flops_ratio")
      elif self.modifier == "prune_24":
          if self.prune_flops_ratio is not None:
              raise ValueError("prune_flops_ratio is modelopt_prune-only")
      return self
  ```
- [ ] **(D 시나리오 조건부)** Task 0 Step 2 에서 `qat-nvidia` fork 도입 결정 시 `pyproject.toml`에 git dependency 추가:
  ```toml
  [project.optional-dependencies]
  gpu = [
      "ultralytics @ git+https://github.com/NVIDIA/ultralytics-modelopt.git@qat-nvidia",
      ...
  ]
  ```
- [ ] `tests/test_schemas.py`에 round-trip + 상호배타 validator 테스트 추가 (기존 QAT/sparsify 테스트 패턴 그대로)
- [ ] `mypy scripts --ignore-missing-imports --no-strict-optional` 통과 확인

---

## Task 2 — `scripts/_modifiers/modelopt_prune.py` 신규

- [ ] 파일 작성 — Task 0 Step 1의 spike 스크립트 구조를 그대로 포팅 (이미 API 검증 완료). `apply()` / `finalize()` 2함수:
  - `apply(yolo, spec)`:
    - **(B1 공식 API)** `mode="fastnas"` (문자열), `constraints={"flops": f"{spec.prune_flops_ratio*100:.0f}%"}` (string percent 형식)
    - **Warmup forward** — `yolo.model(dummy)` 1회로 `Detect.shape` 세팅 (anchor 재할당 우회)
    - **dataloader**: `build_dataloader(check_det_dataset(spec.data_yaml)["train"], batch=spec.batch, workers=spec.workers)`
    - **collect_func**: `lambda batch: batch["img"].cuda().float() / 255.0`
    - **score_func**: `yolo.val(data=spec.data_yaml, imgsz=spec.imgsz, device="cuda", fraction=0.1, verbose=False, plots=False, save=False).box.map50` — fraction=0.1로 val 시간 제한 (full val은 분 단위, 10%면 수십초)
    - `max_iter_data_loader = spec.prune_search_iters or 8`
    - **Silent no-op guard**: `pruned_flops > original_flops * (spec.prune_flops_ratio + 0.05)` 이면 `RuntimeError` raise
    - (Task 0 Step 2 B 시나리오 해결 필요 시) Detect head monkey-patch 포함
  - `finalize(yolo, spec, out_pt)`: `mto.save(yolo.model, str(out_pt))` — 기존 sparsify/qat와 동일. `qat-nvidia` fork 사용 시 추가로 `yolo.save_model(out_pt_yaml)` 호출해 modelopt_state를 별도 YAML로도 저장
- [ ] docstring에 Wave 10 spec 링크 + "pruned model은 채널 수가 변경됨 — ONNX cache key는 Task 3의 `tag_suffix=f'_fastnas{ratio:.2f}'` 로 baseline과 분리 필수" 명시
- [ ] `scripts/_modifiers/__init__.py` docstring 한 줄 추가 (기존 QAT/sparsify 목록에 `modelopt_prune` 포함)
- [ ] 단위 테스트 `tests/test_modelopt_prune_apply.py` 추가 — dummy 2-layer CNN으로 apply 호출 + FLOPs 감소 검증 (ultralytics 의존 없는 smoke test)

---

## Task 3 — `scripts/_weights_io.py` pruned restore + ONNX cache 분리

- [ ] `_resolve_weights` 내부 분기에 `modelopt_prune` 케이스 추가:
  - 기본 경로: Task 0 Step 1.5 시나리오 (A) 통과 가정 — `YOLO(base_ckpt)` → `mto.restore(yolo.model, pruned_pt)` 로 modelopt 내부 structural diff 재적용
  - **(D 시나리오)** `qat-nvidia` fork 도입 시 `yolo.load_checkpoint(pruned_pt)` + `yolo.load_modelopt_state(pruned_pt.with_suffix('.yaml'))` 패턴 (fork 문서 참조)
  - shape mismatch 예외(`RuntimeError: size mismatch`) catch → 명시적 에러 메시지로 "qat-nvidia fork 필요" 안내
- [ ] **(B2 / R1 / R2)** ONNX cache key 분리 — 기존 `_export_onnx(weights, imgsz, ...)` 의 stem 기반 캐시(`{stem}_{imgsz}_fp32_dyn.onnx`)가 baseline(`best_qr`)과 pruned(`best_qr`) 충돌 위험:
  ```python
  # _export_onnx 시그니처 확장
  def _export_onnx(..., tag_suffix: str = ""):
      cache_name = f"{stem}{tag_suffix}_{imgsz}_fp32_dyn.onnx"
      ...
  # run_trt._prepare_onnx → recipe가 modelopt_prune 사용하면
  # tag = f"_fastnas{spec.prune_flops_ratio:.2f}" 로 호출
  ```
- [ ] **(R2)** `YOLO.export(format="onnx")` 내부 `deepcopy` 실패 대비 — Brevitas 경로(`run_trt.py:558-568`) 전례 참조. 실패 시 **ONNX export 전용 헬퍼** 작성:
  ```python
  def _export_pruned_onnx(yolo, onnx_path, imgsz):
      # deepcopy 대신 state_dict swap 방식
      import onnx
      model = yolo.model.eval().cuda()
      dummy = torch.randn(1, 3, imgsz, imgsz, device="cuda")
      torch.onnx.export(model, dummy, str(onnx_path),
                        dynamic_axes={"images": {0: "batch"}},
                        opset_version=17, do_constant_folding=True)
      onnx.checker.check_model(str(onnx_path))
  ```
- [ ] `tests/test_run_cpu_imports_without_tensorrt` 패턴 참고해서 `test_weights_io_pruned_restore_smoke.py` 추가 (CPU-only 환경에서도 dry-run)
- [ ] `tests/test_onnx_cache_key_collision.py` — `tag_suffix` 파라미터가 baseline과 pruned ONNX를 다른 파일로 분리하는지 회귀 방지

---

## Task 4 — Recipe YAML × 3

- [ ] `recipes/23_modelopt_fastnas_70pct_fp16.yaml`:
  ```yaml
  name: modelopt_fastnas_70pct_fp16
  model: {family: yolo26, variant: n, weights: yolo26n.pt}
  runtime: {engine: tensorrt, dtype: fp16}
  technique:
    name: fastnas_70pct_fp16
    source: modelopt
    training:
      base_checkpoint: best_qr.pt
      epochs: 60
      lr0: 0.0005
      modifier: modelopt_prune
      prune_flops_ratio: 0.70
      prune_search_iters: 8
      data_yaml: qr_barcode.yaml
      optimizer: AdamW
  hardware: {requires_compute_capability_min: 7.5}
  measurement: {dataset: coco_val2017, num_images: 500, warmup_iters: 100,
                measure_iters: 100, batch_sizes: [1, 8], input_size: 640,
                gpu_clock_lock: true, seed: 42}
  constraints: {max_map_drop_pct: 1.5, min_fps_bs1: 30}
  ```
- [ ] `recipes/24_modelopt_fastnas_50pct_fp16.yaml` — `prune_flops_ratio: 0.50`, `epochs: 80`, `max_map_drop_pct: 2.5`
- [ ] `recipes/25_modelopt_fastnas_70pct_int8.yaml` — **`base_checkpoint: trained_weights/23_modelopt_fastnas_70pct_fp16.pt`** (W2: #23의 학습 결과 재사용, OMNI_WEIGHTS_OVERRIDE 의존 없이 static path), `prune_flops_ratio: 0.70`, `epochs: 60`, `dtype: int8`, `calibrator: entropy`, `calibration_samples: 512`, `calibration_dataset: coco_val2017`, `max_map_drop_pct: 1.5`. **주의**: `#25.training.modifier`는 `modelopt_qat` (INT8 PTQ 체인) — `modelopt_prune`이 아님. pruning은 #23에서 이미 완료된 상태. validator는 `modifier=modelopt_qat`이면 `prune_flops_ratio=None`을 요구하므로 #25 YAML에서 `prune_flops_ratio` 필드 생략.
- [ ] `pytest tests/test_schemas.py -q` 통과 확인

---

## Task 5 — Makefile + batch 스크립트

- [ ] **(W2)** `Makefile`에 타깃 추가 — `recipe-25`는 `recipe-23`에 **명시적 의존성** (prerequisite). `OMNI_WEIGHTS_OVERRIDE` 환경변수 대신 recipe YAML의 static `base_checkpoint` 경로 사용 (Task 4에서 명시):
  ```makefile
  TRAINED := trained_weights

  $(TRAINED)/23_modelopt_fastnas_70pct_fp16.pt:
  	python scripts/train.py --recipe recipes/23_modelopt_fastnas_70pct_fp16.yaml --out $@

  results/23_fastnas_70pct_fp16.json: $(TRAINED)/23_modelopt_fastnas_70pct_fp16.pt
  	python scripts/run_trt.py --recipe recipes/23_modelopt_fastnas_70pct_fp16.yaml --out $@

  recipe-23: results/23_fastnas_70pct_fp16.json

  # recipe-24 동일 패턴
  recipe-24: results/24_fastnas_50pct_fp16.json

  # recipe-25는 #23 pruned 가중치를 base_checkpoint로 재사용 (static path)
  $(TRAINED)/25_modelopt_fastnas_70pct_int8.pt: $(TRAINED)/23_modelopt_fastnas_70pct_fp16.pt
  	python scripts/train.py --recipe recipes/25_modelopt_fastnas_70pct_int8.yaml --out $@

  results/25_fastnas_70pct_int8.json: $(TRAINED)/25_modelopt_fastnas_70pct_int8.pt
  	python scripts/run_trt.py --recipe recipes/25_modelopt_fastnas_70pct_int8.yaml --out $@

  recipe-25: results/25_fastnas_70pct_int8.json

  wave10: recipe-23 recipe-24 recipe-25
  ```
- [ ] `scripts/run_qr_batch.sh` (있으면) 에 #23–#25 추가
- [ ] `make recipe-23` smoke 테스트 — `OMNI_TRAIN_SMOKE=1` 환경변수 (기존 `_train_core.py:69,85` 참조) 로 epochs=1 강제 실행. `OMNI_TRAIN_SMOKE=1 make wave10` 이 exit 0로 끝나는지 확인.

---

## Task 6 — E2E eval + report 갱신

- [ ] **(W5)** `make wave10` 전체 실행. 예상 총 시간 재산정:
  | 단계 | 예상 시간 (RTX 3060 Laptop) |
  |---|---|
  | #23 FastNAS `prune()` search (subnet sampling × 8 × val fraction=0.1) | **1~2h** |
  | #23 fine-tune (60 epoch, bs=8) | 3~4h |
  | #24 FastNAS `prune()` search | 1~2h |
  | #24 fine-tune (80 epoch) | 4~5h |
  | #25 `modelopt_qat` (PTQ + 60 epoch) — #23 가중치 재사용 | 3~4h |
  | TRT build + measure (3 recipes × 2 bs) | ~30min |
  | **합계** | **12~17h** |

  최초 plan의 "6~10h"는 FastNAS search 자체 시간을 빠뜨린 과소 견적. 실기 예산은 **12~17h 연속** 또는 **2일 분할**. `make wave10`는 실패 시 resume 가능 (Makefile 의존성이 prerequisite 기반으로 재실행 영속).
- [ ] `report_qr.md` 재생성 (`python scripts/recommend.py --results results/ --out report_qr.md`)
- [ ] 결과 검증 — 3개 recipe 각각:
  - `meets_constraints=True`
  - `Result.notes` 에 `fastnas_flops_ratio=<achieved>` 가 목표값의 ±5% 이내
  - `Result.notes` 에 `search_duration_s`, `fastnas_converter`, `pruned_state_dict_mb` 필드 존재
- [ ] Ranking 변동 관찰 — #23/#25 가 기존 #1 `modelopt_int8_entropy` (fps 763.9) 를 넘어서는지. 예상치(previous discussion): #25 ~880~1000 fps.

---

## Task 7 — 문서 갱신

- [ ] `docs/architecture.md` "Recipes" 섹션에 #23–#25 추가 + FastNAS pruning 흐름 1단락 추가
- [ ] `CLAUDE.md` "Current scope" 업데이트 — 28 recipes → 31 recipes, GPU backends에 `modelopt_prune` 추가
- [ ] `README.md` recipe 표 갱신
- [ ] Wave 10 완료 커밋 메시지: `feat(wave10): ModelOpt FastNAS pruning (#23-#25) — <결과 요약>`

---

## Definition of Done

1. `pytest tests/ -q` 전체 green
2. `mypy scripts --ignore-missing-imports --no-strict-optional` 통과
3. `results/23_*.json`, `results/24_*.json`, `results/25_*.json` 3개 존재 + `Result.notes`에 `fastnas_flops_ratio` 기록
4. `report_qr.md`에 #23–#25 랭킹 반영
5. `docs/plans/2026-04-21-wave10-modelopt-fastnas-pruning.md` (이 파일) DONE 스탬프 추가
6. 3개 recipe 중 최소 1개가 기존 GPU Top-5 안에 진입 → 이 경우 `CLAUDE.md`의 GPU 최고 추천 갱신

---

## Rollback (W3: 부분 성공 시나리오 포함)

Task 6 결과에 따라 per-recipe 판정:

| 통과 recipe 수 | 조치 |
|---|---|
| **3/3** | Full ship. Task 7 진행. |
| **2/3** | 실패 recipe YAML을 `recipes/_parked/` 로 이동 (삭제 아님) + `docs/improvements/2026-04-21-wave10-partial.md` 작성 (실패 원인, mAP drop 수치, 재시도 조건). 나머지 2개는 ship + report에 반영. `_modifiers/modelopt_prune.py` / `_schemas.py` 변경 **유지** (2/3 성공이 인프라 가치 입증). |
| **1/3** | 실패 2개 parked, 성공 1개 ship. `CLAUDE.md`에 "부분 성공 Wave"로 기록. |
| **0/3** | Full archive: (1) recipe YAML 3개 삭제, (2) `_modifiers/modelopt_prune.py` / `_schemas.py` 변경 revert, (3) 이 plan에 `ARCHIVED` 마킹 + `docs/improvements/2026-04-21-wave10-archive.md` 작성, (4) Wave 9 DirectML EP로 이동 (Wave 12는 이미 archive). |

**Partial success 판정 기준** — recipe가 통과로 간주되려면:
- `meets_constraints=True` (기본)
- 추가로 `fastnas_flops_ratio` 가 목표 ±10% 이내 (±5% 는 엄격 기준, rollback용은 관대)
- 이 둘 다 만족하지만 Ranking에서 기존 #1 `modelopt_int8_entropy` 를 넘지 못해도 **통과** — pruning은 "속도"보다 "모델 크기 / 엣지 배포 가능성" 가치도 있음. 단 `report_qr.md`의 Recommendation 섹션에는 기존 #1 유지 문구 명시.
