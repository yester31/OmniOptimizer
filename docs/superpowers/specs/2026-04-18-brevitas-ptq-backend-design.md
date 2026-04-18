# Brevitas PTQ 백엔드 (Wave 4) — Design

- **Date**: 2026-04-18
- **Status**: Draft, awaiting user review
- **Scope**: OmniOptimizer에 Brevitas 기반 INT8 PTQ 백엔드 추가 (5번째 INT8 소스)
- **Out of scope**: QAT, INT4/저비트, activation equalization, bias correction 고급 옵션
  (Wave 4 결과 본 뒤 별도 wave로 결정)

## 1. 동기

현재 INT8 PTQ 백엔드는 3종 (`trt_builtin`, `modelopt`, `ort_quant`). Intel Neural
Compressor는 Wave 3에서 YOLO26n attention block과 비호환으로 제거됨. Brevitas는
Xilinx의 PyTorch-native 양자화 라이브러리로, 타 백엔드에 없는 **GPTQ** 캘리브레이션을
제공하고 QDQ ONNX export를 지원해 기존 TRT 빌드 파이프라인에 드롭인 가능하다.

Wave 4의 목표는 Brevitas를 4번째 PTQ 백엔드로 편입해 YOLO26n에서의 mAP/latency
trade-off를 기존 3종과 공정하게 비교하는 것이다. QAT/저비트는 Wave 4 결과가 의미 있는
차이를 보이는 경우에 한해 Wave 5로 진행한다.

## 2. 아키텍처

기존 `TechniqueSpec.source` 디스패치 컨벤션을 그대로 따른다. QDQ ONNX → TRT 엔진
빌드 경로는 수정하지 않는다.

```
recipe YAML (source: brevitas, algo: percentile|mse|entropy|gptq)
  └─ run_trt.py::_prepare_onnx
       └─ _prepare_brevitas_onnx(fp32.onnx, calib_loader, algo, knobs)
            ├─ Brevitas graph_mode PTQ: QuantConv/QuantLinear 래핑
            ├─ calibrate on calib_loader (num_calib_batches)
            └─ export: brevitas.export.onnx.standard.export_onnx_qcdq
       └─ (quant_pre_process 적용 — 기존 modelopt 경로와 동일)
  └─ _build_engine             (변경 없음)
  └─ _make_trt_forward         (변경 없음)
```

## 3. 신규/수정 파일

### 신규
- `recipes/20_brevitas_int8_percentile.yaml`
- `recipes/21_brevitas_int8_mse.yaml`
- `recipes/22_brevitas_int8_entropy.yaml`
- `recipes/23_brevitas_int8_gptq.yaml`

### 수정
- `scripts/run_trt.py`
  - `_prepare_brevitas_onnx(...)` 헬퍼 신규
  - `_prepare_onnx` 디스패처 분기 1개 추가
  - `_SOURCE_TAG["brevitas"] = "_brev"` (engine cache 파일명 단축 — Windows MAX_PATH)
- `pyproject.toml`: `brevitas>=0.11`, `qonnx` extras에 추가
- `Makefile`: `recipe-20` ~ `recipe-23` 타겟 추가
- `docs/architecture.md`: 백엔드 수 4→5, 활성 레시피 수 18→22 갱신

## 4. QDQ 호환 제약 (필수)

Wave 3에서 확립된 QDQ→TRT 호환 체크리스트를 엄격히 준수한다.

- `weight_quant = Int8WeightPerChannelFloat` (per-channel, axis=0)
- `act_quant = Int8ActPerTensorFloat` (per-tensor, symmetric, zero_point=0)
- `bias_quant = None` (TRT가 `act_scale × weight_scale`로 INT32 bias 내부 계산)
- Export: `brevitas.export.onnx.standard.export_onnx_qcdq` — 표준 QDQ. QONNX 경유 시
  `qonnx-to-qcdq` 변환 필수 (TRT에 QONNX 플러그인 없음)
- Export 후 `onnxruntime.quantization.quant_pre_process` (shape inference + folding)
  적용 — 히스토그램 캘리브레이터 OOM 회피

## 5. 레시피 스키마

```yaml
# recipes/23_brevitas_int8_gptq.yaml (예)
name: brevitas_int8_gptq
model: yolo26n
precision: int8
technique:
  source: brevitas
  algo: gptq            # percentile | mse | entropy | gptq
  num_calib_batches: 64
  # algo별 knob (선택):
  percentile: 99.99     # algo=percentile
  gptq_blocksize: 128   # algo=gptq
calibration:
  dataset: coco_val
  num_images: 512
measure:
  warmup_iters: 50
  measure_iters: 300
  batch_sizes: [1, 8]
```

`Recipe`/`TechniqueSpec` pydantic 스키마는 `source`와 `algo`가 이미 자유 문자열
필드이므로 `_schemas.py` 변경 없음.

## 6. 캘리브레이션 데이터

기존 `calib_loader` (COCO val, 512장, bs=8) 재사용. `num_calib_batches`로 사용
배치 수 제어. 알고리즘 특성:
- **percentile**: activation scale = p99.99 (YAML knob)
- **mse**: activation scale을 MSE 최소화로 탐색
- **entropy**: KL divergence 기반 (TRT entropy와 수학적으로 유사, 구현은 상이)
- **gptq**: weight-only 보정 (Brevitas 고유, 타 백엔드 없음). activation은 percentile로 고정

## 7. 실패 모드 (Degrade, don't crash — CLAUDE.md 규약)

모든 실패는 `Result.notes`에 기록하고 `meets_constraints=False`로 반환. `make all`은
계속 진행한다.

- Brevitas import 실패 (미설치) → `notes="brevitas unavailable"`
- GPTQ가 attention block에서 shape 충돌 (INC SmoothQuant 선례) → `notes="brevitas gptq shape mismatch at <node>"`
- QDQ export 실패 → `notes="brevitas export failed: <err>"`
- TRT 빌드 실패 → 기존 `_build_engine` 예외 처리 경로 재사용

## 8. 검증 계획

### 사니티 (GPU 없이)
```bash
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('scripts').glob('*.py')]"
python -c "import sys; sys.path.insert(0,'.'); from scripts._schemas import load_recipe; import pathlib; [load_recipe(str(p)) for p in pathlib.Path('recipes').glob('2[0-3]_*.yaml')]"
```

### E2E
1. `make recipe-20` 통과 → results/20_*.json 생성 확인
2. `make recipe-21`, `22`, `23` 순차 통과
3. `make all` → report.md에 Brevitas 4개 행 추가 확인

### 성공 기준
- 4개 variant 중 최소 1개가 `mAP drop < 0.5` AND `fps(bs=1) ≥ FP16 baseline`
- 전체 4개가 에러 없이 JSON 결과 생성 (일부 notes 기록은 허용)

## 9. 리스크

| 리스크 | 확률 | 완화 |
|---|---|---|
| YOLO26n attention block에서 GPTQ shape 충돌 (INC 선례) | 중 | degrade 경로로 처리, 나머지 3개 variant는 유효 |
| Brevitas의 QDQ export가 TRT 8.6 플러그인과 미묘하게 다른 opset 사용 | 중 | `quant_pre_process` + polygraphy diagnose로 조기 검출 |
| `brevitas` + `qonnx` + 기존 `setuptools<81` 제약 충돌 | 낮 | Docker 이미지 빌드 시 확인. 충돌 시 extras 분리 |
| GPTQ calibration이 512장 기준 TRT 빌드보다 오래 걸려 timeout | 낮 | `num_calib_batches` 하한 (~16)로 튜닝 가능 |

## 10. 향후 작업 (Wave 4 이후)

Wave 4 결과에 따라 다음 중 선택:
- **Wave 5-A (QAT)**: mAP drop이 크면 Brevitas QAT로 parked #07/#11/#19 일부 해제
- **Wave 5-B (INT4)**: fps 여유가 크고 accuracy 마진이 남으면 weight-only INT4
- **Wave 5-C (중단)**: Brevitas가 기존 3종 대비 유의미한 pareto 개선 없으면 종료
