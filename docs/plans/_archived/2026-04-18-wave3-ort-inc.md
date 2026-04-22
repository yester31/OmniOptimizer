# Wave 3: ONNX Runtime Quantization + Intel Neural Compressor 통합

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** OmniOptimizer의 INT8 백엔드를 `trt_builtin`/`modelopt` 2종에서 **4종**으로 확장한다. ONNX Runtime Quantization(`ort_quant`)과 Intel Neural Compressor(`neural_compressor`)를 각각 디스패치 가능한 `technique.source`로 추가하고, 각 백엔드에서 실행 가능한 모든 캘리브레이션/알고리즘 축을 레시피로 한 장씩 뽑아 mAP/레이턴시 공간을 넓힌다.

**Architecture:** 변경 지점은 3곳으로 국한된다. (1) `scripts/_schemas.py::TechniqueSpec.source` Literal 확장. (2) `scripts/run_trt.py`에 `_prepare_ort_quant_onnx`, `_prepare_inc_onnx` 두 함수 신설 + 기존 `_prepare_onnx` 디스패처 분기 추가. (3) `recipes/`에 신규 YAML 7장 (#13–#19). `_build_engine` / `_make_trt_forward`는 modelopt 경로와 동일하게 재사용한다 — 세 백엔드 모두 QDQ-ONNX를 내놓으므로 TRT explicit quantization 경로가 동일하게 처리한다. INC의 QAT 레시피(#19)는 `docs/plans/` 기반 스펙만 작성하고 실행은 parked (training pipeline 도입 후 재평가 — `trt_int8_sparsity` / `modelopt_int8_sparsity`와 동일 취급).

**Tech Stack:** `onnxruntime` (>=1.17, `quantization` 서브모듈), `neural-compressor` (>=2.5,<3.0), ultralytics, pydantic, TensorRT 10.x.

---

## v2 Patches — Expert Review Fixes (2026-04-18, apply these BEFORE the Task steps below)

v1 Task 1–8 섹션이 남아있지만 **아래 패치가 v1과 충돌하면 패치가 우선**합니다. 파일/테스트/커밋 granularity는 동일.

### P0-A. ORT `quantize_static` symmetric 옵션 누락 (가장 중요)

v1 Task 2 Step 2의 `quantize_static(...)` 호출은 `extra_options`를 누락했습니다. ORT 기본값은 **asymmetric activation** — TRT explicit QDQ 경로가 symmetric을 요구하므로 silent fallback이 발생해 mAP/fps 둘 다 기대치와 어긋납니다 (`trt_builtin` INT8에서 겪은 -7.9%p drop 재현 리스크).

필수 추가 인자:

```python
quantize_static(
    model_input=str(clean_onnx),
    model_output=str(cached),
    calibration_data_reader=_NumpyReader(calib_arr, input_name),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
    calibrate_method=method_map[calibrator],
    nodes_to_exclude=nodes_to_exclude or None,
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "AddQDQPairToWeight": True,
        "DedicatedQDQPair": True,
    },
)
```

modelopt가 내부적으로 하는 설정과 동등. 누락 시 Wave 3 측정치 자체가 오독됨.

### P0-B. INC 버전 핀 — `>=2.5,<3.0`

v1 Tech Stack은 `neural-compressor>=3.0`을 요구하지만 Task 4의 `PostTrainingQuantConfig(recipes={"smooth_quant": True, ...})` 패턴은 **INC 2.x API**. 3.x는 `SmoothQuantConfig` 별도 클래스로 분리됐습니다. 코드-의존성 모순 해소:

- `pyproject.toml` extras: `inc = ["neural-compressor>=2.5,<3.0"]`
- Tech Stack 문구도 `>=2.5,<3.0`으로 정정 (위 헤더 반영 완료).

### P0-C. INC ONNX 경로 SmoothQuant 가용성 — 선체크 필수

SmoothQuant는 원래 torch-level activation smoothing 변환. INC의 ONNX backend에서 정식 지원되는지 **Task 4 착수 전에 실기 확인**:

```bash
python -c "
from neural_compressor.config import PostTrainingQuantConfig
c = PostTrainingQuantConfig(
    approach='static', backend='default',
    recipes={'smooth_quant': True, 'smooth_quant_args': {'alpha': 0.5}},
)
print('config ok:', c.recipes)
"
```

실패하면 **Plan B**: torch 모델에 `neural_compressor.torch.quantization` SmoothQuant를 적용 → export QDQ ONNX (modelopt torch 경로와 구조 유사). Plan B 폴백을 Task 4에 명시하고, 대안 경로 구현은 `_apply_modelopt_sparsify`와 같은 헬퍼로 배치.

### P0-D. `TechniqueSpec` QAT 필드 명시 추가

v1 Task 5는 "pydantic `extra="ignore"`이면 필드 추가" 조건부. 현재 `_schemas.py`는 `model_config` 미설정 = pydantic 기본 `extra="ignore"` 확인됨. `#19 inc_int8_qat` 레시피의 `qat_epochs`/`qat_lr`이 **조용히 무시**되므로, **Task 1에서 TechniqueSpec에 명시 추가**:

```python
class TechniqueSpec(BaseModel):
    # ... 기존 필드 ...
    # QAT-only fields (used by recipes targeting training-aware quantization).
    # v1 only writes these for documentation; no runner currently consumes them.
    qat_epochs: Optional[int] = None
    qat_lr: Optional[float] = None
```

Task 1 실패 테스트에도 이 두 필드의 기본값/커스텀값 파싱 테스트 1줄 추가.

### P1-E. `source_suffix` 축약 매핑 (Windows 경로 260자 대비)

v1의 `source_suffix = "_{source}"`는 `_neural_compressor` 18자를 생성. 엔진 캐시 파일명 총 길이가 `results/_engines/yolo26n_640_inc_smoothquant_512_s42_dyn_int8_neural_compressor_bs8_v2.engine` 수준까지 늘어나 Windows 경로 제한에 근접. `run_trt.py`에 축약 매핑 헬퍼 추가 (Task 2에서 함께):

```python
_SOURCE_TAG = {
    "trt_builtin": "",
    "modelopt": "_modelopt",
    "ort_quant": "_ort",
    "neural_compressor": "_inc",
}
# run() 내부:
source_suffix = _SOURCE_TAG.get(source, f"_{source}")
```

기존 modelopt 엔진 캐시 hit은 `_modelopt` 그대로라 invalidation 없음.

### P1-F. 단계적 실행 (Task 7 분할)

v1 Task 7 Step 2는 6개 레시피를 `&&` 체인으로 실행. **#13 단독 → report 1차 확인 → #14-#18 batch**로 2단계 분할. 첫 실행에서 ORT dispatch 경로의 silent bug (symmetric 누락 등)를 잡아내기 위함.

Task 7 Step 2를 다음으로 대체:

```bash
# Step 2a: smoke single
python scripts/run_trt.py --recipe recipes/13_ort_int8_minmax.yaml --out results/13_ort_int8_minmax.json
python scripts/recommend.py --results-dir results --out report.md --exclude "trt_int8_sparsity,modelopt_int8_sparsity,inc_int8_qat"
cat report.md | head -40
# Inspect: mAP ~0.55 (±0.02), fps_bs1 >150, meets_constraints=true if drop<2%p.
# If sanity-fails, diagnose here before batching.

# Step 2b: batch remaining 5
python scripts/run_trt.py --recipe recipes/14_ort_int8_entropy.yaml --out results/14_ort_int8_entropy.json \
 && python scripts/run_trt.py --recipe recipes/15_ort_int8_percentile.yaml --out results/15_ort_int8_percentile.json \
 && python scripts/run_trt.py --recipe recipes/16_ort_int8_distribution.yaml --out results/16_ort_int8_distribution.json \
 && python scripts/run_trt.py --recipe recipes/17_inc_int8_ptq.yaml --out results/17_inc_int8_ptq.json \
 && python scripts/run_trt.py --recipe recipes/18_inc_int8_smoothquant.yaml --out results/18_inc_int8_smoothquant.json
```

### P2-G. (선택) Distribution 가용성 — 사전 검증 완료

`python -c "from onnxruntime.quantization import CalibrationMethod as CM; print([x.name for x in CM])"` 로컬 실행 결과 `['MinMax', 'Entropy', 'Percentile', 'Distribution']` 확인 (2026-04-18). #16 유지.

### P1 비적용 (fact-check로 해소)

- v1 Review에서 걱정했던 `_build_engine`의 INT8 flag 세팅이 source별 분기되는지 문제: 실제 코드(`run_trt.py:489-499`)는 `dtype == "int8"`를 기준으로 분기하고 `quant_preapplied` 플래그로 QDQ 소스를 공통 처리. **별도 변경 불필요** — ORT/INC 모두 modelopt와 동일 경로를 탐.

---

## File Structure

| 파일 | 변경 종류 | 책임 |
|---|---|---|
| `scripts/_schemas.py` | Modify (L30) | `TechniqueSpec.source` Literal에 2개 추가 |
| `scripts/run_trt.py` | Modify | `_prepare_ort_quant_onnx`, `_prepare_inc_onnx` 신설 + `_prepare_onnx` 디스패치 분기 + `run()` source_suffix 처리 |
| `recipes/13_ort_int8_minmax.yaml` | Create | ORT MinMax 캘리브 (기본값) |
| `recipes/14_ort_int8_entropy.yaml` | Create | ORT Entropy 캘리브 |
| `recipes/15_ort_int8_percentile.yaml` | Create | ORT Percentile 캘리브 |
| `recipes/16_ort_int8_distribution.yaml` | Create | ORT Distribution 캘리브 |
| `recipes/17_inc_int8_ptq.yaml` | Create | INC 기본 PTQ (MinMax) |
| `recipes/18_inc_int8_smoothquant.yaml` | Create | INC SmoothQuant — **본 실험** |
| `recipes/19_inc_int8_qat.yaml` | Create (parked) | INC QAT 스펙만 작성, `make all`에서 제외 |
| `Makefile` | Modify | `recipe-13`..`recipe-19` + `make all` 편입 (19는 제외) + `PARKED` 갱신 |
| `pyproject.toml` | Modify | `[project.optional-dependencies]`에 `ort_quant`, `inc` 그룹 추가 |
| `README.md` | Modify | 레시피 표 확장 + `technique.source` 디스패처 섹션 갱신 |
| `docs/improvements/2026-04-18-trt-modelopt-audit.md` | Modify | "Wave 3 results" 섹션 추가 (측정 후) |
| `tests/test_wave3_dispatch.py` | Create | 신규 `source` 두 값의 스키마 수용 + 디스패치 라우팅 단위 테스트 |

---

## Task 1: Schema 확장 — `source` Literal 갱신

**Files:**
- Modify: `scripts/_schemas.py:30`
- Test: `tests/test_wave3_dispatch.py` (new)

- [ ] **Step 1: 실패 테스트 작성 (`tests/test_wave3_dispatch.py`)**

```python
import pytest
from scripts._schemas import TechniqueSpec


@pytest.mark.parametrize("src", ["ort_quant", "neural_compressor"])
def test_source_literal_accepts_new_backends(src):
    spec = TechniqueSpec(name="int8_ptq", source=src)
    assert spec.source == src
```

- [ ] **Step 2: 실행 — 실패 확인**

```bash
python -m pytest tests/test_wave3_dispatch.py::test_source_literal_accepts_new_backends -v
```
Expected: `neural_compressor` 파라미터가 ValidationError (현재 Literal에 없음).

- [ ] **Step 3: `_schemas.py:30` 수정**

```python
source: Literal["trt_builtin", "modelopt", "ort_quant", "neural_compressor"] = "trt_builtin"
```

- [ ] **Step 4: 테스트 재실행 — 통과 확인**

```bash
python -m pytest tests/test_wave3_dispatch.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/_schemas.py tests/test_wave3_dispatch.py
git commit -m "feat(schema): accept ort_quant + neural_compressor as technique.source"
```

---

## Task 2: ORT Quantization 준비 함수

**Files:**
- Modify: `scripts/run_trt.py` (insert new function after `_prepare_modelopt_onnx`)
- Test: `tests/test_wave3_dispatch.py`

- [ ] **Step 1: 실패 테스트 추가** (디스패치 라우팅만 확인 — 실제 ORT 실행은 smoke 수준)

```python
from pathlib import Path
from unittest.mock import patch
from scripts._schemas import Recipe
from scripts.run_trt import _prepare_onnx


def test_prepare_onnx_routes_to_ort_quant(tmp_path, monkeypatch):
    recipe_yaml = tmp_path / "13.yaml"
    recipe_yaml.write_text(
        "name: ort_int8_minmax\n"
        "model: {family: yolo26, variant: n, weights: yolo26n.pt}\n"
        "runtime: {engine: tensorrt, dtype: int8}\n"
        "technique: {name: int8_ptq, source: ort_quant, calibrator: minmax,\n"
        "            calibration_samples: 8, calibration_seed: 42}\n"
        "measurement:\n"
        "  dataset: coco_val2017\n  num_images: 8\n"
        "  warmup_iters: 1\n  measure_iters: 2\n"
        "  batch_sizes: [1]\n  input_size: 640\n"
        "  gpu_clock_lock: false\n  seed: 42\n"
        "constraints: {max_map_drop_pct: 5.0}\n",
        encoding="utf-8",
    )
    from scripts._schemas import load_recipe
    recipe = load_recipe(str(recipe_yaml))
    with patch("scripts.run_trt._prepare_ort_quant_onnx") as m:
        m.return_value = tmp_path / "fake.onnx"
        path, is_qdq = _prepare_onnx(recipe, 640, tmp_path, bs=1)
    assert is_qdq is True
    m.assert_called_once()
```

- [ ] **Step 2: `_prepare_ort_quant_onnx` 구현 (`scripts/run_trt.py`, `_prepare_modelopt_onnx` 바로 뒤)**

```python
def _prepare_ort_quant_onnx(recipe: Recipe, imgsz: int, cache_dir: Path,
                            dynamic: bool = True) -> Path:
    """Quantize via onnxruntime.quantization.quantize_static.

    Produces QDQ ONNX that TRT's explicit quantization path consumes — the same
    contract as modelopt. Calibration method maps 1:1 from recipe.technique.calibrator:
      - "minmax"       -> CalibrationMethod.MinMax
      - "entropy"      -> CalibrationMethod.Entropy
      - "percentile"   -> CalibrationMethod.Percentile
      - "distribution" -> CalibrationMethod.Distribution
    Per-channel weights + symmetric activations/weights are TRT-friendly defaults.
    """
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime.quantization not available. Install onnxruntime>=1.17."
        ) from e

    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "distribution": CalibrationMethod.Distribution,
    }
    calibrator = (recipe.technique.calibrator or "minmax").lower()
    if calibrator not in method_map:
        raise ValueError(
            f"ort_quant only supports {list(method_map)}, got {calibrator!r}"
        )

    n_samples = int(recipe.technique.calibration_samples or 512)
    seed = int(recipe.technique.calibration_seed or 42)
    bs_tag = "dyn" if dynamic else "bs1"
    cached = cache_dir / (
        f"{Path(recipe.model.weights).stem}_{imgsz}_ort_{calibrator}_"
        f"{n_samples}_s{seed}_{bs_tag}.qdq.onnx"
    )
    if cached.exists():
        print(f"[info] ort_quant cache hit: {cached.name}", file=sys.stderr)
        return cached

    clean_onnx = _export_onnx(recipe.model.weights, imgsz, half=False,
                              cache_dir=cache_dir, dynamic=dynamic)
    val_yaml = os.environ.get("OMNI_COCO_YAML")
    calib_arr = _build_calib_numpy(val_yaml, n_samples, imgsz, seed)

    class _NumpyReader(CalibrationDataReader):
        def __init__(self, arr, input_name: str):
            self._iter = iter(arr[:, None, ...] if arr.ndim == 3 else arr)
            self._name = input_name

        def get_next(self):
            try:
                x = next(self._iter)
            except StopIteration:
                return None
            if x.ndim == 3:
                x = x[None, ...]
            return {self._name: x}

    import onnx
    model = onnx.load(str(clean_onnx))
    input_name = model.graph.input[0].name
    del model

    nodes_to_exclude = list(recipe.technique.nodes_to_exclude or [])
    print(
        f"[info] ort_quant: method={calibrator}, samples={n_samples}, "
        f"excludes={len(nodes_to_exclude)}, onnx={clean_onnx.name}",
        file=sys.stderr,
    )
    quantize_static(
        model_input=str(clean_onnx),
        model_output=str(cached),
        calibration_data_reader=_NumpyReader(calib_arr, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=method_map[calibrator],
        nodes_to_exclude=nodes_to_exclude or None,
    )
    print(f"[info] ort_quant wrote QDQ onnx: {cached}", file=sys.stderr)
    return cached
```

- [ ] **Step 3: `_prepare_onnx` 디스패처에 분기 추가 (run_trt.py:271)**

기존 `if source == "ort_quant": raise NotImplementedError(...)` 줄을 교체:

```python
    if source == "ort_quant":
        return _prepare_ort_quant_onnx(recipe, imgsz, cache_dir, dynamic=dynamic), True
```

- [ ] **Step 4: Smoke 테스트 — 디스패치 루팅 동작 확인**

```bash
python -m pytest tests/test_wave3_dispatch.py::test_prepare_onnx_routes_to_ort_quant -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_trt.py tests/test_wave3_dispatch.py
git commit -m "feat(trt): ort_quant dispatch via onnxruntime.quantization.quantize_static"
```

---

## Task 3: ORT 레시피 4장 (#13 ~ #16)

**Files:**
- Create: `recipes/13_ort_int8_minmax.yaml`
- Create: `recipes/14_ort_int8_entropy.yaml`
- Create: `recipes/15_ort_int8_percentile.yaml`
- Create: `recipes/16_ort_int8_distribution.yaml`

- [ ] **Step 1: `13_ort_int8_minmax.yaml` 작성**

```yaml
name: ort_int8_minmax
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
runtime:
  engine: tensorrt
  dtype: int8
technique:
  name: int8_ptq
  source: ort_quant
  calibrator: minmax
  calibration_samples: 512
  calibration_seed: 42
measurement:
  dataset: coco_val2017
  num_images: 5000
  warmup_iters: 100
  measure_iters: 1000
  batch_sizes: [1, 8]
  input_size: 640
  gpu_clock_lock: true
  seed: 42
constraints:
  max_map_drop_pct: 2.0
  min_fps_bs1: 30
```

- [ ] **Step 2: `14_ort_int8_entropy.yaml` 작성**

`13`을 복제하고 `name: ort_int8_entropy`, `calibrator: entropy`로 변경.

- [ ] **Step 3: `15_ort_int8_percentile.yaml` 작성**

`13`을 복제하고 `name: ort_int8_percentile`, `calibrator: percentile`로 변경.

- [ ] **Step 4: `16_ort_int8_distribution.yaml` 작성**

`13`을 복제하고 `name: ort_int8_distribution`, `calibrator: distribution`로 변경.

- [ ] **Step 5: YAML 로더 smoke 검증**

```bash
python -c "import sys; sys.path.insert(0, '.'); from scripts._schemas import load_recipe; import pathlib; [load_recipe(str(p)) for p in pathlib.Path('recipes').glob('1[3-6]_*.yaml')]"
```
Expected: exit code 0.

- [ ] **Step 6: Commit**

```bash
git add recipes/13_*.yaml recipes/14_*.yaml recipes/15_*.yaml recipes/16_*.yaml
git commit -m "feat(recipes): ORT INT8 PTQ recipes #13-#16 (minmax/entropy/percentile/distribution)"
```

---

## Task 4: Neural Compressor 준비 함수

**Files:**
- Modify: `scripts/run_trt.py`
- Test: `tests/test_wave3_dispatch.py`

- [ ] **Step 1: 실패 테스트 추가**

```python
def test_prepare_onnx_routes_to_neural_compressor(tmp_path):
    recipe_yaml = tmp_path / "17.yaml"
    recipe_yaml.write_text(
        "name: inc_int8_ptq\n"
        "model: {family: yolo26, variant: n, weights: yolo26n.pt}\n"
        "runtime: {engine: tensorrt, dtype: int8}\n"
        "technique: {name: int8_ptq, source: neural_compressor,\n"
        "            calibrator: minmax, calibration_samples: 8}\n"
        "measurement:\n"
        "  dataset: coco_val2017\n  num_images: 8\n"
        "  warmup_iters: 1\n  measure_iters: 2\n"
        "  batch_sizes: [1]\n  input_size: 640\n"
        "  gpu_clock_lock: false\n  seed: 42\n"
        "constraints: {max_map_drop_pct: 5.0}\n",
        encoding="utf-8",
    )
    from scripts._schemas import load_recipe
    from unittest.mock import patch
    from scripts.run_trt import _prepare_onnx
    recipe = load_recipe(str(recipe_yaml))
    with patch("scripts.run_trt._prepare_inc_onnx") as m:
        m.return_value = tmp_path / "fake.onnx"
        path, is_qdq = _prepare_onnx(recipe, 640, tmp_path, bs=1)
    assert is_qdq is True
    m.assert_called_once()
```

- [ ] **Step 2: `_prepare_inc_onnx` 구현 (`_prepare_ort_quant_onnx` 바로 뒤)**

```python
def _prepare_inc_onnx(recipe: Recipe, imgsz: int, cache_dir: Path,
                      dynamic: bool = True) -> Path:
    """Quantize via neural_compressor.quantization.fit (PTQ static).

    Supports two algorithms keyed off recipe.technique.calibrator:
      - "minmax" (default) -> standard static PTQ, MinMax-based
      - "smoothquant"      -> SmoothQuant preprocessing (migrates activation
                              outliers into weight scales). Key experiment
                              for YOLO necks where activation ranges blow up.
    """
    try:
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor.quantization import fit
        from neural_compressor.data import DataLoader, Datasets
    except ImportError as e:
        raise RuntimeError(
            "neural-compressor not installed. Install with: pip install neural-compressor"
        ) from e

    calibrator = (recipe.technique.calibrator or "minmax").lower()
    if calibrator not in ("minmax", "smoothquant"):
        raise ValueError(
            f"neural_compressor backend supports calibrator in "
            f"['minmax','smoothquant'], got {calibrator!r}"
        )

    n_samples = int(recipe.technique.calibration_samples or 512)
    seed = int(recipe.technique.calibration_seed or 42)
    bs_tag = "dyn" if dynamic else "bs1"
    cached = cache_dir / (
        f"{Path(recipe.model.weights).stem}_{imgsz}_inc_{calibrator}_"
        f"{n_samples}_s{seed}_{bs_tag}.qdq.onnx"
    )
    if cached.exists():
        print(f"[info] neural_compressor cache hit: {cached.name}", file=sys.stderr)
        return cached

    clean_onnx = _export_onnx(recipe.model.weights, imgsz, half=False,
                              cache_dir=cache_dir, dynamic=dynamic)
    val_yaml = os.environ.get("OMNI_COCO_YAML")
    calib_arr = _build_calib_numpy(val_yaml, n_samples, imgsz, seed)

    import onnx
    model = onnx.load(str(clean_onnx))
    input_name = model.graph.input[0].name
    del model

    class _CalibDataset:
        def __init__(self, arr, name):
            self._arr = arr
            self._name = name

        def __len__(self):
            return len(self._arr)

        def __getitem__(self, idx):
            x = self._arr[idx]
            if x.ndim == 3:
                x = x[None, ...]
            return {self._name: x}, 0  # (inputs, label-placeholder)

    dataset = _CalibDataset(calib_arr, input_name)
    calib_loader = DataLoader(framework="onnxruntime", dataset=dataset, batch_size=1)

    quant_config_kwargs = dict(approach="static", backend="default")
    if calibrator == "smoothquant":
        quant_config_kwargs["recipes"] = {
            "smooth_quant": True,
            "smooth_quant_args": {"alpha": 0.5},
        }

    conf = PostTrainingQuantConfig(**quant_config_kwargs)
    print(
        f"[info] neural_compressor.fit: method={calibrator}, samples={n_samples}, "
        f"onnx={clean_onnx.name}",
        file=sys.stderr,
    )
    q_model = fit(model=str(clean_onnx), conf=conf, calib_dataloader=calib_loader)
    q_model.save(str(cached))
    print(f"[info] neural_compressor wrote QDQ onnx: {cached}", file=sys.stderr)
    return cached
```

- [ ] **Step 3: 디스패처 분기 추가 (`_prepare_onnx` 내)**

```python
    if source == "neural_compressor":
        return _prepare_inc_onnx(recipe, imgsz, cache_dir, dynamic=dynamic), True
```

- [ ] **Step 4: Smoke 테스트 실행**

```bash
python -m pytest tests/test_wave3_dispatch.py::test_prepare_onnx_routes_to_neural_compressor -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_trt.py tests/test_wave3_dispatch.py
git commit -m "feat(trt): neural_compressor dispatch (PTQ + SmoothQuant)"
```

---

## Task 5: INC 레시피 3장 (#17, #18, #19-parked)

**Files:**
- Create: `recipes/17_inc_int8_ptq.yaml`
- Create: `recipes/18_inc_int8_smoothquant.yaml`
- Create: `recipes/19_inc_int8_qat.yaml` (**parked**)

- [ ] **Step 1: `17_inc_int8_ptq.yaml` 작성**

`13_ort_int8_minmax.yaml`을 복제하고:
- `name: inc_int8_ptq`
- `technique.source: neural_compressor`
- `technique.calibrator: minmax`

- [ ] **Step 2: `18_inc_int8_smoothquant.yaml` 작성**

`17`을 복제하고 `name: inc_int8_smoothquant`, `calibrator: smoothquant`로 변경. 코멘트로 "SmoothQuant: activation outlier → weight scale migration, expected to help YOLO neck layers" 추가.

- [ ] **Step 3: `19_inc_int8_qat.yaml` 작성 (parked — 실행 안 함)**

스펙만 작성. 파일 상단에 코멘트:

```yaml
# PARKED — INC QAT는 학습 코드(loss, optimizer, epochs)가 본 저장소에 아직 없으므로
# `make all`에서 제외. 레시피 스키마/runner 디스패치는 유지해서 training pipeline
# 도입 시점에 `make recipe-19`로 단독 실행 가능하게 한다. #7(trt_int8_sparsity),
# #11(modelopt_int8_sparsity)과 동일 취급.
name: inc_int8_qat
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
runtime:
  engine: tensorrt
  dtype: int8
technique:
  name: int8_qat
  source: neural_compressor
  calibrator: minmax  # QAT는 캘리브 대신 학습 중 스케일 학습 — placeholder
  calibration_samples: 0
  qat_epochs: 3
  qat_lr: 1.0e-5
measurement:
  dataset: coco_val2017
  num_images: 5000
  warmup_iters: 100
  measure_iters: 1000
  batch_sizes: [1, 8]
  input_size: 640
  gpu_clock_lock: true
  seed: 42
constraints:
  max_map_drop_pct: 0.5  # QAT는 PTQ보다 타이트하게 — 달성 못 하면 이 경로 무의미
  min_fps_bs1: 30
```

Note: `qat_epochs` / `qat_lr`는 현재 `TechniqueSpec`에 필드 없음. 이 레시피는 스키마 로드 시 필드 무시되는지(pydantic `extra="ignore"`인지) 확인하고, `extra="forbid"`라면 `TechniqueSpec`에 `Optional[int] qat_epochs = None` / `Optional[float] qat_lr = None`을 추가한다. QAT 구현은 미래 작업.

- [ ] **Step 4: YAML 로더 smoke 검증**

```bash
python -c "import sys; sys.path.insert(0, '.'); from scripts._schemas import load_recipe; import pathlib; [load_recipe(str(p)) for p in pathlib.Path('recipes').glob('1[7-9]_*.yaml')]"
```

- [ ] **Step 5: Commit**

```bash
git add recipes/17_*.yaml recipes/18_*.yaml recipes/19_*.yaml scripts/_schemas.py
git commit -m "feat(recipes): INC INT8 PTQ + SmoothQuant (#17,#18) + QAT parked (#19)"
```

---

## Task 6: Makefile + pyproject extras + README

**Files:**
- Modify: `Makefile`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: `Makefile` — 신규 타겟 추가**

기존 `.PHONY` 줄 확장:

```make
.PHONY: all clean env report \
        recipe-00 recipe-00-tf32 \
        recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 recipe-07 \
        recipe-08 recipe-09 recipe-10 recipe-11 recipe-12 \
        recipe-13 recipe-14 recipe-15 recipe-16 \
        recipe-17 recipe-18 recipe-19 \
        diagnose-recipe-%
```

`all:` 확장 (19는 제외):

```make
all: recipe-00 recipe-00-tf32 \
     recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 \
     recipe-08 recipe-09 recipe-10 recipe-12 \
     recipe-13 recipe-14 recipe-15 recipe-16 \
     recipe-17 recipe-18 report
```

`recipe-06` 아래에 신규 타겟 블록:

```make
recipe-13:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/13_ort_int8_minmax.yaml --out $(RESULTS_DIR)/13_ort_int8_minmax.json

recipe-14:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/14_ort_int8_entropy.yaml --out $(RESULTS_DIR)/14_ort_int8_entropy.json

recipe-15:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/15_ort_int8_percentile.yaml --out $(RESULTS_DIR)/15_ort_int8_percentile.json

recipe-16:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/16_ort_int8_distribution.yaml --out $(RESULTS_DIR)/16_ort_int8_distribution.json

recipe-17:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/17_inc_int8_ptq.yaml --out $(RESULTS_DIR)/17_inc_int8_ptq.json

recipe-18:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/18_inc_int8_smoothquant.yaml --out $(RESULTS_DIR)/18_inc_int8_smoothquant.json

recipe-19:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/19_inc_int8_qat.yaml --out $(RESULTS_DIR)/19_inc_int8_qat.json
```

`PARKED` 확장:

```make
PARKED := trt_int8_sparsity,modelopt_int8_sparsity,inc_int8_qat
```

- [ ] **Step 2: `pyproject.toml` — extras 추가**

`[project.optional-dependencies]` 섹션에:

```toml
ort_quant = ["onnxruntime>=1.17"]
inc = ["neural-compressor>=3.0"]
```

그리고 `all`에 두 이름 추가:

```toml
all = [..., "onnxruntime>=1.17", "neural-compressor>=3.0"]
```

- [ ] **Step 3: `README.md` — 레시피 표 확장**

기존 표 아래에 7개 행 추가:

```markdown
| 13 | TensorRT                   | INT8 PTQ (ort_quant, minmax)                     | `ort_quant`  |
| 14 | TensorRT                   | INT8 PTQ (ort_quant, entropy)                    | `ort_quant`  |
| 15 | TensorRT                   | INT8 PTQ (ort_quant, percentile)                 | `ort_quant`  |
| 16 | TensorRT                   | INT8 PTQ (ort_quant, distribution)               | `ort_quant`  |
| 17 | TensorRT                   | INT8 PTQ (neural_compressor, minmax)             | `neural_compressor` |
| 18 | TensorRT                   | INT8 PTQ + SmoothQuant (neural_compressor)       | `neural_compressor` |
| 19 | TensorRT                   | INT8 QAT (neural_compressor) — **parked**        | `neural_compressor` |
```

`technique.source` 디스패처 섹션도 `ort_quant`와 `neural_compressor`를 구현됨으로 갱신.

- [ ] **Step 4: YAML 로더 smoke 전체 검증**

```bash
python -c "import sys; sys.path.insert(0, '.'); from scripts._schemas import load_recipe; import pathlib; [load_recipe(str(p)) for p in pathlib.Path('recipes').glob('*.yaml')]"
```

- [ ] **Step 5: Commit**

```bash
git add Makefile pyproject.toml README.md
git commit -m "build(wave3): Makefile + extras + README for ort_quant and neural_compressor"
```

---

## Task 7: 측정 실행 + 리포트 재생성 (#13–#18, #19 제외)

**Files:**
- Create: `results/13_ort_int8_minmax.json` … `results/18_inc_int8_smoothquant.json`
- Modify: `report.md`

- [ ] **Step 1: 캘리브 데이터셋 준비 확인**

```bash
echo "$OMNI_COCO_YAML"
```
Expected: `coco_val_only.yaml` (또는 val2017 yaml) 경로가 출력되어야 함. 비어있으면 Phase 2 문서의 경로를 참고해 설정.

- [ ] **Step 2: 신규 6개 레시피 순차 실행**

```bash
python scripts/run_trt.py --recipe recipes/13_ort_int8_minmax.yaml --out results/13_ort_int8_minmax.json \
 && python scripts/run_trt.py --recipe recipes/14_ort_int8_entropy.yaml --out results/14_ort_int8_entropy.json \
 && python scripts/run_trt.py --recipe recipes/15_ort_int8_percentile.yaml --out results/15_ort_int8_percentile.json \
 && python scripts/run_trt.py --recipe recipes/16_ort_int8_distribution.yaml --out results/16_ort_int8_distribution.json \
 && python scripts/run_trt.py --recipe recipes/17_inc_int8_ptq.yaml --out results/17_inc_int8_ptq.json \
 && python scripts/run_trt.py --recipe recipes/18_inc_int8_smoothquant.yaml --out results/18_inc_int8_smoothquant.json
```

각 runner가 `meets_constraints=False`로 기록해도 `make all` 정책상 abort하지 않음. 실패 케이스는 `results/*.json::notes`로 기록되어 report에 노출됨.

- [ ] **Step 3: 리포트 재생성**

```bash
python scripts/recommend.py --results-dir results --out report.md --exclude "trt_int8_sparsity,modelopt_int8_sparsity,inc_int8_qat"
```

- [ ] **Step 4: 측정 JSON + 리포트 커밋**

```bash
git add results/13_*.json results/14_*.json results/15_*.json results/16_*.json \
        results/17_*.json results/18_*.json report.md
git commit -m "bench(wave3): ORT + INC INT8 measurements + report refresh"
```

---

## Task 8: Audit 문서 업데이트 — Wave 3 결과 기록

**Files:**
- Modify: `docs/improvements/2026-04-18-trt-modelopt-audit.md`

- [ ] **Step 1: 문서 하단에 신규 섹션 추가**

```markdown
## Wave 3 results — ORT Quantization + Neural Compressor (2026-04-XX)

### 실험 목적
modelopt 백엔드(`#8–#10`)의 mAP drop -1.6~1.9%p를 기준으로, 대체/보완 백엔드가
동등 이상의 정확도를 낼 수 있는지 검증.

### 측정치 (fill in after Task 7)

| 레시피 | calibrator | mAP@0.5 | drop %p | fps(bs1) | vs modelopt best |
|---|---|---|---|---|---|
| 13 ort_int8_minmax | MinMax | ... | ... | ... | ... |
| 14 ort_int8_entropy | Entropy | ... | ... | ... | ... |
| 15 ort_int8_percentile | Percentile | ... | ... | ... | ... |
| 16 ort_int8_distribution | Distribution | ... | ... | ... | ... |
| 17 inc_int8_ptq | MinMax | ... | ... | ... | ... |
| 18 inc_int8_smoothquant | SmoothQuant | ... | ... | ... | ... |

### 해석 포인트
- ORT Percentile vs modelopt Percentile: 동일 알고리즘 계열 — 스케일 차이는
  구현 디테일(퍼센타일 값, 대칭 여부)에서 온다.
- INC SmoothQuant: activation outlier가 YOLO neck (`/model.*Concat*/Conv`)에서
  주범이라면 여기서 mAP가 눈에 띄게 올라갈 여지.
- 모두 modelopt Entropy(-1.6%p)보다 나쁘면: modelopt는 유지하고 신규 백엔드는
  "동등 실패"로 문서만 남긴다. 더 나으면 report.md 기본 추천 후보에 반영.
```

- [ ] **Step 2: Commit**

```bash
git add docs/improvements/2026-04-18-trt-modelopt-audit.md
git commit -m "docs(audit): record Wave 3 ORT + INC measurement results"
```

---

## Self-Review 체크리스트 (구현 완료 직전)

- [ ] `_schemas.py`의 `source` Literal에 `ort_quant`와 `neural_compressor` 둘 다 있는가?
- [ ] `_prepare_onnx` 디스패처가 4개 분기(`trt_builtin` / `modelopt` / `ort_quant` / `neural_compressor`)를 모두 처리하는가?
- [ ] 신규 7개 YAML이 `load_recipe`로 파싱되는가?
- [ ] `run()` 내 `source_suffix` 처리가 `trt_builtin` 이외의 3개 모두에서 엔진 캐시 파일명에 반영되는가? (`_ort_quant_bs1_...engine`, `_neural_compressor_bs1_...engine`)
- [ ] `make all`이 #19를 제외하는가? `recommend.py --exclude`에 `inc_int8_qat`가 포함되는가?
- [ ] `tests/test_wave3_dispatch.py`의 모든 테스트가 통과하는가?
- [ ] README의 표와 `technique.source` 섹션이 일치하는가?
- [ ] pyproject `all` extras에 신규 2개 패키지가 들어가 있는가?

---

## Known risks / future work

1. **INC API 변경 리스크**: `PostTrainingQuantConfig.recipes["smooth_quant"]` 인자명은 INC 2.x/3.x에서 바뀐 이력 있음. 구현 전에 `pip show neural-compressor` 버전 확인하고 해당 버전 문서로 교차검증.
2. **ORT Distribution 성능**: 상대적으로 새 알고리즘 — YOLO 같은 detection 모델에서 실전 데이터 적음. 실패해도 예상된 리스크로 남기고 `meets_constraints=False`로 기록.
3. **QAT 학습 파이프라인 (#19)**: Ultralytics trainer + INC `QuantAwareTrainingConfig` 통합은 `#7/#11` sparsity-aware training과 묶어 별도 epic. Phase 4 후보.
4. **Olive 미포함**: Wave 3에서 의도적으로 제외 (OmniOptimizer 자체와 scope 중복). 필요 시 Phase 5.
