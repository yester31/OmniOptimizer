# Wave 13 archive — ModelOpt ONNX pass API 불일치 발견

**Date**: 2026-04-21
**Status**: Wave 13 plan (draft) ARCHIVED without execution.
**Decision**: Wave 9 DirectML EP 로 이동. modelopt.onnx.autocast/autotune은 향후 modelopt이 CNN 공식 지원 발표 시 재평가.

## TL;DR

`/gsd-plan-phase` 교차 검증이 plan의 **전제 API 절반이 허구**임을 확인. `modelopt.onnx.autotune` 모듈은 **존재하지 않으며**, 실제 경로(`modelopt.onnx.quantization.autotune`)는 목적이 완전히 다른 INT8 Q/DQ placement 도구. `modelopt.onnx.autocast.convert_to_f16` 도 시그니처가 plan 호출과 불일치(`ModelProto` 기대, `sensitivity_samples` 파라미터 부재). 실행 시 즉시 `ModuleNotFoundError` / `TypeError`. Plan 전면 재설계 비용이 "저위험 filler" 가치 제안을 무너뜨려 Wave 13 포기.

## 발견 1 — `modelopt.onnx.autotune` 모듈 부재 (Recipe #28)

Plan 전제:
```python
from modelopt.onnx.autotune import autotune
best_engine = autotune(onnx_path, num_trials=8, objective="min_latency",
                       allow_fp16=True, allow_int8=False)
```

실제 `modelopt/onnx/__init__.py` exports:
```python
from modelopt.onnx import quantization, configure_logging
```

`autotune/` 서브디렉토리는 **없음**. 출처: <https://github.com/NVIDIA/Model-Optimizer/tree/main/modelopt/onnx>

### 진짜 `modelopt.onnx.quantization.autotune`의 정체

```python
# modelopt/onnx/quantization/autotune/__init__.py (실제)
from modelopt.onnx.quantization.autotune.qdq_autotune import (
    MODE_PRESETS, StoreWithExplicitFlag, get_node_filter_list,
)
# tensorrt 설치 시 optional:
# QDQAutotuner, TrtExecBenchmark, RegionPattern, CombinedRegionSearch
```

Docstring: *"automated optimization of Quantize/Dequantize (Q/DQ) node placement in ONNX computation graphs"*

**용도**: INT8 QDQ 노드 배치 최적화 전용. 즉 기존 modelopt INT8 recipe (#09 entropy 등) 대비 QDQ placement를 더 잘 하자는 **INT8 전용 도구**. Plan이 상상한 "FP16 builder param sweep"와 목적 자체가 다름.

**호출 방식**: `autotune()` 단일 함수가 아니라 `QDQAutotuner` 인스턴스 + `TrtExecBenchmark` 조립 + region pattern 설정의 **저수준 클래스 API**. Plan의 `num_trials`, `objective`, `allow_fp16`, `allow_int8` 파라미터는 **어디에도 없음**.

→ Recipe #28은 plan의 모든 전제가 틀림. 수리 불가.

## 발견 2 — `convert_to_f16` 시그니처 불일치 (Recipe #27)

Plan 전제:
```python
from modelopt.onnx.autocast import convert_to_f16
onnx_path_fp16 = convert_to_f16(onnx_path_fp32,
                                 sensitivity_samples=32,
                                 op_block_list=["Softmax", "LayerNormalization"])
```

실제 `modelopt/onnx/autocast/__init__.py`:
```python
from modelopt.onnx.autocast.convert import convert_to_f16, convert_to_mixed_precision
```

| Plan 가정 | 실제 |
|---|---|
| 첫 인자 `onnx_path_fp32` (str) | **`model: onnx.ModelProto` 객체** — `TypeError` |
| `sensitivity_samples=32` | 존재하지 않음 (양쪽 함수 모두) |
| `op_block_list=[...]` | `convert_to_f16`에만 있음. `convert_to_mixed_precision`은 `op_types_to_exclude` |
| "sensitivity-aware" 동작 | `convert_to_f16` 문서: *"bypasses NodeClassifier, and uses a simple op_block_list"* — sensitivity-aware 경로 **아님** |
| `calibration_data` | 파일 경로(npz/npy). plan은 "QR val 32장" 이라 했지만 npz 직렬화 단계 누락 |

출처: <https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/onnx/autocast/__init__.py>

## 발견 3 — Recipe YAML ↔ ONNX 에셋 모순

Plan은 "base ONNX = `best_qr_640_fp32_dyn.onnx` (Wave 6 asset 재사용)" 이라고 서술. 그러나 recipe YAML은:

```yaml
model: {family: yolo26, variant: n, weights: yolo26n.pt}   # ← COCO 가중치
```

실제 `best_qr` prefix ONNX는 `OMNI_WEIGHTS_OVERRIDE=best_qr.pt` 환경변수 세팅 시에만 생성됨 (`scripts/_weights_io.py:91`). Plan 어디에도 `OMNI_WEIGHTS_OVERRIDE` 컨벤션 명시 없음. Recipe가 그대로 실행되면 `yolo26n_640_fp32_dyn.onnx`가 cache에 저장됨. **기존 Wave 6 QR asset 재사용 주장이 성립 안 함**.

## 발견 4 — 디테일 오염 (LLM 예제 복붙 흔적)

- `op_block_list: ["Softmax", "LayerNormalization"]` — YOLO는 **BatchNormalization** 기반, LayerNormalization 없음. Transformer 예제 그대로 복사.
- `sensitivity_samples: 32` — Wave 5가 이미 "QR val 133장 부족 → COCO 512장 전환" 결정(`docs/qr_barcode_eval_v2.md:70`)을 내림. 32는 그 결정에 역행. 게다가 API에 파라미터 부재라 실행돼도 silent ignore.

## 실행 안 한 대신 얻은 것

본 archive 문서로:
1. 향후 modelopt ONNX pass 재검토 시 **같은 함정 반복 회피** — GitHub `modelopt/onnx/__init__.py` 직접 읽기 우선
2. API 검증 프로세스 강화 근거 — 웹 검색 + ChatGPT 답변만으로 plan 쓰지 말고 **repo 소스 `__init__.py` 직접 확인** 컨벤션 수립 필요

## 후속 방향

**Wave 9 DirectML EP** 로 이동. 이유:
- ORT 네이티브 경로(`onnxruntime-directml` wheel). 외부 변환기/허구 API 리스크 없음
- Windows NPU / Intel Arc GPU / AMD GPU 커버 범위 확장
- Wave 7/8 "외부 변환기 호환성" + Wave 13 "허구 API" 리스크 클래스 모두 회피

Recipe 번호 #27-#28 은 DirectML에 재할당 가능. #29는 buffer.

## 교훈 (CLAUDE.md 후보)

- 외부 라이브러리 최신 API는 **GitHub 소스의 `__init__.py` 읽기가 grounding** — 공식 문서 페이지 / ChatGPT / 웹 블로그 단독 근거로 plan 쓰지 말 것
- Plan 품질 게이트로 `/gsd-plan-phase` cross-verify가 **BLOCKER 발견에 결정적** — plan 작성자 본인은 자신의 가정을 못 깬다. 독립 에이전트 검증이 ROI 높음
- "low-risk filler" 라는 포지셔닝 자체가 경계 신호 — filler라고 가볍게 plan을 쓰면 API 검증도 가볍게 하게 됨

## 참고 링크

- [NVIDIA/Model-Optimizer — onnx package tree](https://github.com/NVIDIA/Model-Optimizer/tree/main/modelopt/onnx)
- [NVIDIA/Model-Optimizer — autocast __init__.py](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/onnx/autocast/__init__.py)
- [NVIDIA/Model-Optimizer — quantization.autotune __init__.py](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/onnx/quantization/autotune/__init__.py)
- [AutoCast (ONNX) guide](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/8_autocast.html)
- [DeepWiki — ONNX AutoQDQ and Autotuning](https://deepwiki.com/NVIDIA/Model-Optimizer/4.4-onnx-autoqdq-and-autotuning)
