# INT4 Weight-only on Ampere × CNN: not feasible (Wave 12 분석)

**Date**: 2026-04-21
**Status**: Wave 12 plan (draft) ARCHIVED without execution. 본 문서가 그 자리를 대체.
**Decision**: YOLO26n × RTX 3060 Laptop (Ampere SM 8.6) × INT4 weight-only 조합은 **실행 가치 없음**. FP8 (Ada SM 8.9+) 또는 NVFP4 (Blackwell SM 10.0+) HW 확보 후 재평가.

## TL;DR

NVIDIA 공식 TensorRT 10 문서가 INT4 Weight-only Quantization (WoQ)를 **"GEMM layers only"** 로 명시. YOLO26n은 거의 전부 Conv2d이므로 실행해도 `modelopt.INT4_BLOCKWISE_WEIGHT_ONLY_CFG.apply` 단계에서 silent skip 또는 apply guard RuntimeError로 즉시 실패. Wave 3/7/8의 "외부 라이브러리가 LLM/MatMul 전제로 설계된 경로에서 CNN 좌초" 패턴 연장선.

## 공식 문서 증거 (핵심 인용)

### TensorRT 10 Working with Quantized Types
> "WoQ is available only for INT4 block quantization with **GEMM layers**."

출처: <https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html>

TRT 10.x 모든 minor 버전(10.1~10.13)에서 동일 제약. cuBLAS 12.x의 W4A16 MatMul 경로는 Ampere SM 8.6에서 제공되나 **MatMul/GEMM 전용** — Conv는 cuDNN 경로를 타고 INT4 네이티브 지원 없음.

### HuggingFace diffusers — NVIDIA ModelOpt 통합
> "channel and block quantization generally don't work well with convolutional layers ... `disable_conv_quantization` parameter"

출처: <https://huggingface.co/docs/diffusers/main/en/quantization/modelopt>

NVIDIA 자신이 "conv에는 block quantization 적용 금지" default로 설정한 것이 최종 확정. `INT4_BLOCKWISE_WEIGHT_ONLY_CFG`는 Conv2d를 wrap하지 않는 설계.

### AWQ on CNN
AWQ(`INT4_AWQ_CFG`)는 per-output-channel activation magnitude 기반 salient weight scaling으로 LLM Linear hidden_dim 축에 최적화. Conv2d의 "channel"은 `out_channels`과 spatial kernel이 entangled라 AWQ 알고리즘 의미 자체가 약함. NVIDIA 공식 `llm_ptq` example이 주 사용처, vision은 VILA(multimodal with LLM) 만 언급.

## YOLO26n 구체 대입

| 지점 | 예상 결과 |
|---|---|
| `mtq.quantize(yolo.model, INT4_BLOCKWISE_WEIGHT_ONLY_CFG)` | default `disable_conv_quantization=True`로 Conv2d silently skip. Linear/MatMul은 ~0개라 wrapped count ≈ 0 |
| `modelopt_qat.apply`의 `wrapped == 0` guard (line 53-57) | RuntimeError 즉시 raise. PTQ 단계조차 통과 못함 |
| (guard 우회 시) TRT parser | Conv에 inject된 INT4 Q/DQ 노드 reject 또는 silent FP16 fallback |
| 최종 Engine | FP16 engine과 byte-identical 또는 동등 성능. **INT4 명목만 있는 결과 JSON** 생성 위험 |

즉 실행해도 얻는 데이터는:
1. RuntimeError stack trace (그마저도 `_prepare_onnx`의 예외 처리가 `run_trt.py`에 부재하여 `meets_constraints=False`로 graceful degrade 안 됨 — 별도 수정 필요)
2. 또는 "FP16과 동일한 결과" — 이미 `#05 trt_fp16`에서 확보한 데이터

→ **새 정보 가치 ≒ 0**.

## 실행 대신 이 문서가 충족하는 것

Wave 12의 원래 목적("Ampere에서 YOLO-class CNN에 INT4 weight-only가 적합하지 않음"의 근거 확보)은:
- 실측 데이터 → 공식 문서 인용으로 대체
- 후임/외부 리뷰어가 "왜 이 조합을 포기했는지" 찾을 때 본 문서와 인용 링크만으로 충분
- Wave 3 INC SmoothQuant archive와 동일한 구조 (실측 없이 호환성 매트릭스로 결정)

## 재개 조건 (언제 다시 보나)

다음 **둘 중 하나**가 충족되면 재평가:

| 조건 | 의미 |
|---|---|
| HW 업그레이드 — FP8 지원 GPU (Ada Lovelace SM 8.9, Hopper SM 9.0, Blackwell SM 10.0) 확보 | FP8 native 경로로 `FP8_DEFAULT_CFG`, `W4A8_AWQ_BETA_CFG` 유효화. TRT FP8 engine builder 경로 활성화 |
| ModelOpt가 Conv2d INT4 공식 지원 발표 | 현 `disable_conv_quantization` default가 바뀌거나 `CONV2D_INT4_CFG` 신규 preset 등장 시 |

그 전까지 recipe bank의 INT4 공백(#26-#29)은 **의도된 공백**. 번호 할당 재사용 금지.

## 관련 아카이브

- Wave 3 INC SmoothQuant (CNN 좌초) — `docs/improvements/2026-04-18-trt-modelopt-audit.md`
- Wave 7 PyTorch PT2E + XNNPACK (외부 변환기 좌초) — `docs/improvements/2026-04-21-wave7-r3-r5-spike-results.md`
- Wave 8 ncnn (외부 변환기 좌초) — `docs/improvements/2026-04-21-wave8-r1-spike-results.md`
- Wave 12 (본 문서) — 공식 문서만으로 archive (실기 없음)

## 참고 링크

- [NVIDIA TensorRT 10 — Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [NVIDIA Model Optimizer — PyTorch Quantization Guide](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html)
- [NVIDIA Model Optimizer — Best practices for choosing quant methods](https://nvidia.github.io/Model-Optimizer/guides/_choosing_quant_methods.html)
- [HuggingFace diffusers — ModelOpt `disable_conv_quantization`](https://huggingface.co/docs/diffusers/main/en/quantization/modelopt)
- [NVIDIA blog — Optimizing LLMs with PTQ (AWQ scope)](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)
