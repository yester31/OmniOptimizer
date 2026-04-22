# Wave 8 R1/R2 spike results (2026-04-21)

**TL;DR**: R2 (pip wheel) 해소, **R1 (YOLO26n 변환 호환성) BLOCKED**. Wave 7에 이어 Wave 8도 archive. pnnx가 YOLO26n Detect head의 `ReduceMax`/`TopK`/`GatherElements` op을 silent drop해 ncnn 런타임이 그래프를 reject.

## Environment
- Windows 11, Intel i7-11375H
- Python 3.13.3
- `ncnn==1.0.20260114` (PyPI wheel) — Python 3.13 win_amd64 제공 확인
- `pnnx==20260409` (PyPI wheel) — Scripts/pnnx.exe 설치
- Spike: `scripts/_spike_wave8_r1.py` + ad-hoc 재현

## R2 — pip wheel : RESOLVED

```
$ pip install ncnn
Successfully installed ncnn-1.0.20260114 portalocker-3.2.0

$ pip install pnnx
Successfully installed pnnx-20260409
```

두 패키지 모두 Windows Python 3.13 wheel 존재. Wave 7 TFLite가 실패한 경로(Windows wheel 부재)는 Wave 8에선 재현 안 됨.

## R1 — YOLO26n 변환/실행 : BLOCKED

### Stage 2: pnnx 변환
`pnnx best_qr_640_fp32_dyn.onnx inputshape=[1,3,640,640] ...` 실행 결과 변환
자체는 성공 (exit 0)하고 4개 파일 생성:
- `best_qr.pnnx.param`, `best_qr.pnnx.bin` (pnnx 중간 포맷)
- `best_qr.ncnn.param`, `best_qr.ncnn.bin` (ncnn 최종 포맷)

그러나 **stdout에 `ignore ...` 메시지 다수**:
```
ignore ReduceMax ReduceMax_367 param axes=-1
ignore ReduceMax ReduceMax_367 param keepdims=0
ignore TopK TopK_368 param axis=-1
ignore TopK TopK_368 param largest=1
ignore torch.tile torch.tile_37 param dims=(1,1,2)
ignore GatherElements GatherElements_372 param axis=1
ignore TopK TopK_374 ...
ignore pnnx.Expression pnnx_expr_0 param expr=2
ignore Mod Mod_377 param fmod=0
ignore Tensor.to Tensor.to_0 param dtype=torch.float
ignore Gather Gather_381 param axis=0
ignore GatherElements GatherElements_384 param axis=1
```

"ignore"는 파라미터 파싱 실패가 아니라 **op이 ncnn graph에 emit되지 않음을
의미**. 즉 YOLO26n의 end-to-end NMS 파이프라인(ReduceMax, TopK 기반의
post-processing + anchor decode)이 ncnn 변환 결과에서 **통째로 빠짐**.

### Stage 3: ncnn forward
```python
net = ncnn.Net()
net.load_param("best_qr.ncnn.param")
net.load_model("best_qr.ncnn.bin")
ex = net.create_extractor()
ex.input("in0", ncnn.Mat(...))
ret, out = ex.extract("out0")
```

결과:
```
layer ReduceMax not exists or registered
network graph not ready
find_blob_index_by_name in0 failed
find_blob_index_by_name out0 failed
Segmentation fault (SIGSEGV)
```

ncnn 런타임은 로드된 graph가 valid하지 않다고 판단 (drop된 layer 때문) →
blob 참조 실패 → native crash.

### 분석 — 반복되는 YOLO26n 호환성 클래스

Wave 3 INC SmoothQuant, Wave 7 R3 torch.export, Wave 8 R1 pnnx → 모두
**YOLO26n의 end-to-end NMS/Detect head에서 막힘**. 공통 원인:
- 동적 shape (TopK, Gather)
- 모듈 state mutation (anchors / strides 재할당)
- 비-standard op (GatherElements, ScatterND)

Workaround 후보 (모두 Wave 8 rescope 또는 별도 wave 필요):
1. **YOLO26n `end2end=False` export** — Detect head 없이 raw grid output만 ONNX로. Python에서 NMS 후처리. ncnn은 pure inference만 담당.
2. **pnnx `moduleop=...` 옵션으로 custom op 등록** — R&D 수 시간, 성공 보장 없음.
3. **raw backbone+FPN만 변환 + 상위 NMS Python 작성** — 측정 대상 분할, accuracy eval 재구축 필요.

### Parity check — Wave 6 ONNX가 정말 그 op들을 포함하는가?

Wave 6 `best_qr_640_fp32_dyn.onnx`는 ultralytics `model.export(format="onnx",
dynamic=True)`의 결과로, ultralytics는 기본적으로 end-to-end NMS를 포함해
export한다. 그래서 ReduceMax/TopK/GatherElements가 ONNX 그래프에 실제로
존재. ORT CPU EP와 OpenVINO는 이 op들을 네이티브 지원하지만 ncnn은 아님.

## Verdict

Wave 8도 archive. 다음 후보:

### Option A — Wave 8 rescope: end2end=False 경로 (미래 세션)
- ultralytics export 옵션을 `end2end=False`로 해서 Detect head 제거
- ncnn 변환 성공 예상 (backbone/FPN만 — 표준 op만 포함)
- 별도 Python NMS adapter 필요 (`accuracy eval` 경로 변경)
- 추가 작업 비용: 구현 +~4h, accuracy 검증 +~2h
- Wave 6 backbone과 동일 accuracy/latency 기대

### Option B — Wave 9 DirectML EP 직진
- ONNX → ORT with DirectML provider → Windows NPU/Arc GPU/AMD
- Wave 7/8이 겪은 "외부 변환기 호환성" 리스크 없음 (ORT 네이티브 경로)
- 리스크는 venv 분리 / QDQ INT8 fallback 정도

### Option C — Wave 6 마감, 모바일 확장 자체를 보류
- Wave 6 결과(28 recipes, 6 CPU backend benchmark)로 이미 충분한 value
- 모바일 배포는 "프로젝트 범위 밖"으로 선언
- 남은 에너지를 Wave 6의 finer-point 개선에 투자 (예: #33 mAP=0 debug,
  iter_cooldown_ms 실험)

## Reproduce
```bash
python scripts/_spike_wave8_r1.py
```
첫 실행은 pnnx 변환에 2-5분 소요.
