# Wave 7 R3/R5 spike results (2026-04-21)

**TL;DR**: Wave 7 plan의 두 축(PyTorch PT2E + ORT XNNPACK EP)이 모두 BLOCKED.
Plan 재검토 필요.

## Environment
- Windows 11 Home, Intel i7-11375H (Tiger Lake, AVX-512 VNNI)
- Python 3.13.3
- `torch==2.8.0+cu129`
- `onnxruntime==1.22.0` (실은 `onnxruntime-gpu` wheel — CUDA/TRT EP 제공)
- `ultralytics==8.4.27`
- Spike script: `scripts/_spike_wave7_r3.py`

## R3 — torch.export + ultralytics YOLO26n : BLOCKED

### strict=True
```
AssertionError: Mutating module attribute anchors during export.
from user code:
  File ".../ultralytics/nn/modules/head.py", line 179, in _get_decode_boxes
    self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(...))
```

Detect head (`ultralytics.nn.modules.head.Detect`)의 `_get_decode_boxes`
메서드가 forward 중 `self.anchors` / `self.strides`를 **모듈 attribute로
재할당**. torch.export strict 모드는 export 트레이싱 중 모듈 state 변화를
금지하므로 실패.

### strict=False
- export 자체는 2655ms에 성공
- 그러나 `exported.module()(example)` 호출 시 output이 `dict` (training
  mode 형태로 추정)로 반환 → Wave 7이 기대하는 `Tensor` 또는 `(batch,
  300, 6)` 형태 incompatible
- 후속 PT2E 경로가 이 dict 위에서 동작 안 함

### 파장
- Wave 7 Task 2 (PT2E #40/#41) **전체 parked**
- Mitigation: ultralytics Detect head를 stateless wrapper로 감싸는 2~4시간
  R&D 가능. 성공 확률 미확인.

## R5 — ORT XNNPACK EP : **아예 존재하지 않음** (plan 오류)

```python
>>> import onnxruntime as ort
>>> ort.__version__
'1.22.0'
>>> ort.get_available_providers()
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

**`XnnpackExecutionProvider` 부재**. Wave 7 plan은 "onnxruntime 1.17+는
XNNPACK EP 내장"이라고 적었지만 이는 **일반 `onnxruntime` wheel**에 한함.
우리는 `onnxruntime-gpu` wheel(Wave 1~3 GPU recipes 때문에)을 설치한 상태로
XNNPACK EP는 포함 안 됨. pip 상 두 패키지는 **상호배타** (같은 `onnxruntime`
import 네임 공유).

### 교차 검증
Stage 4에서 Wave 6 static INT8 ONNX를 `providers=["Xnnpack...", "CPU..."]`로
요청했으나:
```
UserWarning: Specified provider 'XnnpackExecutionProvider' is not in available
provider names. Available providers: 'TensorrtExecutionProvider,
CUDAExecutionProvider, CPUExecutionProvider'
```
→ silently CPU로 fallback, 즉 Wave 7 #42/#43은 현재 환경에서는 Wave 6 #30/#33
과 byte-identical 결과를 낼 뿐.

### 파장
- Wave 7 Task 3 (XNNPACK #42/#43) **현재 venv에서 불가능**
- Mitigation 옵션: 별도 venv에 `onnxruntime` CPU wheel 설치. Wave 6/7 공존이
  아니라 venv 전환 전제의 wave로 재정의 필요.

## Verdict

Wave 7 양 축이 blocked. 단순 partial-scope로 해결 안 됨.

### 옵션 A (권장) — Wave 7 archive
- Wave 6 결과를 마지막 active wave로 확정
- Wave 8 ncnn으로 점프 (모바일 배포 가치 명확)
- 이 spike의 PT2E R&D는 1~2년 후 ExecutorTorch 성숙 시점에 함께 재검토

### 옵션 B — Wave 7 rescope "별도 venv XNNPACK only"
- 신규 venv `.venv-cpu-xnn/` → `pip install onnxruntime`(CPU)
- Wave 7 scope = #42/#43만
- PT2E #40/#41은 Wave 8+ 후보로 이관
- 사용자 DX 비용: venv 두 개 관리

### 옵션 C — PT2E unlock R&D + 별도 venv XNNPACK
- ultralytics Detect head wrapper 작성 (수 시간)
- venv 분리도 병행
- 가장 많은 backend 다양성, 가장 큰 투자

## Reproduce
```bash
python scripts/_spike_wave7_r3.py
```
~30초 (torch warm cache 가정).
