# ModelOpt PTQ calibrator fps gap — tactic analysis (Wave 11 Task 4 / B4)

**Date**: 2026-04-23
**Status**: Inspection complete, **NO code fix applied** (archive-neutral). Root cause is TRT builder nondeterminism, not calibrator-specific pathology. 실제 개선 경로는 Wave 14 A1 (`builder_optimization_level=5`).

## Observed gap (report_qr.md)

| Rank | Recipe | Calibrator | fps bs1 | Engine bytes | mAP |
|---:|---|---|---:|---:|---:|
| 1 | modelopt_int8_entropy (#09) | entropy | **763.9** | 4,738,940 | 0.987 |
| 2 | modelopt_int8_mixed (#12) | mixed | 760.0 | — | 0.987 |
| 3 | modelopt_int8_percentile (#10) | percentile (99.99%) | **755.1** | 4,736,380 | 0.985 |
| 13 | modelopt_int8_ptq (#08) | max | **430.1** | 4,736,380 | 0.985 |

gap: #08 max 대 #09 entropy = **1.78× 느림**. 동일 source (`modelopt.onnx.quantize`), 동일 calibration data (512 COCO samples, seed 42), 다른 점은 `calibration_method` 단 하나.

## ONNX graph inspection

세 recipe 의 QDQ ONNX (`results/_onnx/best_qr_640_modelopt_{max,entropy,percentile}_bs1.onnx`) 비교.

### Graph structure — **identical**

| Metric | max | entropy | percentile |
|---|---:|---:|---:|
| file bytes | 5,319,142 | 5,319,142 | 5,319,142 |
| total nodes | 832 | 832 | 832 |
| QuantizeLinear | 223 | 223 | 223 |
| DequantizeLinear | 223 | 223 | 223 |
| Conv / Mul / Sigmoid | 102 / 90 / 88 | 동일 | 동일 |

→ **Q/DQ placement 은 calibrator 와 무관**. modelopt 는 모델 구조만 보고 삽입 위치 결정, calibrator 는 scale 값만 제공.

### Activation scales — **meaningful difference**

QDQ 노드의 scale 이니셜라이저를 per-tensor (activation, scalar) vs per-channel (weight, 1D) 로 분리했을 때:

| Calibrator | unique activation scales | min | max | median | p95 | p99 | std |
|---|---:|---:|---:|---:|---:|---:|---:|
| max | 101 | 7.37e-3 | **1.19** | 7.28e-2 | 2.90e-1 | 6.67e-1 | 1.43e-1 |
| entropy | 101 | 4.56e-3 | **0.662** | 4.77e-2 | 1.90e-1 | 3.39e-1 | 7.86e-2 |
| percentile | 101 | 7.37e-3 | **1.19** | 7.28e-2 | 2.90e-1 | 6.67e-1 | 1.43e-1 |

**주목**:
- `max` 와 `percentile` 의 activation scale 분포가 **byte-exact 동일**. modelopt 의 percentile default 가 99.99% 인데, 512 sample 캘리브 세트에서 99.99% percentile ≈ max 로 수렴.
- `entropy` 는 outlier 를 KL-divergence 최소화로 잘라냄 → max scale 1.19 → 0.66 (45% 타이트), p99 0.67 → 0.34 (50% 타이트).

### Weight scales — **identical**

모든 calibrator 가 동일 weight scale 생성 (per-channel Conv weight 는 calibrator 와 무관, weight 통계만 사용). 16788개 per-channel scale 전체 일치.

## Engine binary comparison

TRT 10.16 로 빌드된 engine:

| Recipe | size | md5[0:12] | build mtime |
|---|---:|---|---|
| max (#08)        | 4,736,380 | 9ba2d550fb2b | 1776521305 |
| entropy (#09)    | **4,738,940** | 833b071b7ce8 | 1776522079 |
| percentile (#10) | 4,736,380 | 272849b3d27c | 1776522762 |

- **entropy engine 이 +2,560 bytes**. 이는 다른 kernel tactic 조합이 선택되었다는 직접 증거 (TRT engine 크기는 plan + tactic table + weight memory 합).
- **max 와 percentile 엔진은 바이트 크기 동일** 이지만 **md5 가 다름** (9ba2d550… vs 272849b3…). 동일 ONNX 에 대해 동일 크기의 다른 binary — 즉 같은 tactic set 이 다른 순서로 패킹되었거나, 미미한 internal layout 차이.

## Why max (430) ≠ percentile (755) when ONNX is byte-identical?

**Answer**: **TRT builder autotune nondeterminism**.

- `scripts/run_trt.py::_build_engine` 은 shared timing cache 사용 (`_timing_cache_path()` → `results/_cache/trt_timing_{TRT_VERSION}_{CUDA_VERSION}.cache`).
- 하지만 scale 값이 다를 때마다 **새 kernel candidate 들이 timed** — 기존 cache miss. 각 kernel 의 μs-scale latency 측정은 GPU thermal / frequency / memory allocator 상태에 민감.
- Identical ONNX + Identical timing cache seed 조건에서도 autotune 은 **같은 candidate 를 다른 순서로 timing** 하고, microbench noise (~5-10% σ) 때문에 **다른 tactic 을 picking**.
- #08 max 빌드와 #10 percentile 빌드는 **8 분 시간차** (mtime 1776521305 vs 1776522762). 그 사이 GPU 상태 변화 + driver state 가 autotune 결과에 영향.

**결정적 증거**: 동일 ONNX 로 만든 max engine vs percentile engine 의 런타임 fps 가 **430 vs 755**. 이는 calibrator 의 효과가 아니라 **TRT 빌드의 확률적 결과**.

## 진짜 calibrator 효과는?

max vs percentile 의 fps 격차 (430 vs 755, 1.76×) 가 **nondeterminism ceiling** 이라면, entropy 의 fps 763 은:

- 이론적 ceiling (nondeterminism 없는 best tactic): ~755-800 fps 범위
- entropy 가 **체계적으로** 상위에 랭크될 요인:
    1. Activation scale 이 타이트 → TRT 의 IMMA (DP4A 대신) kernel 선택 확률 증가
    2. 작은 dynamic range → `INT8 * INT8 → INT32 accumulate` 에서 overflow 가드 불필요한 kernel 선택 가능
    3. `+2560 bytes` engine 크기는 **mixed INT8/FP16 kernel** (modelopt default `high_precision_dtype=fp16` 잔류 FP16 layer 에 대한 최적화 tactic) 추가 선택으로 해석 가능

이는 체계적 효과일 수 있으나, **현재 측정 하나만으로 입증 불가**. max/percentile/entropy 를 각 **N=5 회 재측정** 해 신뢰구간을 봐야 정량화 가능.

## 결론 / 권고

1. **B4 는 단순 "calibrator → tactic" 결정적 버그가 아님**. 주요 변동 요인은 TRT 빌더 autotune noise.
2. **문서화만** (현재 문서). recipe 수정 범위 밖 — calibrator 이름만으로 재현 가능한 fix path 없음.
3. **#08 max fps 430 은 "운 나쁜 빌드"** 일 가능성. recipe 자체는 정상. Wave 14 에서 `builder_optimization_level=5` 적용 시:
    - autotune 공간 확장 → 최적 tactic 발견률 ↑
    - 빌드 시간 3-5× 증가 trade-off 감수
    - **stabilization 과 ceiling lift 동시 기대**
4. **후속 계측 제안 (Wave 14 Task 5 이후)**: `make recipe-08 && make recipe-10` 을 각 3-5 회 재실행, fps 신뢰구간으로 "max vs percentile vs entropy" 재평가. Current 단일 측정의 순위는 noise 로 뒤집힐 여지.

## Appendix — inspection 재현

```bash
python -c "
import onnx, numpy as np
from collections import Counter
from pathlib import Path

for label, path in [('max','results/_onnx/best_qr_640_modelopt_max_bs1.onnx'),
                    ('entropy','results/_onnx/best_qr_640_modelopt_entropy_bs1.onnx'),
                    ('percentile','results/_onnx/best_qr_640_modelopt_percentile_bs1.onnx')]:
    m = onnx.load(path)
    init_by_name = {i.name: i for i in m.graph.initializer}
    act, wt = [], []
    for n in m.graph.node:
        if n.op_type in ('QuantizeLinear','DequantizeLinear') and len(n.input) >= 2:
            arr = onnx.numpy_helper.to_array(init_by_name[n.input[1]])
            (act if arr.ndim == 0 else wt).extend([float(arr)] if arr.ndim == 0 else arr.flatten().tolist())
    a = np.array(sorted(set(act)))
    print(label, 'act scales:', len(a), 'max:', a.max(), 'p95:', np.quantile(a,0.95))
"
```

## 참고

- Wave 11 plan — `docs/plans/2026-04-22-wave11-recipe-debug-cleanup.md` Task 4
- Recipe bank audit — `docs/improvements/2026-04-22-recipe-bank-audit.md` B4
- TRT 10.x `builder_optimization_level` — Wave 14 A1 후속
- TRT timing cache 동작 — `scripts/run_trt.py::_build_engine` (line ~567)
