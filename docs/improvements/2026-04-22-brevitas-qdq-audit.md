# Brevitas QDQ 감사 — 2026-04-22

**Status**: 가설 기각. brevitas 경로의 fps 손실은 **"Missing scale" 경고와 무관**.
**Decision**: brevitas recipes (#20, #21, #22) 는 modelopt 대비 추가 가치 없음. **archive 권고** (사용자 결정 대기).

## 가설

Phase 8 baseline audit 에서 `brevitas_int8_*` QDQ ONNX 가 **Q=92 / DQ=204 비대칭** (weight DQ-only 패턴) 로 확인됨. modelopt 은 Q=223 / DQ=223 완전 대칭. report_qr.md 에서 brevitas fps (401) 가 modelopt (763) 대비 47% 수준 → **"TRT 가 activation scale 부재로 일부 layer FP16 fallback → 손실"** 으로 가설.

## 방법

`scripts/_spike_brevitas_audit.py` — 각 QDQ ONNX 를 `_build_engine(dtype="int8", quant_preapplied=True)` 로 새로 빌드하면서 stderr 의 `Missing scale and zero-point` 경고를 regex 로 카운트.

## 결과

| target | missing_scale warnings | detect_head missing | engine MB |
|---|---:|---:|---:|
| `brev_percentile` | **0** | 0 | 4.86 |
| `brev_mse` | **0** | 0 | 4.86 |
| `modelopt_entropy` | 0 | 0 | 4.59 |

**가설 기각**: brevitas 도 TRT build 시 "Missing scale" 경고 **전혀 없음**. 즉 Q=92/DQ=204 비대칭은 **TRT가 정상 해석 가능**한 QDQ 포맷 (weight-only DQ 는 INT8 weight tensor + scale 로 해석되고, 그 downstream activation 은 별도 Q 노드 없이도 FP16 경로로 자연스럽게 연결됨).

## 그럼 왜 brevitas fps 47%?

가능한 요인 (확인 불가):
1. **Calibration data 차이**: brevitas 는 brevitas 자체 QAT/PTQ flow (Percentile/MSE, per-tensor) — modelopt 은 COCO val 이미지 기반 entropy. 실제 activation range 차이로 per-tensor scale 값이 달라져 TRT tactic 선택 분기.
2. **Per-channel vs per-tensor**: modelopt 기본 per-channel weight quantization (`INT8_DEFAULT_CFG`). brevitas 는 per-tensor. Per-channel은 TRT 가 더 tight한 INT8 kernel 선택 가능.
3. **BN fold 패턴 차이**: brevitas raw ONNX 에 BatchNorm 96개 포함. bs1 ONNX는 785 nodes(BN fold 후), modelopt 832 nodes. node count 는 brevitas 가 적지만 layer fusion 이 다를 수 있음.
4. **Model 크기 difference**: report_qr.md 에 brevitas `mem=90 MB` vs modelopt `mem=38 MB`. engine 자체는 4.86 MB 로 거의 동일하지만 Result.peak_gpu_mem_mb 가 90 MB — TRT runtime allocation 차이.

추가 검증은 **TRT engine inspection API** (`engine.get_layer_information()`, `trt-engine-explorer` 등) 로 가능하나, 이 wave 는 "brevitas 중복 recipe 제거 여부 결정" 이 목적이고 심층 profiling 은 scope 초과.

## 결론

- **brevitas recipes 의 실질 이점 없음** 확인
  - mAP: baseline 0.988 대비 drop −0.01~−0.03%p (유리) — 그러나 modelopt_int8_entropy (+0.07%p) 와 사실상 동일 수준
  - fps: modelopt 대비 47% (열세)
  - memory: modelopt 대비 2.4× (열세)
- **Q/DQ 비대칭이 원인 아님** → brevitas 쪽 graph export 전략 자체를 모방해도 개선 불가
- **archive 권고**: recipe bank 단순화, 유지보수 비용 절감

## Archive 후보 recipes

- `#20 brevitas_int8_percentile` (rank 13, fps 401.2)
- `#21 brevitas_int8_mse` (rank 14, fps 396.9)
- `#22 brevitas_int8_entropy` (존재하지만 실행되지 않은 recipe)

## Archive 반대 의견 (반박 가능)

- Brevitas 는 **QAT 파이프라인**을 원래 지원 (fake-quant + 학습 가능 scale). PTQ fine-tune 없는 플릿(생산 배포) 에서는 유지 가치 있음.
- **Academic baseline** 로 인용 가능 — 상업용 modelopt 이 아닌 오픈소스 QAT 프레임워크 비교 대상.
- 하지만 이 프로젝트 (OmniOptimizer) 는 **벤치 자동화** 가 목적이라 학술적 비교 가치는 부차적.

최종 판단은 사용자 — archive / 유지 선택.

## 산출물

- `scripts/_spike_brevitas_audit.py` — 재현 스크립트
- `logs/brevitas_audit.json` — 측정 결과
- `trained_weights/23_fair_bench_engines/audit/brev_*_audit.engine` — 진단용 engine (삭제 가능)

## 참고

- Phase 8 baseline audit — `docs/improvements/2026-04-22-wave10-pruning-extended-eval.md` 의 "다른 모델 QDQ 커버리지 감사" 섹션
- modelopt QDQ 전략 — `scripts/run_trt.py::_prepare_modelopt_onnx`
- brevitas export 코드 — `scripts/run_trt.py::_prepare_brevitas_onnx` (line 371~)
