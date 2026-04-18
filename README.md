# OmniOptimizer

Vision 모델을 특정 기기에 배포할 때, 여러 추론 엔진 × 여러 최적화 기법을
자동으로 돌려 보고 "이 환경엔 이게 제일 낫다"는 1등 추천을 뱉는 도구.

v1은 **YOLO26n × NVIDIA GPU 1대**에서 아래 12장 레시피를 끝까지 돌립니다.

| #  | Runtime                    | Technique                                        | Source       |
|---:|----------------------------|--------------------------------------------------|--------------|
| 0  | TensorRT                   | FP32 / TF32 baselines (00 + 00-tf32)             | `trt_builtin`|
| 1  | PyTorch eager              | FP32 (baseline)                                  | —            |
| 2  | PyTorch + `torch.compile`  | FP16                                             | —            |
| 3  | ONNX Runtime (CUDA EP)     | FP16                                             | —            |
| 4  | ONNX Runtime (TensorRT EP) | FP16                                             | —            |
| 5  | TensorRT                   | FP16                                             | —            |
| 6  | TensorRT                   | INT8 PTQ (entropy, 512 calib samples)            | `trt_builtin`|
| 7  | TensorRT                   | INT8 PTQ + 2:4 Sparsity (Ampere+) — **parked**   | `trt_builtin`|
| 8  | TensorRT                   | INT8 PTQ (modelopt, max calib)                   | `modelopt`   |
| 9  | TensorRT                   | INT8 PTQ (modelopt, entropy calib)               | `modelopt`   |
| 10 | TensorRT                   | INT8 PTQ (modelopt, percentile calib)            | `modelopt`   |
| 11 | TensorRT                   | INT8 PTQ + 2:4 Sparsity (modelopt) — **parked**  | `modelopt`   |
| 12 | TensorRT                   | INT8 PTQ + FP16 excludes (stem + cv2.*)          | `modelopt`   |
| 13 | TensorRT                   | INT8 PTQ (ort_quant, minmax)                     | `ort_quant`  |
| 14 | TensorRT                   | INT8 PTQ (ort_quant, entropy)                    | `ort_quant`  |
| 15 | TensorRT                   | INT8 PTQ (ort_quant, percentile)                 | `ort_quant`  |
| 16 | TensorRT                   | INT8 PTQ (ort_quant, distribution)               | `ort_quant`  |
| 17 | TensorRT                   | INT8 PTQ (neural_compressor, minmax)             | `neural_compressor` |
| 18 | TensorRT                   | INT8 PTQ + SmoothQuant (neural_compressor)       | `neural_compressor` |
| 19 | TensorRT                   | INT8 QAT (neural_compressor) — **parked**        | `neural_compressor` |

레시피 #8–#10은 NVIDIA ModelOpt의 ONNX-path PTQ로, `trt_builtin` INT8
캘리브레이터의 mAP drop 문제(-7.9%p)를 **-1.6~1.9%p 수준으로 개선**합니다. #12는
민감 레이어(stem + bbox regression head)를 FP16에 남기는 혼합 정밀도 실험.
자세한 비교와 Phase 3 acceptance 결과는 `report.md` 참조.

> **#7, #11 parked — sparsity는 학습 코드가 들어오기 전까지 평가 대상에서 제외**
>
> 레시피/스키마/runner 코드는 **그대로 유지**됩니다. `make recipe-07`, `make recipe-11`로
> 개별 실행은 여전히 가능 (기존 결과 JSON도 `results/`에 보존). 다만 `make all`에서
> 자동 실행되지 않고, `make report` 순위표에서도 `--exclude` 플래그로 빠집니다.
> Phase 3 실험에서 post-training 2:4는 nano YOLO의 mAP를 사실상 0으로 무너뜨린 것이
> 확인되어 (`docs/plans/2026-04-18-phase3-int8-accuracy.md` 참조), sparsity-aware
> training이 본 저장소에 들어오는 시점에 재평가할 예정입니다.

## Quick start (Docker 권장)

```bash
docker build -t omnioptimizer:v1 .
docker run --rm --gpus all -v "$PWD":/workspace/omnioptimizer omnioptimizer:v1 \
    bash -c "make all && cat report.md"
```

## Quick start (pip, 본인 환경)

```bash
python -m venv .venv && source .venv/bin/activate   # 또는 .venv\Scripts\activate
pip install -e ".[all]"
# tensorrt는 NVIDIA 휠 인덱스에서 별도 설치 필요:
#   pip install --extra-index-url https://pypi.nvidia.com tensorrt
make all
cat report.md
```

## 결과 해석

`make all`이 끝나면:

- `results/01_pytorch_fp32.json` … `results/11_modelopt_int8_sparsity.json` — 레시피별 측정값.
- `results/_env.json` — 실행 환경 스냅샷 (GPU, CUDA, 드라이버, 버전).
- `report.md` — Windows/WSL 비교 순위표 + 시나리오별 추천.

각 JSON은 `scripts/_schemas.py::Result` 스키마를 따릅니다.

### INT8 캘리브레이션 데이터셋

ModelOpt 계열(#8–#11)은 COCO val 이미지로 캘리브레이션합니다:

```bash
export OMNI_COCO_YAML=$PWD/coco_val_only.yaml   # val2017.txt 경로 포함
```

미설정 시 random-normal 텐서로 폴백합니다(정확도 저하 큼).

## 재현성 체크

동일 Docker 이미지 + 동일 GPU에서 `make all`을 한 번 더 돌리면:

- 각 레시피의 p50 latency 차이 ±5% 이내,
- mAP@0.5 차이 ±0.002 이내

가 기대 동작입니다. 벗어나면 `nvidia-smi -lgc`로 클럭 락이 걸렸는지, 다른
프로세스가 GPU를 나눠 쓰고 있는지 먼저 확인하세요.

## 한 레시피만 다시 돌리기

```bash
make recipe-06      # TRT INT8 PTQ만 재실행
python scripts/recommend.py --results-dir results --out report.md
```

## v1이 일부러 뺀 것

OpenVINO, TFLite, CoreML, TVM, NCNN, SNPE, MIGraphX, QAT, Knowledge
Distillation, Early Exit, Token Merging, W4A16, FP8, 크로스 하드웨어 비교.
전부 v2 이후.

## 레시피 YAML 스키마

예: `recipes/06_trt_int8_ptq.yaml`

```yaml
name: trt_int8_ptq
model:
  family: yolo26
  variant: n
  weights: yolo26n.pt
runtime:
  engine: tensorrt
  dtype: int8
technique:
  name: int8_ptq
  calibrator: entropy
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
  max_map_drop_pct: 1.0
  min_fps_bs1: 30
```

새 런타임·기법을 추가하려면: `recipes/`에 YAML 한 장 + 해당 runner
(`run_pytorch.py` / `run_ort.py` / `run_trt.py`) 에 dispatch 분기 추가.

### `technique.source` 디스패처 (v1.1+)

`run_trt.py`는 INT8/sparsity 백엔드를 `technique.source`로 라우팅합니다:

- `trt_builtin` (기본): TRT 내장 `IInt8EntropyCalibrator2` + SPARSE_WEIGHTS.
  참고: TRT 10.1 기준 implicit quantization 경로는 deprecated이며 NVIDIA는
  explicit quantization (QDQ-ONNX, = modelopt 경로)을 권장합니다. 본 저장소는
  비교 baseline으로 유지합니다.
- `modelopt`: `modelopt.onnx.quantization.quantize`로 clean ONNX에 QDQ 주입
  (explicit quantization). caliber별 스케일(`max`/`entropy`/`percentile`)은
  `technique.calibrator`로 선택.
- `ort_quant` (Wave 3+): `onnxruntime.quantization.quantize_static`로 QDQ 주입.
  `calibrator`는 `minmax` / `entropy` / `percentile` / `distribution` 4종.
  TRT 호환을 위해 내부적으로 `ActivationSymmetric`, `WeightSymmetric`,
  `AddQDQPairToWeight`, `DedicatedQDQPair`를 강제 설정.
- `neural_compressor` (Wave 3+): Intel Neural Compressor PTQ + SmoothQuant.
  `calibrator`는 `minmax` (기본 PTQ) 또는 `smoothquant` (α=0.5, activation
  outlier 이주). INC 2.x 필수 (`neural-compressor>=2.5,<3.0`), Python 3.13에서는
  `setuptools<81`도 함께 필요 (INC 내부가 `pkg_resources`에 의존).

모델옵트 경로는 ultralytics의 inference head 파이프를 보존하므로 validator가
그대로 호환됩니다.

#### v1.2 옵션 필드

- **`technique.sparsity_preprocess: "2:4"`** — ONNX export 전에
  `modelopt.torch.sparsity.sparsify(mode="sparse_magnitude")`로 가중치를
  2:4 패턴으로 사전 pruning. 이게 있어야 TRT의 `SPARSE_WEIGHTS` 플래그가
  실제 sparse INT8 tactic을 선택합니다. 단독으로 플래그만 세우는 v1.1
  경로는 no-op이었음.
- **`technique.nodes_to_exclude: [str]`** — modelopt QDQ 주입 시 해당 ONNX
  노드를 FP16에 남깁니다. 전형적 후보: stem Conv (`/model.0/conv/Conv`),
  detect head bbox regression (`/model.23/cv2.*/Conv`). classification
  branch (`/model.23/cv3.*/Conv`)는 softmax 덕에 INT8 오차에 robust하므로
  **일반적으로 exclude에 포함하지 않습니다** — 넣으면 fps만 깎임.

## 설계 배경

이 저장소의 구조·결정 사유는 `CLAUDE.md`와 (개인 머신의)
`~/.gstack/projects/yester31-OmniOptimizer/yeste-main-design-20260417-093458.md`
에 기록되어 있습니다.
