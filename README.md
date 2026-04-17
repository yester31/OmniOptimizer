# OmniOptimizer

Vision 모델을 특정 기기에 배포할 때, 여러 추론 엔진 × 여러 최적화 기법을
자동으로 돌려 보고 "이 환경엔 이게 제일 낫다"는 1등 추천을 뱉는 도구.

v1은 **YOLO26n × NVIDIA GPU 1대**에서 아래 7장 레시피를 끝까지 돌립니다.

| #  | Runtime                    | Technique                              |
|---:|----------------------------|----------------------------------------|
| 1  | PyTorch eager              | FP32 (baseline)                        |
| 2  | PyTorch + `torch.compile`  | FP16                                   |
| 3  | ONNX Runtime (CUDA EP)     | FP16                                   |
| 4  | ONNX Runtime (TensorRT EP) | FP16                                   |
| 5  | TensorRT                   | FP16                                   |
| 6  | TensorRT                   | INT8 PTQ (entropy, 512 calib samples)  |
| 7  | TensorRT                   | INT8 PTQ + 2:4 Sparsity (Ampere+)      |

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

- `results/01_pytorch_fp32.json` … `results/07_trt_int8_sparsity.json` — 레시피별 측정값.
- `results/_env.json` — 실행 환경 스냅샷 (GPU, CUDA, 드라이버, 버전).
- `report.md` — 순위표 + 추천 한 줄.

각 JSON은 `scripts/_schemas.py::Result` 스키마를 따릅니다.

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

## 설계 배경

이 저장소의 구조·결정 사유는 `CLAUDE.md`와 (개인 머신의)
`~/.gstack/projects/yester31-OmniOptimizer/yeste-main-design-20260417-093458.md`
에 기록되어 있습니다.
