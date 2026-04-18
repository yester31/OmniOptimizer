# OmniOptimizer Report

Baseline: `pytorch_fp32`  |  GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`  |  CUDA: `12.4`


| Rank | Recipe | fps(bs1) | fps(bs8) | p50 ms | mAP@0.5 | drop | mem MB | meets? |
|-----:|--------|---------:|---------:|-------:|--------:|-----:|-------:|:------:|
| 1 | `trt_fp16` | 391.7 | 782.3 | 2.55 | 0.554 | -0.11%p | 38 | ✔ |
| 2 | `ort_trt_fp16` | 202.9 | 382.2 | 4.93 | 0.554 | -0.11%p | — | ✔ |
| 3 | `torchcompile_fp16` | 164.6 | 378.3 | 6.07 | 0.553 | -0.01%p | 415 | ✔ |
| 4 | `ort_cuda_fp16` | 68.5 | 119.3 | 14.60 | 0.554 | -0.11%p | — | ✔ |
| 5 | `pytorch_fp32` | 52.8 | 238.5 | 18.93 | 0.553 | +0.00%p | 281 | ✔ |
| 6 | `trt_int8_ptq` | 785.2 | 670.0 | 1.27 | 0.475 | +7.82%p | 38 | ✘ |
| 7 | `modelopt_int8_mixed` | 418.9 | 859.7 | 2.39 | 0.537 | +1.64%p | 38 | ✘ |
| 8 | `modelopt_int8_entropy` | 409.1 | 859.6 | 2.44 | 0.537 | +1.64%p | 38 | ✘ |
| 9 | `modelopt_int8_ptq` | 400.3 | 766.9 | 2.50 | 0.521 | +3.19%p | 38 | ✘ |
| 10 | `modelopt_int8_percentile` | 366.4 | 777.9 | 2.73 | 0.521 | +3.19%p | 38 | ✘ |

## Recommendation

**`trt_fp16`** — fps 391.7 (bs1), mAP@0.5 0.554, drop -0.11%p.

## Issues
- `ort_trt_fp16`: execution_provider=TensorrtExecutionProvider, onnx=yolo26n_640_fp16_bs1.onnx
- `ort_cuda_fp16`: execution_provider=CUDAExecutionProvider, onnx=yolo26n_640_fp16_bs1.onnx
- `trt_int8_ptq`: mAP drop 7.82%p > 1.0%p
- `modelopt_int8_mixed`: mAP drop 1.64%p > 1.0%p
- `modelopt_int8_entropy`: mAP drop 1.64%p > 1.0%p
- `modelopt_int8_ptq`: mAP drop 3.19%p > 1.0%p
- `modelopt_int8_percentile`: mAP drop 3.19%p > 1.0%p
