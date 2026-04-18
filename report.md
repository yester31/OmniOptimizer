# OmniOptimizer Report

Baseline: `pytorch_fp32`  |  GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`  |  CUDA: `12.4`


| Rank | Recipe | fps(bs1) | fps(bs8) | p50 ms | mAP@0.5 | drop | mem MB | meets? |
|-----:|--------|---------:|---------:|-------:|--------:|-----:|-------:|:------:|
| 1 | `trt_fp16` | 238.9 | 496.4 | 4.19 | 0.554 | -0.11%p | 38 | ✔ |
| 2 | `ort_trt_fp16` | 202.9 | 382.2 | 4.93 | 0.554 | -0.11%p | — | ✔ |
| 3 | `torchcompile_fp16` | 164.6 | 378.3 | 6.07 | 0.553 | -0.01%p | 415 | ✔ |
| 4 | `ort_cuda_fp16` | 68.5 | 119.3 | 14.60 | 0.554 | -0.11%p | — | ✔ |
| 5 | `pytorch_fp32` | 52.8 | 238.5 | 18.93 | 0.553 | +0.00%p | 281 | ✔ |
| 6 | `modelopt_int8_percentile` | 436.0 | 675.0 | 2.29 | 0.520 | +3.32%p | 38 | ✘ |
| 7 | `modelopt_int8_ptq` | 424.8 | 679.6 | 2.35 | 0.521 | +3.19%p | 38 | ✘ |
| 8 | `modelopt_int8_entropy` | 403.3 | 689.3 | 2.48 | 0.537 | +1.63%p | 38 | ✘ |
| 9 | `trt_int8_ptq` | 310.3 | 695.9 | 3.22 | 0.474 | +7.92%p | 38 | ✘ |
| 10 | `modelopt_int8_mixed` | 303.2 | 754.7 | 3.30 | 0.537 | +1.62%p | 38 | ✘ |

## Recommendation

**`trt_fp16`** — fps 238.9 (bs1), mAP@0.5 0.554, drop -0.11%p.

## Issues
- `ort_trt_fp16`: execution_provider=TensorrtExecutionProvider, onnx=yolo26n_640_fp16_bs1.onnx
- `ort_cuda_fp16`: execution_provider=CUDAExecutionProvider, onnx=yolo26n_640_fp16_bs1.onnx
- `modelopt_int8_percentile`: mAP drop 3.32%p > 1.0%p
- `modelopt_int8_ptq`: mAP drop 3.19%p > 1.0%p
- `modelopt_int8_entropy`: mAP drop 1.63%p > 1.0%p
- `trt_int8_ptq`: mAP drop 7.92%p > 1.0%p
- `modelopt_int8_mixed`: mAP drop 1.62%p > 1.0%p
