# OmniOptimizer Report

Baseline: `pytorch_fp32`  |  GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`  |  CUDA: `12.4`


| Rank | Recipe | fps(bs1) | fps(bs8) | p50 ms | gpu ms | mAP@0.5 | drop | mem MB | meets? |
|-----:|--------|---------:|---------:|-------:|-------:|--------:|-----:|-------:|:------:|
| 1 | `trt_fp16` | 391.7 | 782.3 | 2.55 | — | 0.554 | -0.11%p | 38 | ✔ |
| 2 | `trt_fp32_tf32` | 203.6 | 387.6 | 4.91 | — | 0.554 | -0.11%p | 38 | ✔ |
| 3 | `ort_trt_fp16` | 202.9 | 382.2 | 4.93 | — | 0.554 | -0.11%p | — | ✔ |
| 4 | `trt_fp32` | 193.7 | 370.9 | 5.16 | — | 0.554 | -0.11%p | 38 | ✔ |
| 5 | `torchcompile_fp16` | 164.6 | 378.3 | 6.07 | — | 0.553 | -0.01%p | 415 | ✔ |
| 6 | `ort_cuda_fp16` | 68.5 | 119.3 | 14.60 | — | 0.554 | -0.11%p | — | ✔ |
| 7 | `pytorch_fp32` | 52.8 | 238.5 | 18.93 | — | 0.553 | +0.00%p | 281 | ✔ |
| 8 | `trt_int8_ptq` | 785.2 | 670.0 | 1.27 | — | 0.475 | +7.82%p | 38 | ✘ |
| 9 | `ort_int8_entropy` | 679.3 | 655.4 | 1.47 | — | 0.501 | +5.16%p | 38 | ✘ |
| 10 | `ort_int8_percentile` | 647.0 | 647.8 | 1.55 | — | 0.528 | +2.50%p | 38 | ✘ |
| 11 | `ort_int8_distribution` | 644.0 | 629.5 | 1.55 | — | 0.503 | +4.97%p | 38 | ✘ |
| 12 | `ort_int8_minmax` | 536.1 | 650.6 | 1.87 | — | 0.494 | +5.94%p | 38 | ✘ |
| 13 | `modelopt_int8_mixed` | 418.9 | 859.7 | 2.39 | — | 0.537 | +1.64%p | 38 | ✘ |
| 14 | `modelopt_int8_entropy` | 409.1 | 859.6 | 2.44 | — | 0.537 | +1.64%p | 38 | ✘ |
| 15 | `modelopt_int8_ptq` | 400.3 | 766.9 | 2.50 | — | 0.521 | +3.19%p | 38 | ✘ |
| 16 | `brevitas_int8_entropy` | 372.6 | 630.2 | 2.68 | — | 0.543 | +1.02%p | 91 | ✘ |
| 17 | `modelopt_int8_percentile` | 366.4 | 777.9 | 2.73 | — | 0.521 | +3.19%p | 38 | ✘ |
| 18 | `brevitas_int8_mse` | 362.5 | 635.9 | 2.76 | — | 0.543 | +1.02%p | 91 | ✘ |
| 19 | `brevitas_int8_percentile` | 66.5 | 629.1 | 15.05 | — | 0.543 | +1.02%p | 80 | ✘ |

## Recommendation

**`trt_fp16`** — fps 391.7 (bs1), mAP@0.5 0.554, drop -0.11%p.

## Issues
- `ort_trt_fp16`: execution_provider=TensorrtExecutionProvider, onnx=yolo26n_640_fp16_bs1.onnx
- `ort_cuda_fp16`: execution_provider=CUDAExecutionProvider, onnx=yolo26n_640_fp16_bs1.onnx
- `trt_int8_ptq`: mAP drop 7.82%p > 1.0%p
- `ort_int8_entropy`: mAP drop 5.16%p > 1.0%p
- `ort_int8_percentile`: mAP drop 2.50%p > 1.0%p
- `ort_int8_distribution`: mAP drop 4.97%p > 1.0%p
- `ort_int8_minmax`: mAP drop 5.94%p > 1.0%p
- `modelopt_int8_mixed`: mAP drop 1.64%p > 1.0%p
- `modelopt_int8_entropy`: mAP drop 1.64%p > 1.0%p
- `modelopt_int8_ptq`: mAP drop 3.19%p > 1.0%p
- `brevitas_int8_entropy`: mAP drop 1.02%p > 1.0%p
- `modelopt_int8_percentile`: mAP drop 3.19%p > 1.0%p
- `brevitas_int8_mse`: mAP drop 1.02%p > 1.0%p
- `brevitas_int8_percentile`: mAP drop 1.02%p > 1.0%p
