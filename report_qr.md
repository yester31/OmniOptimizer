# OmniOptimizer Report

Baseline: `pytorch_fp32`  |  GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`  |  CUDA: `12.9`


| Rank | Recipe | fps(bs1) | fps(bs8) | p50 ms | gpu ms | mAP@0.5 | drop | mem MB | meets? |
|-----:|--------|---------:|---------:|-------:|-------:|--------:|-----:|-------:|:------:|
| 1 | `modelopt_int8_entropy` | 763.9 | 1078.5 | 1.31 | — | 0.987 | +0.07%p | 38 | ✔ |
| 2 | `modelopt_int8_mixed` | 760.0 | 840.8 | 1.32 | — | 0.987 | +0.07%p | 38 | ✔ |
| 3 | `modelopt_int8_percentile` | 755.1 | 993.7 | 1.32 | — | 0.985 | +0.24%p | 38 | ✔ |
| 4 | `ort_int8_percentile` | 674.5 | 694.4 | 1.48 | — | 0.988 | -0.01%p | 38 | ✔ |
| 5 | `ort_int8_entropy` | 653.5 | 697.9 | 1.53 | — | 0.987 | +0.02%p | 38 | ✔ |
| 6 | `ort_int8_minmax` | 598.7 | 698.6 | 1.67 | — | 0.984 | +0.33%p | 38 | ✔ |
| 7 | `trt_int8_ptq` | 504.0 | 726.7 | 1.98 | — | 0.985 | +0.29%p | 38 | ✔ |
| 8 | `modelopt_int8_sparsity` | 439.4 | 832.9 | 2.28 | — | 0.987 | +0.07%p | 38 | ✔ |
| 9 | `trt_fp16` | 435.1 | 864.4 | 2.30 | — | 0.989 | -0.16%p | 38 | ✔ |
| 10 | `modelopt_int8_ptq` | 430.1 | 832.9 | 2.33 | — | 0.985 | +0.24%p | 38 | ✔ |
| 11 | `modelopt_int8_qat` | 425.8 | 833.2 | 2.35 | — | 0.985 | +0.24%p | 38 | ✔ |
| 12 | `ort_int8_distribution` | 402.5 | 696.7 | 2.48 | — | 0.987 | +0.02%p | 38 | ✔ |
| 13 | `brevitas_int8_percentile` | 401.2 | 679.1 | 2.49 | — | 0.988 | -0.01%p | 90 | ✔ |
| 14 | `brevitas_int8_mse` | 396.9 | 675.1 | 2.52 | — | 0.988 | -0.03%p | 91 | ✔ |
| 15 | `trt_fp32` | 356.5 | 527.1 | 2.80 | — | 0.989 | -0.17%p | 38 | ✔ |
| 16 | `trt_fp32_tf32` | 346.6 | 467.6 | 2.89 | — | 0.989 | -0.17%p | 38 | ✔ |
| 17 | `ort_trt_fp16` | 211.0 | 399.3 | 4.74 | — | 0.989 | -0.17%p | — | ✔ |
| 18 | `pytorch_fp32` | 46.5 | 213.4 | 21.50 | — | 0.988 | +0.00%p | 280 | ✔ |
| 19 | `torchcompile_fp16` | 39.0 | 312.1 | 25.62 | — | 0.988 | +0.00%p | 160 | ✔ |
| 20 | `ort_cuda_fp16` | 3.2 | — | 314.61 | — | 0.989 | -0.17%p | — | ✔ |
| 21 | `trt_int8_sparsity` | 649.2 | 726.1 | 1.54 | — | 0.973 | +1.50%p | 38 | ✘ |

## Recommendation

**`modelopt_int8_entropy`** — fps 763.9 (bs1), mAP@0.5 0.987, drop +0.07%p.

## Issues
- `ort_trt_fp16`: execution_provider=TensorrtExecutionProvider, onnx=best_qr_640_fp16_bs1.onnx
- `torchcompile_fp16`: torch.compile unavailable, fell back to eager: TritonMissing: Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at: https://github.com/triton-lang/triton

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- `ort_cuda_fp16`: execution_provider=CUDAExecutionProvider, onnx=best_qr_640_fp16_bs1.onnx
- `trt_int8_sparsity`: mAP drop 1.50%p > 1.0%p
