# OmniOptimizer Report

Baseline: `pytorch_fp32`  |  GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`  |  CUDA: `12.9`


| Rank | Recipe | fps(bs1) | fps(bs8) | p50 ms | gpu ms | mAP@0.5 | drop | mem MB | meets? |
|-----:|--------|---------:|---------:|-------:|-------:|--------:|-----:|-------:|:------:|
| 1 | `modelopt_int8_asymmetric` | 770.5 | 1080.0 | 1.30 | — | 0.987 | +0.07%p | 38 | ✔ |
| 2 | `modelopt_int8_entropy` | 763.9 | 1078.5 | 1.31 | — | 0.987 | +0.07%p | 38 | ✔ |
| 3 | `modelopt_int8_mixed` | 760.0 | 840.8 | 1.32 | — | 0.987 | +0.07%p | 38 | ✔ |
| 4 | `modelopt_int8_percentile` | 755.1 | 993.7 | 1.32 | — | 0.985 | +0.24%p | 38 | ✔ |
| 5 | `ort_int8_percentile` | 674.5 | 694.4 | 1.48 | — | 0.988 | -0.01%p | 38 | ✔ |
| 6 | `ort_int8_entropy` | 653.5 | 697.9 | 1.53 | — | 0.987 | +0.02%p | 38 | ✔ |
| 7 | `trt_fp16_opt5` | 645.2 | 460.9 | 1.55 | — | 0.989 | -0.16%p | 38 | ✔ |
| 8 | `ort_int8_minmax` | 598.7 | 698.6 | 1.67 | — | 0.984 | +0.33%p | 38 | ✔ |
| 9 | `trt_int8_ptq` | 504.0 | 726.7 | 1.98 | — | 0.985 | +0.29%p | 38 | ✔ |
| 10 | `modelopt_int8_sparsity` | 439.4 | 832.9 | 2.28 | — | 0.987 | +0.07%p | 38 | ✔ |
| 11 | `trt_fp16` | 435.1 | 864.4 | 2.30 | — | 0.989 | -0.16%p | 38 | ✔ |
| 12 | `modelopt_int8_ptq` | 430.1 | 832.9 | 2.33 | — | 0.985 | +0.24%p | 38 | ✔ |
| 13 | `modelopt_int8_qat` | 425.8 | 833.2 | 2.35 | — | 0.985 | +0.24%p | 38 | ✔ |
| 14 | `ort_int8_distribution` | 402.5 | 696.7 | 2.48 | — | 0.987 | +0.02%p | 38 | ✔ |
| 15 | `trt_bf16` | 372.7 | 540.0 | 2.68 | — | 0.989 | -0.16%p | 38 | ✔ |
| 16 | `trt_fp32` | 356.5 | 527.1 | 2.80 | — | 0.989 | -0.17%p | 38 | ✔ |
| 17 | `trt_fp32_tf32` | 346.6 | 467.6 | 2.89 | — | 0.989 | -0.17%p | 38 | ✔ |
| 18 | `ort_trt_fp16` | 211.0 | 399.3 | 4.74 | — | 0.989 | -0.17%p | — | ✔ |
| 19 | `pytorch_fp32` | 46.5 | 213.4 | 21.50 | — | 0.988 | +0.00%p | 280 | ✔ |
| 20 | `modelopt_fastnas_int8` | 716.3 | — | 1.40 | 1.12 | 0.947 | +4.06%p | 5 | ✘ |
| 21 | `modelopt_fastnas_sp_int8` | 697.4 | — | 1.43 | 1.18 | 0.948 | +3.98%p | 5 | ✘ |
| 22 | `trt_int8_sparsity` | 649.2 | 726.1 | 1.54 | — | 0.973 | +1.50%p | 38 | ✘ |

## Recommendation

**`modelopt_int8_asymmetric`** — fps 770.5 (bs1), mAP@0.5 0.987, drop +0.07%p.

## Issues
- `ort_trt_fp16`: execution_provider=TensorrtExecutionProvider, onnx=best_qr_640_fp16_bs1.onnx
- `modelopt_fastnas_int8`: mAP drop 4.06%p > 1.0%p | Wave 10 reopened 2026-04-22. FastNAS pruning (15.7% FLOPs) pre-applied + QR fine-tuned. modelopt.onnx.quantize INT8 entropy QDQ (223 Q / 223 DQ, Detect head 42 Q/DQ pairs). Engine 4.72MB (-88% vs baseline 38MB). mAP drop vs baseline 0.988: 4.10%p. bs=8 not measured (engine built bs=1 only).
- `modelopt_fastnas_sp_int8`: mAP drop 3.98%p > 1.0%p | Wave 10 reopened 2026-04-22. FastNAS pruning (15.7% FLOPs) pre-applied + QR fine-tuned. modelopt.onnx.quantize INT8 entropy QDQ (223 Q / 223 DQ, Detect head 42 Q/DQ pairs). Engine 4.72MB (-88% vs baseline 38MB). mAP drop vs baseline 0.988: 4.02%p. bs=8 not measured (engine built bs=1 only).
- `trt_int8_sparsity`: mAP drop 1.50%p > 1.0%p
