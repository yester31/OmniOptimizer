# OmniOptimizer Report


| Rank | Recipe | fps(bs1) | fps(bs8) | p50 ms | gpu ms | mAP@0.5 | drop | mem MB | meets? |
|-----:|--------|---------:|---------:|-------:|-------:|--------:|-----:|-------:|:------:|
| 1 | `openvino_fastnas_int8_nncf` | 32.2 | 28.0 | 31.08 | — | 0.947 | — | — | ✔ |
| 2 | `openvino_int8_nncf` | 23.9 | 22.9 | 41.85 | — | 0.988 | — | — | ✔ |
| 3 | `openvino_fastnas_fp32` | 23.1 | 23.6 | 43.24 | — | 0.947 | — | — | ✔ |
| 4 | `openvino_fp32` | 18.6 | 17.3 | 53.86 | — | 0.988 | — | — | ✔ |
| 5 | `ort_cpu_fp32` | 14.4 | 13.7 | 69.46 | — | 0.988 | — | — | ✔ |
| 6 | `ort_cpu_fastnas_fp32` | 14.2 | 14.7 | 70.48 | — | 0.947 | — | — | ✔ |
| 7 | `ort_cpu_int8_dynamic` | 10.0 | 8.0 | 99.76 | — | 0.982 | — | — | ✔ |
| 8 | `ort_cpu_int8_static` | 6.2 | 4.7 | 160.51 | — | 0.000 | — | — | ✔ |
| 9 | `ort_cpu_bf16` | — | — | — | — | — | — | — | ✘ |

## Recommendation

**`openvino_fastnas_int8_nncf`** — fps 32.2 (bs1), mAP@0.5 0.947, drop —.

## Issues
- `ort_cpu_bf16`: missing measurements | ort_cpu + bf16: host CPU lacks BF16 ISA (need amx_tile or avx512_bf16; saw flags=['avx', 'avx2', 'avx512_vnni', 'avx512bw', 'avx512cd', 'avx512dq', 'avx512f', 'avx512vl', 'fma', 'sse4_1', 'sse4_2']). Recipe skipped; Result.meets_constraints=False.
