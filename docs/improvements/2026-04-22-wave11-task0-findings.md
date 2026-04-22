# Wave 11 Task 0 findings — 2026-04-22

**Status**: **RESOLVED 2026-04-22** — TRT EP DLL blocker 수복 완료. `scripts/run_ort.py::_add_tensorrt_dll_dir()` 추가로 `TensorrtExecutionProvider` primary 로드 확인. pytest 96 passed, 0 regression. Wave 11 Task 3 진입 가능 상태.

**원 Status (archived below)**: Task 0 partial — 0.2 / 0.3 / 0.4 완료, 0.1 (recipe 재실행) 보류. STOP 사유: ORT 1.22.0 Windows 환경에서 `TensorrtExecutionProvider` 실제 로드 실패 확인.

## 환경 snapshot (Task 0.2 / 0.3)

| 항목 | 값 |
|---|---|
| torch | 2.8.0+cu129 |
| torch.version.cuda | 12.9 |
| torch.backends.cudnn | 9.10.2 (91002) |
| tensorrt | 10.16.0.72 |
| onnxruntime | 1.22.0 |
| `ort.get_device()` | GPU |
| `ort.get_available_providers()` | `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider` |
| NVIDIA driver | 576.80 |
| GPU | RTX 3060 Laptop, 6144 MiB, 5996 MiB free |

## Task 0.4 ORT provider options 키명 검증

ORT 1.22 에서 `TensorrtExecutionProvider` provider_options 로 다음 키명 **accept** (unknown-key 경고 없음):
- `trt_fp16_enable`
- `trt_engine_cache_enable`
- `trt_engine_cache_path`

Wave 11 Task 3 (B3 ort_trt_fp16 개선) 에 사용될 키명 3개 전부 유효. CLAUDE.md §1 convention 충족.

## 핵심 발견 — TRT EP DLL 로드 실패

`ort.InferenceSession(providers=[("TensorrtExecutionProvider", {...}), ...])` 호출 시:

```
[E:onnxruntime:Default, provider_bridge_ort.cc:2167]
Error loading "...\onnxruntime\capi\onnxruntime_providers_tensorrt.dll"
which depends on "nvinfer_10.dll" which is missing. (Error 126)

EP Error ... Please install TensorRT libraries as mentioned in the GPU
requirements page, make sure they're in the PATH or LD_LIBRARY_PATH

Falling back to ['CPUExecutionProvider'] and retrying.
```

**해석**:
1. TRT 10.16 은 Python 에서 `import tensorrt` 정상 — torch/pycuda 경로와는 연결됨
2. ORT 의 native DLL (`onnxruntime_providers_tensorrt.dll`) 이 `nvinfer_10.dll` 을 찾지 못함 — system PATH 에 TRT bin 경로 누락
3. ORT 는 silent fallback 해서 CPU EP 로 작동 — 사용자는 "TRT EP 가 느리다" 고 오해

**API 주의사항 (향후 세션용 기록)**:
`ort.get_available_providers()` 가 반환하는 리스트는 **stub 검증**만 수행 (provider DLL 이 설치는 되어 있는지). 실제 DLL dependency 해석 가능 여부는 session 생성 시에만 확인됨. provider 선택 로직에서 `get_available_providers()` 만 의존하면 silent fallback 에 노출.

## Wave 11 B-series 영향 재평가

| ID | Recipe | Task 0 영향 | 조치 |
|---|---|---|---|
| B1 | #21 ort_cuda_fp16 | CUDAExecutionProvider 는 정상 로드 확인 → 가설 ("CUDA EP 초기화 실패") 부분 기각. 다른 원인 (cuDNN mismatch, memory allocator, dynamic shape) 필요 | Task 1 계속 진행 가능 |
| B2 | #20 torchcompile_fp16 | 무영향 (ORT path 아님) | Task 2 계속 가능 |
| B3 | #18 ort_trt_fp16 | **TRT EP 로드 자체가 실패 — 이 환경에선 debug 불가능** | **선행 조치: TRT bin PATH 수복 필요** (별도 env 작업) |
| B4 | #13 modelopt_int8_ptq | 무영향 (native TRT 경로) | Task 4 계속 가능 |
| B5 | #33 ort_cpu_int8_static | 무영향 (CPU EP) | Task 5 계속 가능 |

## TRT EP DLL 수복 경로 (B3 선행 조건)

1. TRT wheel 경로 확인: `pip show tensorrt` 로 설치 경로 + `TRT_ROOT\lib` 에 `nvinfer_10.dll` 존재 확인
2. system PATH 에 추가 (Windows):
   ```
   setx PATH "%PATH%;<TRT_ROOT>\lib"
   ```
   또는 Python 런타임 시 `os.add_dll_directory(r"<TRT_ROOT>\lib")` 삽입 — `scripts/run_ort.py` 초기화 단계에.
3. session 재생성 후 `sess.get_providers()[0] == "TensorrtExecutionProvider"` 확인

이 작업은 Wave 11 scope 에 선행 조건으로 추가하거나, B3 를 별도 Wave 로 이관.

## 결정 — Wave 11 plan 수정 필요

- Task 3 (B3) 에 precondition 명시: "TRT EP DLL path 수복 선행 — 이 작업이 완료되기 전에는 B3 debug 결과 신뢰 불가"
- Task 0 에 "0.5 TRT EP session smoke" step 추가 — provider 실제 로드 검증
- `get_available_providers()` misleading 패턴을 CLAUDE.md "Critical conventions" 에 추가 고려 (재발 방지)

## 진행 분기 (다음 세션용)

- (a) TRT EP DLL path 먼저 복구 → B3 debug 재개
- (b) B3 를 Wave 11 scope 에서 제외하고 별도 Wave (환경 수복 + B3 재개) 로 이관 → 이번 Wave 11 은 B1/B2/B4/B5 만 진행
- (c) B3 를 archive 결정 — recipe #18 드롭, `recipes/_archived/` 이동

## 참고

- Wave 11 plan — `docs/plans/2026-04-22-wave11-recipe-debug-cleanup.md`
- ORT TRT EP 요구사항 — <https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements>
- TRT Windows installation — <https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-windows>
