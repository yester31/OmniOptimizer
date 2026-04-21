from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    family: str
    variant: str
    weights: str
    ultralytics_version: Optional[str] = None


class RuntimeSpec(BaseModel):
    engine: Literal["pytorch", "onnxruntime", "tensorrt", "openvino"]
    mode: Optional[str] = None
    execution_provider: Optional[str] = None
    version: Optional[str] = None
    # Wave 6 adds "bf16" for ORT CPU EP on SPR+ (AMX) / Tiger Lake+ (AVX-512 BF16).
    dtype: Literal["fp32", "fp16", "bf16", "int8"]
    sparsity: Optional[str] = None


class TechniqueSpec(BaseModel):
    name: str
    # Where quantization / sparsity logic comes from.
    # v1 uses TensorRT's built-in calibrator + SPARSE_WEIGHTS flag.
    # v1.1+ plans to add "modelopt" (nvidia-modelopt: torch-level quantization
    # + QDQ-ONNX export) and possibly "ort_quant" for ONNX Runtime's quantizer.
    source: Literal[
        "trt_builtin", "modelopt", "ort_quant", "brevitas",
        # Wave 6: CPU backends. ort_cpu uses onnxruntime CPUExecutionProvider;
        # openvino uses Intel OpenVINO runtime directly (NNCF PTQ for INT8).
        "ort_cpu", "openvino",
    ] = "trt_builtin"
    calibrator: Optional[str] = None
    calibration_samples: Optional[int] = None
    calibration_dataset: Optional[str] = None
    calibration_seed: Optional[int] = None
    # v1.2: when set, apply structured weight pruning *before* QDQ injection
    # so the engine builder can pick real 2:4 sparse INT8 kernels. The plain
    # runtime.sparsity flag alone only sets SPARSE_WEIGHTS, which is a no-op
    # unless the weights actually have the 2:4 pattern.
    sparsity_preprocess: Optional[Literal["2:4"]] = None
    # v1.2: ONNX node names to leave at FP16 during modelopt.onnx quantize.
    # Protects sensitivity-critical layers (stem Conv, detect head branches).
    nodes_to_exclude: Optional[list[str]] = None
    # v1.3: fine-tune before quantize (QAT / sparsity recovery). None for
    # PTQ-only recipes. Drives scripts/train.py; see TrainingSpec.
    training: Optional["TrainingSpec"] = None


class TrainingSpec(BaseModel):
    """Fine-tuning recipe for QAT / sparsity modifiers.

    Appears under ``TechniqueSpec.training`` and activates the
    ``scripts/train.py`` entry point. Absent for non-training recipes.
    """
    base_checkpoint: str
    epochs: int
    batch: int = 8
    workers: int = 4
    imgsz: int = 640
    lr0: float = 0.001
    optimizer: str = "AdamW"
    seed: int = 42
    data_yaml: Optional[str] = None
    modifier: Literal["prune_24", "modelopt_sparsify", "modelopt_qat"]
    prune_amount: Optional[float] = None
    quant_config: Optional[str] = "int8_default"


TechniqueSpec.model_rebuild()


class HardwareSpec(BaseModel):
    gpu: Optional[str] = None
    cuda: Optional[str] = None
    driver: Optional[str] = None
    requires_compute_capability_min: Optional[float] = None
    # Wave 6: CPU recipes populate these; GPU recipes leave None. Used by
    # env_lock.py for reproducibility and by recommend.py to distinguish
    # results from different CPU tiers (e.g., Xeon 8480+ vs i9-13900K).
    cpu_model: Optional[str] = None
    cpu_cores_physical: Optional[int] = None
    cpu_flags: Optional[list[str]] = None
    numa_node: Optional[int] = None
    governor: Optional[str] = None


class MeasurementSpec(BaseModel):
    dataset: str
    num_images: int
    warmup_iters: int
    measure_iters: int
    batch_sizes: list[int]
    input_size: int = 640
    gpu_clock_lock: bool = True
    seed: int = 42
    # Wave 6: explicit CPU thread count. None → auto-detect physical cores.
    # Zero forbidden because ORT treats 0 as "all logical cores including
    # hyperthreads," which typically regresses perf vs physical-core pinning.
    # See docs/plans/2026-04-21-wave6-cpu-inference.md Task 6 Step 3.
    thread_count: Optional[int] = Field(default=None, gt=0)


class ConstraintSpec(BaseModel):
    max_map_drop_pct: Optional[float] = None
    min_fps_bs1: Optional[float] = None


class Recipe(BaseModel):
    name: str
    model: ModelSpec
    runtime: RuntimeSpec
    technique: TechniqueSpec
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    measurement: MeasurementSpec
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)


class LatencyStats(BaseModel):
    # Wall-clock (perf_counter + cuda.synchronize) percentiles — primary
    # metric, captures end-to-end user-visible latency incl. launch overhead.
    # Optional only so failed-build recipes (engine parse error, OOM, etc.)
    # can still serialize a Result JSON with p50=None and propagate the
    # failure reason through Result.notes. Successful runs MUST populate
    # these; recommend.py treats missing p50 as "no valid measurement".
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    # CUDA Event based percentiles — isolates on-GPU execution time from
    # Python / TRT enqueue overhead. Populated by measure_latency when
    # CUDA + torch are available; remains None otherwise.
    p50_gpu: Optional[float] = None
    p95_gpu: Optional[float] = None
    p99_gpu: Optional[float] = None


class ThroughputStats(BaseModel):
    bs1: Optional[float] = None
    bs8: Optional[float] = None


class AccuracyStats(BaseModel):
    map_50: Optional[float] = None
    map_50_95: Optional[float] = None


class EnvInfo(BaseModel):
    gpu: Optional[str] = None
    gpu_compute_capability: Optional[str] = None
    cuda: Optional[str] = None
    cudnn: Optional[str] = None
    driver: Optional[str] = None
    tensorrt: Optional[str] = None
    os: Optional[str] = None
    python: Optional[str] = None
    torch: Optional[str] = None
    onnxruntime: Optional[str] = None
    ultralytics: Optional[str] = None
    # Wave 6: CPU run captures these; GPU runs leave None. Historical GPU
    # result JSONs stay valid (all-None defaults).
    cpu_model: Optional[str] = None
    cpu_cores_physical: Optional[int] = None
    cpu_flags: Optional[list[str]] = None
    openvino: Optional[str] = None


class Result(BaseModel):
    recipe: str
    started_at: str
    finished_at: str
    env: EnvInfo
    model_size_mb: Optional[float] = None
    latency_ms: LatencyStats
    throughput_fps: ThroughputStats = Field(default_factory=ThroughputStats)
    peak_gpu_mem_mb: Optional[float] = None
    # NVML process-memory delta (baseline -> peak during measurement).
    # Captures allocations from TRT's own allocator that torch's caching
    # allocator misses (`peak_gpu_mem_mb` above is torch-only). Optional
    # for backward compatibility with historical JSONs.
    peak_gpu_mem_mb_nvml_delta: Optional[float] = None
    cold_start_ms: Optional[float] = None
    accuracy: AccuracyStats = Field(default_factory=AccuracyStats)
    meets_constraints: Optional[bool] = None
    notes: Optional[str] = None


def load_recipe(path: str) -> Recipe:
    import os
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Env override so one recipe bank can evaluate multiple checkpoints
    # (e.g., generic yolo26n.pt vs a fine-tuned best.pt). Keeps recipe files
    # as the canonical default while letting batch runs swap weights without
    # editing 21 YAMLs.
    weights_override = os.environ.get("OMNI_WEIGHTS_OVERRIDE")
    if weights_override:
        data.setdefault("model", {})["weights"] = weights_override
    return Recipe.model_validate(data)
