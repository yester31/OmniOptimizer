PYTHON ?= python
RECIPES_DIR := recipes
RESULTS_DIR := results
RESULTS_CPU_DIR := results_cpu
REPORT := report.md
REPORT_CPU := report_cpu.md

.PHONY: all clean env report \
        recipe-00 recipe-00-tf32 \
        recipe-01 recipe-04 recipe-05 recipe-06 recipe-07 \
        recipe-08 recipe-09 recipe-10 recipe-11 recipe-12 \
        recipe-13 recipe-14 recipe-15 recipe-16 recipe-17 \
        recipe-23 recipe-24 \
        recipe-30 recipe-31 recipe-32 recipe-33 recipe-34 recipe-35 \
        recipe-36 recipe-37 recipe-38 \
        recipe-40 recipe-41 recipe-42 \
        cpu-all cpu-report cpu-qr fastnas-gpu fastnas-cpu \
        diagnose-recipe-%

# Re-run a single recipe:  make recipe-11
# Re-run report only:      make report
#
# recipe-07 (trt_int8_sparsity) and recipe-11 (modelopt_int8_sparsity) are
# intentionally excluded from `make all`. Post-training 2:4 sparsity collapses
# mAP on nano YOLO without sparsity-aware training; both will come back once
# the training pipeline lands. The targets below still exist so you can run
# them manually (e.g. `make recipe-11`).
all: recipe-00 recipe-00-tf32 \
     recipe-01 recipe-04 recipe-05 recipe-06 \
     recipe-08 recipe-09 recipe-10 recipe-12 \
     recipe-13 recipe-14 recipe-15 recipe-16 \
     recipe-40 recipe-41 recipe-42 \
     report

env:
	$(PYTHON) scripts/env_lock.py --out $(RESULTS_DIR)/_env.json

recipe-00:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/00_trt_fp32.yaml --out $(RESULTS_DIR)/00_trt_fp32.json

recipe-00-tf32:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/00_trt_fp32_tf32.yaml --out $(RESULTS_DIR)/00_trt_fp32_tf32.json

recipe-01:
	$(PYTHON) scripts/run_pytorch.py --recipe $(RECIPES_DIR)/01_pytorch_fp32.yaml --out $(RESULTS_DIR)/01_pytorch_fp32.json

recipe-04:
	$(PYTHON) scripts/run_ort.py --recipe $(RECIPES_DIR)/04_ort_trt_fp16.yaml --out $(RESULTS_DIR)/04_ort_trt_fp16.json

recipe-05:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/05_trt_fp16.yaml --out $(RESULTS_DIR)/05_trt_fp16.json

recipe-06:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/06_trt_int8_ptq.yaml --out $(RESULTS_DIR)/06_trt_int8_ptq.json

recipe-07:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/07_trt_int8_sparsity.yaml --out $(RESULTS_DIR)/07_trt_int8_sparsity.json

recipe-08:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/08_modelopt_int8_ptq.yaml --out $(RESULTS_DIR)/08_modelopt_int8_ptq.json

recipe-09:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/09_modelopt_int8_entropy.yaml --out $(RESULTS_DIR)/09_modelopt_int8_entropy.json

recipe-10:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/10_modelopt_int8_percentile.yaml --out $(RESULTS_DIR)/10_modelopt_int8_percentile.json

recipe-11:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/11_modelopt_int8_sparsity.yaml --out $(RESULTS_DIR)/11_modelopt_int8_sparsity.json

recipe-12:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/12_modelopt_int8_mixed.yaml --out $(RESULTS_DIR)/12_modelopt_int8_mixed.json

recipe-13:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/13_ort_int8_minmax.yaml --out $(RESULTS_DIR)/13_ort_int8_minmax.json

recipe-14:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/14_ort_int8_entropy.yaml --out $(RESULTS_DIR)/14_ort_int8_entropy.json

recipe-15:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/15_ort_int8_percentile.yaml --out $(RESULTS_DIR)/15_ort_int8_percentile.json

recipe-16:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/16_ort_int8_distribution.yaml --out $(RESULTS_DIR)/16_ort_int8_distribution.json

recipe-17:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/17_modelopt_int8_qat.yaml --out $(RESULTS_DIR)/17_modelopt_int8_qat.json

# -----------------------------------------------------------------------------
# Wave 10 GPU FastNAS pruning + INT8 (#23, #24). Require the pruned +
# fine-tuned weights at trained_weights/23_fastnas_*/weights/best.pt —
# produced by scripts/_spike_wave10_*.py. Excluded from `make all` because
# they depend on a pre-existing training artifact (like #17 QAT).
# Combined path: engine size ~5MB vs baseline 38MB (edge/embedded target).
# -----------------------------------------------------------------------------
recipe-23:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/23_modelopt_fastnas_int8.yaml --out $(RESULTS_DIR)/23_modelopt_fastnas_int8.json

recipe-24:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/24_modelopt_fastnas_sp_int8.yaml --out $(RESULTS_DIR)/24_modelopt_fastnas_sp_int8.json

fastnas-gpu: recipe-23 recipe-24

# -----------------------------------------------------------------------------
# Wave 14 TRT tuning (#40 opt_level=5, #41 bf16, #42 asymmetric INT8).
# opt_level=5 build time can be 3-5x default; expect minutes per recipe.
# -----------------------------------------------------------------------------
recipe-40:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/40_trt_fp16_opt5.yaml --out $(RESULTS_DIR)/40_trt_fp16_opt5.json

recipe-41:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/41_trt_bf16.yaml --out $(RESULTS_DIR)/41_trt_bf16.json

recipe-42:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/42_modelopt_int8_asymmetric.yaml --out $(RESULTS_DIR)/42_modelopt_int8_asymmetric.json

# Parked recipes keep their JSON on disk for history but are dropped from the
# ranking. #7/#11 need sparsity-aware training.
PARKED :=
# CPU parked list: openvino_int8_qat is a reserved #36 slot (training
# pipeline for OV not in Wave 6 scope). Present for forward compatibility.
PARKED_CPU := openvino_int8_qat

report:
	$(PYTHON) scripts/recommend.py --results-dir $(RESULTS_DIR) --out $(REPORT) --exclude "$(PARKED)"

# -----------------------------------------------------------------------------
# Wave 6 CPU recipes (#30-#35) — separate target so GPU measurement runs
# aren't polluted by CPU load on laptops where both share the host.
# -----------------------------------------------------------------------------

recipe-30:
	$(PYTHON) scripts/run_cpu.py --recipe $(RECIPES_DIR)/30_ort_cpu_fp32.yaml --out $(RESULTS_CPU_DIR)/30_ort_cpu_fp32.json

recipe-31:
	$(PYTHON) scripts/run_cpu.py --recipe $(RECIPES_DIR)/31_ort_cpu_bf16.yaml --out $(RESULTS_CPU_DIR)/31_ort_cpu_bf16.json

recipe-32:
	$(PYTHON) scripts/run_cpu.py --recipe $(RECIPES_DIR)/32_ort_cpu_int8_dynamic.yaml --out $(RESULTS_CPU_DIR)/32_ort_cpu_int8_dynamic.json

recipe-33:
	$(PYTHON) scripts/run_cpu.py --recipe $(RECIPES_DIR)/33_ort_cpu_int8_static.yaml --out $(RESULTS_CPU_DIR)/33_ort_cpu_int8_static.json

recipe-34:
	$(PYTHON) scripts/run_cpu.py --recipe $(RECIPES_DIR)/34_openvino_fp32.yaml --out $(RESULTS_CPU_DIR)/34_openvino_fp32.json

recipe-35:
	$(PYTHON) scripts/run_cpu.py --recipe $(RECIPES_DIR)/35_openvino_int8_nncf.yaml --out $(RESULTS_CPU_DIR)/35_openvino_int8_nncf.json

# -----------------------------------------------------------------------------
# Wave 10 CPU FastNAS variants (#36 OV+INT8+NNCF, #37 OV+FP32, #38 ORT+FP32).
# Same training-artifact dependency as GPU FastNAS (#23/#24) — excluded from
# `make cpu-all` until the training pipeline is part of CI.
# -----------------------------------------------------------------------------
recipe-36:
	$(PYTHON) scripts/run_cpu.py --recipe $(RECIPES_DIR)/36_openvino_fastnas_int8_nncf.yaml --out $(RESULTS_CPU_DIR)/36_openvino_fastnas_int8_nncf.json

recipe-37:
	$(PYTHON) scripts/run_cpu.py --recipe $(RECIPES_DIR)/37_openvino_fastnas_fp32.yaml --out $(RESULTS_CPU_DIR)/37_openvino_fastnas_fp32.json

recipe-38:
	$(PYTHON) scripts/run_cpu.py --recipe $(RECIPES_DIR)/38_ort_cpu_fastnas_fp32.yaml --out $(RESULTS_CPU_DIR)/38_ort_cpu_fastnas_fp32.json

fastnas-cpu: recipe-36 recipe-37 recipe-38

# Full CPU bank + report. Does NOT include `recipe-31` automatically if
# the host lacks BF16 ISA (the recipe self-skips with a notes entry).
cpu-all: recipe-30 recipe-31 recipe-32 recipe-33 recipe-34 recipe-35 cpu-report

cpu-report:
	$(PYTHON) scripts/recommend.py --results-dir $(RESULTS_CPU_DIR) --out $(REPORT_CPU) --exclude "$(PARKED_CPU)"

# CPU bank against the QR-fine-tuned checkpoint (mirrors GPU QR batch flow).
cpu-qr:
	bash scripts/run_cpu_batch.sh

# Accuracy / NaN / precision diagnostics via polygraphy.
# Usage:  make diagnose-recipe-06   (runs both --validate and --debug-precision)
# Requires: pip install polygraphy, and an ONNX already built in results/_onnx.
diagnose-recipe-%:
	mkdir -p $(RESULTS_DIR)/_diag
	@ONNX=$$(ls $(RESULTS_DIR)/_onnx/*.onnx 2>/dev/null | grep -E "(modelopt|fp32).*$*" | head -1); \
	if [ -z "$$ONNX" ]; then echo "no onnx matching '$*' in $(RESULTS_DIR)/_onnx/"; exit 1; fi; \
	echo "Diagnosing $$ONNX"; \
	polygraphy run $$ONNX --validate --log-file $(RESULTS_DIR)/_diag/recipe$*.validate.log || true; \
	polygraphy run $$ONNX --trt --debug-precision --log-file $(RESULTS_DIR)/_diag/recipe$*.debug_precision.log || true; \
	echo "logs in $(RESULTS_DIR)/_diag/"

train-qr:
	bash scripts/run_qr_train_batch.sh

.PHONY: train-qr

clean:
	rm -rf $(RESULTS_DIR)/*.json $(REPORT) *.engine *.onnx build/ dist/ *.egg-info/
