PYTHON ?= python
RECIPES_DIR := recipes
RESULTS_DIR := results
REPORT := report.md

.PHONY: all clean env report \
        recipe-00 recipe-00-tf32 \
        recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 recipe-07 \
        recipe-08 recipe-09 recipe-10 recipe-11 recipe-12 \
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
     recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 \
     recipe-08 recipe-09 recipe-10 recipe-12 report

env:
	$(PYTHON) scripts/env_lock.py --out $(RESULTS_DIR)/_env.json

recipe-00:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/00_trt_fp32.yaml --out $(RESULTS_DIR)/00_trt_fp32.json

recipe-00-tf32:
	$(PYTHON) scripts/run_trt.py --recipe $(RECIPES_DIR)/00_trt_fp32_tf32.yaml --out $(RESULTS_DIR)/00_trt_fp32_tf32.json

recipe-01:
	$(PYTHON) scripts/run_pytorch.py --recipe $(RECIPES_DIR)/01_pytorch_fp32.yaml --out $(RESULTS_DIR)/01_pytorch_fp32.json

recipe-02:
	$(PYTHON) scripts/run_pytorch.py --recipe $(RECIPES_DIR)/02_torchcompile_fp16.yaml --out $(RESULTS_DIR)/02_torchcompile_fp16.json

recipe-03:
	$(PYTHON) scripts/run_ort.py --recipe $(RECIPES_DIR)/03_ort_cuda_fp16.yaml --out $(RESULTS_DIR)/03_ort_cuda_fp16.json

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

# Sparsity recipes are parked until training code lands; their JSON files stay
# on disk for history but the report drops them from the ranking via --exclude.
PARKED := trt_int8_sparsity,modelopt_int8_sparsity

report:
	$(PYTHON) scripts/recommend.py --results-dir $(RESULTS_DIR) --out $(REPORT) --exclude "$(PARKED)"

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

clean:
	rm -rf $(RESULTS_DIR)/*.json $(REPORT) *.engine *.onnx build/ dist/ *.egg-info/
