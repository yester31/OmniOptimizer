PYTHON ?= python
RECIPES_DIR := recipes
RESULTS_DIR := results
REPORT := report.md

.PHONY: all clean env report \
        recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 recipe-07

all: recipe-01 recipe-02 recipe-03 recipe-04 recipe-05 recipe-06 recipe-07 report

env:
	$(PYTHON) scripts/env_lock.py --out $(RESULTS_DIR)/_env.json

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

report:
	$(PYTHON) scripts/recommend.py --results-dir $(RESULTS_DIR) --out $(REPORT)

clean:
	rm -rf $(RESULTS_DIR)/*.json $(REPORT) *.engine *.onnx build/ dist/ *.egg-info/
