VENV := $(CURDIR)/.venv
PYTHON ?= $(VENV)/bin/python

TORCH_VERSION ?= 2.2.2
# GPU NVIDIA -> CUDA, else CPU
ifneq ($(shell command -v nvidia-smi 2>/dev/null),)
TORCH_INDEX ?= https://download.pytorch.org/whl/cu121
else
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
endif

export XDG_SESSION_TYPE := x11
export GDK_BACKEND := x11
export LIBGL_ALWAYS_SOFTWARE := 1

.PHONY: help install m1 m2-1 m2-2

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

install:
	python3 -m venv "$(VENV)"
	"$(VENV)/bin/pip" install --upgrade pip
	"$(VENV)/bin/pip" install --resume-retries 20 -r requirements.txt
	"$(VENV)/bin/pip" install --resume-retries 20 torch==$(TORCH_VERSION) --index-url $(TORCH_INDEX)
	"$(VENV)/bin/python" -c "import open3d.ml.torch" && echo "Installation OK"

m1: ## Module 1 BIMtoPC
	cd "ModuleOne BIMtoPC/Python" && mkdir -p resultados_dados_semanticos resultado_nuvem_global && \
		"$(PYTHON)" dados_semanticos.py && "$(PYTHON)" geometriaEnuvem.py

m2-1: ## Module 2 RandLA-Net
	cd "ModuleTwo RandlaNET/Python" && "$(PYTHON)" train_ponte.py && "$(PYTHON)" test_ponte.py

m2-2: ## Module 2 Registration
	cd "ModuleTwo Registration/Python" && "$(PYTHON)" obbp-icp.py
