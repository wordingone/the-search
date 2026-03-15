# FluxCore Entity — CUDA Makefile

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++14
ARCH = -arch=sm_70

# Detect architecture if possible
ifneq ($(shell which nvidia-smi 2>/dev/null),)
    GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    ifneq ($(GPU_ARCH),)
        ARCH = -arch=sm_$(GPU_ARCH)
    endif
endif

TARGETS = fluxcore_entity fluxcore_test

.PHONY: all clean test

all: $(TARGETS)

fluxcore_entity: fluxcore_entity_impl.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $< -o $@

fluxcore_test: fluxcore_true.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $< -o $@

test: fluxcore_entity
	./fluxcore_entity

clean:
	rm -f $(TARGETS)

# Help
help:
	@echo "FluxCore Entity — Build Options"
	@echo "================================"
	@echo "make all          - Build all targets"
	@echo "make fluxcore_entity - Build entity version (3 leaps)"
	@echo "make fluxcore_test   - Build baseline version"
	@echo "make test         - Run entity demonstration"
	@echo "make clean        - Remove built files"
	@echo ""
	@echo "Detected GPU arch: $(ARCH)"
