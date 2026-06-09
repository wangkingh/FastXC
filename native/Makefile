# Top-level build entry for FastXC CUDA/C backends.

BUILD ?= release
include common.mk
.DEFAULT_GOAL := all

MODE ?= par
ifeq ($(filter $(MODE),par seq),)
  $(error MODE must be par or seq, got '$(MODE)')
endif

ifeq ($(MODE),par)
  PARFLAG = -j
else
  PARFLAG =
endif

ifeq ($(shell expr $(firstword $(subst ., ,$(MAKE_VERSION))) \>= 4),1)
  OSYNC ?= --output-sync=target
else
  OSYNC :=
endif

BACKEND_DIRS := sac2spec xc pws tfpws
BUILD_DIRS := $(BACKEND_DIRS)
CLEAN_DIRS := $(BACKEND_DIRS)

.PHONY: all debug recurse clean veryclean print-config help $(BACKEND_DIRS)

all: cuda-preflight ## Build all native backends in release mode by default
	@printf "\033[32m=== FastXC CUDA build: BUILD=%s MODE=%s ARCH=%s ===\033[0m\n" "$(BUILD)" "$(MODE)" "$(CUDA_SM)"
	$(MAKE) $(PARFLAG) $(OSYNC) BUILD=$(BUILD) recurse

debug: BUILD=debug
debug: cuda-preflight ## Build all native backends in debug mode
	@printf "\033[33m=== FastXC CUDA debug build: MODE=%s ARCH=%s ===\033[0m\n" "$(MODE)" "$(CUDA_SM)"
	$(MAKE) $(PARFLAG) $(OSYNC) BUILD=$(BUILD) recurse

recurse: $(BUILD_DIRS)

$(BACKEND_DIRS):
	@printf "\033[35m========== [ %s ] =========\033[0m\n" "$@"
	$(MAKE) -C "$@" BUILD=$(BUILD) all

clean: ## Remove object files from all CUDA backend directories
	@for dir in $(CLEAN_DIRS); do \
	  if [ -d "$$dir" ]; then \
	    printf "\033[34m>> Cleaning in %s\033[0m\n" "$$dir"; \
	    $(MAKE) -C "$$dir" clean; \
	  fi; \
	done

veryclean: ## Remove object files and generated binaries
	@for dir in $(CLEAN_DIRS); do \
	  if [ -d "$$dir" ]; then \
	    printf "\033[31m>> Very clean in %s\033[0m\n" "$$dir"; \
	    $(MAKE) -C "$$dir" veryclean; \
	  fi; \
	done

print-config: ## Show compiler and CUDA architecture settings
	@printf "BUILD=%s\n" "$(BUILD)"
	@printf "NVCC=%s\n" "$(NVCC)"
	@printf "CUDA_HOME=%s\n" "$(CUDA_HOME)"
	@printf "CUDA_INCLUDEDIR=%s\n" "$(CUDA_INCLUDEDIR)"
	@printf "CUDA_LIBDIR=%s\n" "$(CUDA_LIBDIR)"
	@printf "CUDA_STD=%s\n" "$(CUDA_STD)"
	@printf "NVCC_CCBIN=%s\n" "$(NVCC_CCBIN)"
	@printf "ARCH_REQUEST=%s\n" "$(ARCH)"
	@printf "DETECTED_ARCHS=%s\n" "$(DETECTED_ARCHS)"
	@printf "CUDA_SM=%s\n" "$(CUDA_SM)"
	@printf "CUDA_COMPUTE=%s\n" "$(CUDA_COMPUTE)"
	@printf "NVCC_ARCH_FLAGS=%s\n" "$(NVCC_ARCH_FLAGS)"

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage: make \033[36m<TARGET>\033[0m [MODE=par|seq] [BUILD=release|debug] [ARCH=auto|sm_89|sm89|8.9]\n\nTargets:\n"} \
	     /^[a-zA-Z0-9_-]+:[^#]*##/{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)
