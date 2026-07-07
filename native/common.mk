# Shared CUDA build settings for FastXC native backends.

CUDA_SRC_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

CC   ?= gcc
CXX  ?= g++
NVCC ?= $(shell command -v nvcc 2>/dev/null || \
          command -v /usr/local/cuda/bin/nvcc 2>/dev/null || \
          command -v /usr/local/cuda-12.6/bin/nvcc 2>/dev/null)

CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(NVCC))
CUDA_TARGET_TRIPLET ?= x86_64-linux
CUDA_TARGET_ROOT := $(CUDA_HOME)/targets/$(CUDA_TARGET_TRIPLET)
CUDA_TARGET_DIR ?= $(if $(wildcard $(CUDA_TARGET_ROOT)/include),$(CUDA_TARGET_ROOT),$(CUDA_HOME))
CUDA_INCLUDEDIR ?= $(CUDA_TARGET_DIR)/include
CUDA_LIBDIR ?= $(or $(firstword $(wildcard $(CUDA_TARGET_DIR)/lib $(CUDA_HOME)/lib64 $(CUDA_HOME)/lib)),$(CUDA_HOME)/lib64)

WARN     ?= -Wall
BITS64   ?= -D_FILE_OFFSET_BITS=64
CINCLUDE ?= -I$(CUDA_INCLUDEDIR)
CUDA_STD ?= c++17

ifeq ($(origin NVCC_CCBIN),undefined)
  ifneq ($(filter command line environment environment override,$(origin CXX)),)
    NVCC_CCBIN := $(CXX)
  else
    NVCC_CCBIN :=
  endif
endif
NVCC_CCBIN_FLAG = $(if $(strip $(NVCC_CCBIN)),-ccbin $(NVCC_CCBIN),)

# Direct subdirectory builds default to debug. The top-level Makefile overrides
# this to release unless BUILD=debug is requested.
BUILD ?= $(if $(filter 0,$(MAKELEVEL)),debug,release)

ifeq ($(filter $(BUILD),release debug),)
  $(error BUILD must be release or debug, got '$(BUILD)')
endif

# Backward-compatible override:
#   make ARCH=sm_89
# Automatic mode tries tools/check_gpu first, then a direct
# nvidia-smi query, then falls back to CUDA_ARCH_FALLBACK.
ARCH ?= auto
CUDA_ARCH_FALLBACK ?= sm_80
CUDA_ARCH_DETECTOR ?= $(CUDA_SRC_ROOT)/detect_cuda_archs.sh
CHECK_GPU ?= $(CUDA_SRC_ROOT)/../tools/check_gpu

normalize_sm = $(shell printf '%s\n' '$(1)' | awk '{ arch=$$0; gsub(/^[[:space:]]+|[[:space:]]+$$/, "", arch); sub(/^(sm_|sm|compute_|compute)/, "", arch); if (arch ~ /^[0-9]+[.][0-9]+$$/) { split(arch, parts, "."); printf "sm_%s%s", parts[1], parts[2] } else { printf "sm_%s", arch } }')

ifeq ($(strip $(ARCH)),auto)
  DETECTED_ARCHS := $(strip $(shell \
    CUDA_ARCH_FALLBACK="$(CUDA_ARCH_FALLBACK)" \
    CHECK_GPU="$(CHECK_GPU)" \
    $(SHELL) "$(CUDA_ARCH_DETECTOR)" 2>/dev/null))
  CUDA_SM := $(if $(DETECTED_ARCHS),$(firstword $(DETECTED_ARCHS)),$(CUDA_ARCH_FALLBACK))
else
  DETECTED_ARCHS :=
  CUDA_SM := $(call normalize_sm,$(ARCH))
endif

sm_version = $(patsubst sm_%,%,$(1))
CUDA_COMPUTE := compute_$(call sm_version,$(CUDA_SM))
NVCC_ARCH_FLAGS := -gencode arch=$(CUDA_COMPUTE),code=$(CUDA_SM)
CUDA_REQUIRED_CPPFLAGS := $(CINCLUDE)
CUDA_REQUIRED_NVCCFLAGS := -std=$(CUDA_STD) $(NVCC_ARCH_FLAGS) $(NVCC_CCBIN_FLAG)
CUDA_LDFLAGS := -L$(CUDA_LIBDIR)
CUDA_CUFFT_HEADER := $(CUDA_INCLUDEDIR)/cufft.h
CUDA_CUFFT_LIB := $(firstword $(wildcard $(CUDA_LIBDIR)/libcufft.so $(CUDA_LIBDIR)/libcufft_static.a))
CUDA_CUDART_LIB := $(firstword $(wildcard $(CUDA_LIBDIR)/libcudart.so $(CUDA_LIBDIR)/libcudart_static.a))

ifeq ($(BUILD),debug)
  OPT_CFLAGS    ?= -O0 -g
  OPT_NVCCFLAGS ?= -O0 -g -G
else
  OPT_CFLAGS    ?= -O3
  OPT_NVCCFLAGS ?= -O3 --use_fast_math --generate-line-info
endif

.PHONY: cuda-preflight

cuda-preflight:
	@{ test -n "$(NVCC)" && test -x "$(NVCC)"; } || { \
	  echo "ERROR: nvcc not found. Put nvcc on PATH or set NVCC=/path/to/cuda/bin/nvcc."; \
	  exit 1; \
	}
	@if "$(NVCC)" --list-gpu-arch >/dev/null 2>&1; then \
	  "$(NVCC)" --list-gpu-arch | grep -qx "$(CUDA_COMPUTE)" || { \
	    echo "ERROR: $(NVCC) does not support $(CUDA_COMPUTE) ($(CUDA_SM))."; \
	    echo "Use a newer CUDA Toolkit for this GPU, or set ARCH to a supported architecture."; \
	    exit 1; \
	  }; \
	fi
	@test -f "$(CUDA_CUFFT_HEADER)" || { \
	  echo "ERROR: cuFFT header not found: $(CUDA_CUFFT_HEADER)"; \
	  echo "If using conda CUDA, install a complete CUDA Toolkit or set CUDA_HOME/NVCC to the toolkit root."; \
	  exit 1; \
	}
	@test -n "$(CUDA_CUFFT_LIB)" || { \
	  echo "ERROR: cuFFT library not found in $(CUDA_LIBDIR)"; \
	  echo "If using conda CUDA, install a complete CUDA Toolkit or set CUDA_HOME/NVCC to the toolkit root."; \
	  exit 1; \
	}
	@test -n "$(CUDA_CUDART_LIB)" || { \
	  echo "ERROR: CUDA runtime library not found in $(CUDA_LIBDIR)"; \
	  echo "If using conda CUDA, install a complete CUDA Toolkit or set CUDA_HOME/NVCC to the toolkit root."; \
	  exit 1; \
	}
