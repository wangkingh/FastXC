# Shared CUDA build settings for FastXC native backends.

CUDA_SRC_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

CC   ?= gcc
NVCC ?= $(shell command -v nvcc 2>/dev/null || \
          command -v /usr/local/cuda/bin/nvcc 2>/dev/null || \
          command -v /usr/local/cuda-12.6/bin/nvcc 2>/dev/null)

CUDA_HOME   ?= $(patsubst %/bin/nvcc,%,$(NVCC))
CUDA_LIBDIR ?= $(CUDA_HOME)/lib64

WARN     ?= -Wall
BITS64   ?= -D_FILE_OFFSET_BITS=64
CINCLUDE ?= -I$(CUDA_HOME)/include

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

normalize_sm = $(if $(filter sm_%,$(1)),$(1),sm_$(1))

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
NVCC_ARCH_FLAGS := -gencode arch=compute_$(call sm_version,$(CUDA_SM)),code=$(CUDA_SM)

ifeq ($(BUILD),debug)
  OPT_CFLAGS    ?= -O0 -g
  OPT_NVCCFLAGS ?= -O0 -g -G -lineinfo
else
  OPT_CFLAGS    ?= -O3
  OPT_NVCCFLAGS ?= -O3 --use_fast_math --generate-line-info
endif
