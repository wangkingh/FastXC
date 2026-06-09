# PWS-only stack backend.

include ../common.mk
.DEFAULT_GOAL := all

INCLUDES := -I. -Iinclude -Iio -Ikernels -Iworker
PROJECT_CPPFLAGS := $(INCLUDES) $(CUDA_REQUIRED_CPPFLAGS)
NVCC_HOST_FLAGS := -Xcompiler "$(BITS64),-pthread"

CFLAGS    ?= $(OPT_CFLAGS) $(WARN)
NVCCFLAGS ?= $(OPT_NVCCFLAGS)

LDFLAGS = $(CUDA_LDFLAGS) -lcufft -lcudart -lm -lpthread

CUDAOBJS = \
    cuda.util.o \
    cuda.main.o \
    io/sourcepack_io.o \
    kernels/cuda.pws.o \
    worker/host_stage.o \
    worker/gpu.o \
    worker/sourcepack_pipeline.o

OBJS = \
    arguproc.o \
    logger.o \
    io/path_util.o

BINDIR := ../../bin
PROG   := $(BINDIR)/ncf_pws

.PHONY: all clean veryclean

all: $(BINDIR) $(PROG)

$(BINDIR):
	@mkdir -p $@

$(PROG): $(OBJS) $(CUDAOBJS)
	$(NVCC) $(PROJECT_CPPFLAGS) $(NVCCFLAGS) $(CUDA_REQUIRED_NVCCFLAGS) $(NVCC_HOST_FLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(PROJECT_CPPFLAGS) $(CFLAGS) -pthread $(BITS64) -c $< -o $@

%.o: %.cu
	$(NVCC) $(PROJECT_CPPFLAGS) $(NVCCFLAGS) $(CUDA_REQUIRED_NVCCFLAGS) $(NVCC_HOST_FLAGS) -c $< -o $@

cuda.main.o: worker/gpu_config.hpp worker/sourcepack_pipeline.hpp include/logger.h
arguproc.o: include/logger.h
logger.o: include/logger.h
io/sourcepack_io.o: io/sourcepack_io.hpp worker/host_stage.hpp io/sac_output.hpp io/path_util.h include/types.hpp include/logger.h
worker/host_stage.o: worker/host_stage.hpp include/concurrency.hpp include/types.hpp
worker/gpu.o: worker/gpu.hpp worker/host_stage.hpp include/concurrency.hpp include/types.hpp io/sac_output.hpp kernels/cuda.pws.cuh include/cuda.util.cuh include/cuda_resources.hpp io/path_util.h include/logger.h
worker/sourcepack_pipeline.o: worker/sourcepack_pipeline.hpp io/sourcepack_io.hpp worker/gpu.hpp worker/gpu_budget.hpp worker/host_stage.hpp worker/memory_estimate.hpp include/concurrency.hpp include/progress_sidecar.hpp include/cuda.util.cuh arguproc.h include/logger.h
kernels/cuda.pws.o: kernels/cuda.pws.cuh
cuda.util.o: include/cuda.util.cuh include/logger.h

clean:
	@rm -f *.o io/*.o kernels/*.o worker/*.o

veryclean: clean
	@rm -f $(PROG)
