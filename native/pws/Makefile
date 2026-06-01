# PWS-only stack backend.

include ../common.mk

CUDA_STD ?= c++11

INCLUDES := -I. -Iinclude -Iio -Ikernels -Iworker

CFLAGS    ?= $(OPT_CFLAGS) $(WARN) $(INCLUDES) -pthread $(BITS64) $(CINCLUDE)
NVCCFLAGS ?= $(OPT_NVCCFLAGS) -std=$(CUDA_STD) $(NVCC_ARCH_FLAGS) $(INCLUDES) -Xcompiler "$(BITS64),-pthread" $(CINCLUDE)

LDFLAGS = -L$(CUDA_LIBDIR) -lcufft -lcudart -lm -lpthread

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
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

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
