# TF-PWS-only stack backend.

include ../common.mk
.DEFAULT_GOAL := all

INCLUDES := -I. -Iinclude -Iio -Iworker -Icore -Ikernels
PROJECT_CPPFLAGS := $(INCLUDES) $(CUDA_REQUIRED_CPPFLAGS)
NVCC_HOST_FLAGS := -Xcompiler "$(BITS64),-pthread"

CFLAGS    ?= $(OPT_CFLAGS) $(WARN)
NVCCFLAGS ?= $(OPT_NVCCFLAGS)

LDFLAGS = $(CUDA_LDFLAGS) -lcufft -lcudart -lm -lpthread

CUDAOBJS = \
    kernels/cuda.stransform.o \
    cuda.util.o \
    core/tfpws_workspace.o \
    core/tfpws_compute.o \
    io/sourcepack_io.o \
    worker/prescan.o \
    worker/pipeline.o \
    cuda.main.o

OBJS = \
    arguproc.o \
    logger.o \
    io/path_util.o

BINDIR := ../../bin
PROG   := $(BINDIR)/ncf_tfpws

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

cuda.main.o: include/tfpws_types.hpp include/tfpws_input.hpp include/tfpws_schedule.hpp worker/pipeline.hpp cuda.util.cuh include/logger.h
cuda.util.o: cuda.util.cuh include/logger.h
arguproc.o: include/logger.h
logger.o: include/logger.h
core/tfpws_workspace.o: core/tfpws_workspace.hpp core/tfpws_compute.hpp include/tfpws_schedule.hpp cuda.util.cuh arguproc.h include/sac.h include/logger.h
core/tfpws_compute.o: core/tfpws_compute.hpp core/tfpws_workspace.hpp include/tfpws_schedule.hpp include/tfpws_sac.hpp kernels/cuda.stransform.cuh cuda.util.cuh arguproc.h include/sac.h include/logger.h
io/sourcepack_io.o: io/sourcepack_io.hpp io/path_util.h include/sac.h include/logger.h
worker/prescan.o: worker/prescan.hpp io/sourcepack_io.hpp core/tfpws_compute.hpp include/tfpws_schedule.hpp include/tfpws_types.hpp arguproc.h include/sac.h include/logger.h
worker/pipeline.o: worker/pipeline.hpp worker/prescan.hpp worker/host_workspace_budget.hpp worker/work_queue.hpp io/sourcepack_io.hpp core/tfpws_compute.hpp include/tfpws_schedule.hpp include/tfpws_types.hpp include/progress_sidecar.hpp arguproc.h include/sac.h include/logger.h

clean:
	@find . -type f -name '*.o' -exec rm -f {} +

veryclean: clean
	@rm -f $(PROG)
