# TF-PWS-only stack backend.

include ../common.mk

CUDA_STD ?= c++14

INCLUDES := -I. -Iinclude -Iio -Iworker -Icore -Ikernels

CFLAGS    ?= $(OPT_CFLAGS) $(WARN) -pthread $(BITS64) $(CINCLUDE) $(INCLUDES)
NVCCFLAGS ?= $(OPT_NVCCFLAGS) -std=$(CUDA_STD) $(NVCC_ARCH_FLAGS) -Xcompiler "$(BITS64),-pthread" $(CINCLUDE) $(INCLUDES)

LDFLAGS = -L$(CUDA_LIBDIR) -lcufft -lcudart -lm -lpthread

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
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

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
	@rm -f *.o core/*.o io/*.o worker/*.o kernels/*.o

veryclean: clean
	@rm -f $(PROG)
