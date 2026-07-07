# sac2spec backend.

include ../common.mk
.DEFAULT_GOAL := all

INCLUDES  := -I. -Iinclude
PROJECT_CPPFLAGS := $(INCLUDES) $(CUDA_REQUIRED_CPPFLAGS)
CFLAGS    ?= $(OPT_CFLAGS) $(WARN)
NVCCFLAGS ?= $(OPT_NVCCFLAGS)

LFLAG = $(CUDA_LDFLAGS) -lcufft -lcudart -lm -lpthread

CUDAOBJS = \
    cuda.util.o \
    device_memory.o \
    device_memory_estimate.o \
    output_layout.o \
    progress.o \
    timestamp_tracker.o \
    worker/task_queue.o \
    worker/capacity.o \
    worker/memory.o \
    worker/compute.o \
    worker/batch_pipeline.o \
    worker/async_workers.o \
    io/worker_batch_io.o \
    cuda.main.o

KERNELOBJS = \
    kernels/real_matrix.o \
    kernels/complex_matrix.o \
    kernels/rdcrtr.o \
    kernels/taper.o

CORECUDAOBJS = \
    core/smoothing.o \
    core/normalization.o \
    core/preprocess.o \
    core/spectrum.o

OBJS = \
    core/filter_response.o \
    arguproc.o \
    fft_length.o \
    logger.o \
    io/io_pool.o \
    io/parallel_io.o \
    io/stepack_writer.o \
    io/sac_index.o \
    io/sacio.o \
    io/sacio_legacy.o \
    path_utils.o \
    util.o

BINDIR := ../../bin
PROG   := $(BINDIR)/sac2spec

.PHONY: all clean veryclean

all: $(BINDIR) $(PROG)

$(BINDIR):
	@mkdir -p $@

$(PROG): $(OBJS) $(CUDAOBJS) $(KERNELOBJS) $(CORECUDAOBJS)
	$(NVCC) $(PROJECT_CPPFLAGS) $(NVCCFLAGS) $(CUDA_REQUIRED_NVCCFLAGS) -o $@ $^ $(LFLAG)

%.o: %.c
	$(CC) $(PROJECT_CPPFLAGS) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(PROJECT_CPPFLAGS) $(NVCCFLAGS) $(CUDA_REQUIRED_NVCCFLAGS) -c $< -o $@

clean:
	@find . -type f -name '*.o' -exec rm -f {} +

veryclean: clean
	@rm -f $(PROG)
