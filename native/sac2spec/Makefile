# sac2spec backend.

include ../common.mk

CUDA_STD ?= c++14

INCLUDES  := -I. -Iinclude
CFLAGS    ?= $(OPT_CFLAGS) $(WARN) $(INCLUDES)
NVCCFLAGS ?= $(OPT_NVCCFLAGS) -std=$(CUDA_STD) $(NVCC_ARCH_FLAGS) $(INCLUDES)

LFLAG = -L$(CUDA_LIBDIR) -lcufft -lcudart -lm -lpthread

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
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LFLAG)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	@rm -f *.o core/*.o io/*.o kernels/*.o worker/*.o

veryclean: clean
	@rm -f $(PROG)
