# xc_fast backend.

include ../common.mk
.DEFAULT_GOAL := all

INCLUDES := -I. -Iinclude -Iio -Iworker -Imemory -Ikernels -Ipack_io
PROJECT_CPPFLAGS := $(INCLUDES) $(CUDA_REQUIRED_CPPFLAGS)

CFLAGS    ?= $(OPT_CFLAGS) $(WARN)
NVCCFLAGS ?= $(OPT_NVCCFLAGS)

LFLAG = $(CUDA_LDFLAGS) -lcufft -lcudart -lm -lpthread

CUDAOBJS = \
    cuda.main.o \
    cuda.util.o \
    kernels/cuda.kernels.o \
    worker/async_workers.o \
    worker/executor.o \
    worker/scheduler.o \
    memory/memory.o \
    io/input.o \
    io/sac_record.o \
    io/writer.o \
    pack_io/pack_writer.o

OBJS = \
    arguproc.o \
    logger.o \
    io/sacio.o \
    io/path_table.o \
    io/fs.o \
    io/sac_header.o

BINDIR := ../../bin
PROG   := $(BINDIR)/xc_fast

.PHONY: all clean veryclean

all: $(BINDIR) $(PROG)

$(BINDIR):
	@mkdir -p $@

$(PROG): $(OBJS) $(CUDAOBJS)
	$(NVCC) $(PROJECT_CPPFLAGS) $(NVCCFLAGS) $(CUDA_REQUIRED_NVCCFLAGS) -o $@ $^ $(LFLAG)

%.o: %.c
	$(CC) $(PROJECT_CPPFLAGS) $(CFLAGS) -pthread -c $< -o $@

%.o: %.cu
	$(NVCC) $(PROJECT_CPPFLAGS) $(NVCCFLAGS) $(CUDA_REQUIRED_NVCCFLAGS) -c $< -o $@

arguproc.o cuda.main.o memory/memory.o io/input.o: arguproc.h
cuda.main.o worker/async_workers.o memory/memory.o: memory/memory.hpp
cuda.main.o worker/async_workers.o worker/executor.o worker/scheduler.o memory/memory.o io/input.o io/sac_record.o io/writer.o: runtime.hpp
cuda.main.o worker/async_workers.o worker/scheduler.o: worker/scheduler.hpp
cuda.main.o worker/async_workers.o: worker/async_workers.hpp
worker/async_workers.o worker/executor.o memory/memory.o kernels/cuda.kernels.o: kernels/cuda.kernels.cuh
worker/async_workers.o worker/executor.o memory/memory.o cuda.util.o: cuda.util.cuh
cuda.main.o io/input.o worker/async_workers.o: io/input.hpp
worker/async_workers.o io/writer.o: io/writer.hpp
io/sac_record.o io/writer.o: io/sac_record.hpp
cuda.main.o io/sac_record.o io/writer.o pack_io/pack_writer.o: pack_io/pack_writer.hpp
pack_io/pack_writer.o io/sac_record.o io/writer.o: io/fs.h
cuda.main.o io/path_table.o worker/scheduler.o: io/path_table.h
io/writer.o io/fs.o: io/fs.h
io/sac_record.o io/writer.o io/sac_header.o: io/sac_header.h
arguproc.o cuda.main.o io/input.o memory/memory.o io/path_table.o io/sacio.o io/writer.o worker/async_workers.o: include/logger.h
cuda.main.o io/writer.o: progress_sidecar.hpp

clean:
	@find . -type f -name '*.o' -exec rm -f {} +

veryclean: clean
	@rm -f $(PROG)
