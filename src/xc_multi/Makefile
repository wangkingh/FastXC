# Compiler
CC=gcc
# NVCC=/usr/local/cuda/bin/nvcc

# cuda architecture
# ARCH = sm_89

# compiler flags
CFLAGS=-Wall -O0 -g
NVCCFLAGS = -g -G -arch=$(ARCH)

# linker flags
LFLAG=-L/usr/local/cuda/lib64  -lcufft -lcudart -lm

CUDAOBJS=cuda.main.o \
		cuda.xc_dual.o \
		cuda.util.o

OBJS=arguproc.o \
	sacio.o \
	cal_dist.o \
	gen_ccfpath.o \
	par_read_spec.o \
	par_write_sac.o \
	par_filter_nodes.o \
	read_segspec.o \
	read_spec_lst.o \
	util.o
	
BINDIR=../../bin
PROG=$(BINDIR)/xc_multi_channel

all: $(BINDIR) $(PROG)

$(BINDIR):
	mkdir -p $(BINDIR)

$(PROG):$(OBJS) $(CUDAOBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LFLAG)

%.o: %.c
	$(CC) $(CFLAG) -c $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< 

clean:
	-rm *.o
veryclean:
	-rm $(PROG) *.o 
