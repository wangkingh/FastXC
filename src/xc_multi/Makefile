#CC=gcc -Wall -DDEBUG
CC=gcc -Wall

CFLAG=-O2
CFLAG=-O0 -g

# static version 
#
#CFLAG=-O2 -static

NVCC=/usr/local/cuda/bin/nvcc
LFLAG=-L/usr/local/cuda/lib64 -lcufft -lcudart

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
	$(NVCC) $(CFLAG) -o $@ $^ -lm  $(LFLAG)

%.o: %.c
	$(CC) $(CFLAG) -c $<

%.o: %.cu
	$(NVCC) $(CFLAG) -c $< 

clean:
	-rm *.o
veryclean:
	-rm $(PROG) *.o 
