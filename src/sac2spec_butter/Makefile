# ----------------------------------------------------
#  sac2spec_butter ‒ 子目录 Makefile（Release / Debug 自动切换）
# ----------------------------------------------------

# 编译器
CC        ?= gcc
NVCC      ?= /usr/local/cuda/bin/nvcc
ARCH      ?= sm_89
WARN      ?= -Wall

# ---------- 优化级别自动切换 ----------
ifeq ($(MAKELEVEL),0)              # 手动调试
    CFLAGS     ?= -O0 -g $(WARN)
    NVCCFLAGS  ?= -g -G -lineinfo -arch=$(ARCH)
else                                # 顶层批量构建
    CFLAGS     ?= -O3 $(WARN)
    NVCCFLAGS  ?= -O3 --generate-line-info -arch=$(ARCH)
endif
# -------------------------------------

# 链接库
LFLAG = -L/usr/local/cuda/lib64 -lcufft -lcudart -lm

# 源文件
CUDAOBJS = cuda.rdcrtr.o \
           cuda.util.o  \
           cuda.misc.o  \
           cuda.filter.o \
           cuda.processing.o \
           cuda.taper.o \
           cuda.main.o

OBJS = design_filter_response.o \
       arguproc.o \
       cal_nseg.o \
       par_rw_data.o \
       read_filelist.o \
       sacio.o \
       util.o

# 输出可执行文件
BINDIR := ../../bin
PROG   := $(BINDIR)/sac2spec_butter

# 伪目标
.PHONY: all clean veryclean

# 默认目标
all: $(BINDIR) $(PROG)

# 创建输出目录
$(BINDIR):
	@mkdir -p $@

# 链接
$(PROG): $(OBJS) $(CUDAOBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LFLAG)

# C 文件编译
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# CUDA 文件编译
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 清理
clean:
	@rm -f *.o

veryclean: clean
	@rm -f $(PROG)
