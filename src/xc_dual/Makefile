# ----------------------------------------------------
#  xc_dual_channel ‒ 子目录 Makefile
#  • 顶层递归调用  → Release  (-O3, --generate-line-info)
#  • 手动进入目录   → Debug    (-O0 -g -G -lineinfo)
# ----------------------------------------------------

# 编译器
CC        ?= gcc
NVCC      ?= /usr/local/cuda/bin/nvcc
ARCH      ?= sm_89
WARN      ?= -Wall

# ---------- 优化级别自动切换 ----------
ifeq ($(MAKELEVEL),0)                # 手动调试
    CFLAGS     ?= -O0 -g $(WARN)
    NVCCFLAGS  ?= -g -G -lineinfo -arch=$(ARCH)
else                                  # 顶层批量构建
    CFLAGS     ?= -O3 $(WARN)
    NVCCFLAGS  ?= -O3 --generate-line-info -arch=$(ARCH)
endif
# -------------------------------------

# 链接库
LDFLAGS = -L/usr/local/cuda/lib64 -lcufft -lcudart -lm \
          -L/usr/lib/x86_64-linux-gnu

# 源文件
CUDAOBJS = cuda.stransform.o \
           cuda.pws_util.o \
           cuda.estimate_batch.o\
           cuda.xc_dual.o \
           cuda.util.o \
           cuda.main.o

OBJS = arguproc.o \
       sacio.o \
       gen_ncf_path.o \
       gen_pair.o \
       read_segspec.o \
       util.o \
       cal_dist.o

# 输出可执行文件
BINDIR := ../../bin
PROG   := $(BINDIR)/xc_dual_channel

# 伪目标
.PHONY: all clean veryclean

# 默认目标
all: $(BINDIR) $(PROG)

# 创建输出目录
$(BINDIR):
	@mkdir -p $@

# 链接
$(PROG): $(OBJS) $(CUDAOBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

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
