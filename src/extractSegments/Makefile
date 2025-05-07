# ----------------------------------------------------
#  extractSegments ‒ 子目录 Makefile
#  · 顶层递归调用  → Release  (-O3)
#  · 手动进目录 make → Debug    (-O0 -g)
# ----------------------------------------------------

# 编译器
CC      ?= gcc
WARN    ?= -Wall            # 通用警告

# ---------- 优化级别自动切换 ----------
ifeq ($(MAKELEVEL),0)        # 手动调试
    CFLAGS  ?= -O0 -g $(WARN)
else                          # 顶层批量构建
    CFLAGS  ?= -O3 $(WARN)
endif
# -------------------------------------

# 链接库（目前只需要 libm，如有其他库再追加）
LDFLAGS ?= -lm

# 源文件 / 目标文件
OBJS = main.o arguproc.o sacio.o

# 输出可执行文件
BINDIR := ../../bin
PROG   := $(BINDIR)/extractSegments

# 伪目标
.PHONY: all clean veryclean

# 默认目标
all: $(BINDIR) $(PROG)

# 创建输出目录
$(BINDIR):
	@mkdir -p $@

# 链接
$(PROG): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# C 源编译
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 清理
clean:
	@rm -f *.o

veryclean: clean
	@rm -f $(PROG)
