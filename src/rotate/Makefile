# -------------------------------------------
#  RotateNCF ‒ 子目录 Makefile（纯手写依赖）
# -------------------------------------------

# 编译器
CC      ?= gcc
WARN    ?= -Wall

# --- 优化级别自动切换 ---
ifeq ($(MAKELEVEL),0)          # 手动调试
    OPTFLAGS := -O0 -g
else                           # 顶层批量构建
    OPTFLAGS := -O3
endif

CFLAGS ?= $(OPTFLAGS) $(WARN)
# ------------------------

# 源文件和目标文件
SRCS := main.c arguproc.c sacio.c hddistance.c rotate.c
OBJS := $(SRCS:.c=.o)

# 输出目录 / 可执行文件
BINDIR := ../../bin
PROG   := $(BINDIR)/RotateNCF

# 伪目标
.PHONY: all clean veryclean

# 默认目标
all: $(BINDIR) $(PROG)

# 创建输出目录
$(BINDIR):
	@mkdir -p $@

# 链接
$(PROG): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# 通用编译规则
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# -------------------------------------------
#  手写依赖区（按需增删，别省略）
# -------------------------------------------
main.o:        main.c sac.h arguproc.h
arguproc.o:    arguproc.c sac.h arguproc.h
sacio.o:       sacio.c sac.h
hddistance.o:  hddistance.c sac.h hddistance.h
rotate.o:      rotate.c sac.h arguproc.h hddistance.h
# -------------------------------------------

# 清理
clean:
	@rm -f *.o
veryclean: clean
	@rm -f $(PROG)
