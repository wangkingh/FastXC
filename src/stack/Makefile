# ----------------------------------------------------
#  ncfstack ‒ 子目录 Makefile（纯手写依赖）
#  • 顶层 make 递归调用 → Release (-O3)          (MAKELEVEL=1)
#  • 手动进入目录 make   → Debug (-O0 -g)       (MAKELEVEL=0)
# ----------------------------------------------------

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

# 源文件 / 目标文件
SRCS := main.c arguproc.c sacio.c read_sac_lst.c util.c
OBJS := $(SRCS:.c=.o)

# 输出目录 / 可执行文件
BINDIR := ../../bin
PROG   := $(BINDIR)/ncfstack

# 伪目标
.PHONY: all clean veryclean

# 默认目标
all: $(BINDIR) $(PROG)

# 创建输出目录（若已存在则静默）
$(BINDIR):
	@mkdir -p $@

# 链接
$(PROG): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# 通用编译规则
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ----------------------------------------------------
#  手写依赖区——按实际头文件增删
# ----------------------------------------------------
main.o:          main.c sac.h arguproc.h read_sac_lst.h util.h
arguproc.o:      arguproc.c sac.h arguproc.h
sacio.o:         sacio.c sac.h
read_sac_lst.o:  read_sac_lst.c sac.h read_sac_lst.h
util.o:          util.c util.h sac.h
# ----------------------------------------------------

# 清理
clean:
	@rm -f *.o
veryclean: clean
	@rm -f $(PROG)
