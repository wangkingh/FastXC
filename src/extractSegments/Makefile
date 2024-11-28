# 定义编译器
CC = gcc

# 定义编译选项，-g 用于调试信息，-Wall 开启所有警告
CFLAGS = -g -Wall

BINDIR=../../bin
PROG=$(BINDIR)/extractSegments

# 定义对象文件列表
OBJS = main.o arguproc.o sacio.o


all: $(BINDIR) $(PROG)

$(BINDIR):
	mkdir -p $(BINDIR)


# 构建可执行文件的规则
$(PROG): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ -lm

# 从.c文件生成.o文件的通用规则
%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	-rm *.o

veryclean:
	-rm -f $(PROG) *.o
