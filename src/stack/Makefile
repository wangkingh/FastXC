CC=gcc -Wall

CFLAG=-O2

OBJS=main.o\
	arguproc.o \
	sacio.o \
	read_sac_lst.o \
	util.o

BINDIR=../../bin
PROG=$(BINDIR)/ncfstack

all: $(BINDIR) $(PROG)

$(BINDIR):
	mkdir -p $(BINDIR)

$(PROG):$(OBJS)
	$(CC) $(CFLAG) -o $@ $^ -lm 

%.o: %.c
	$(CC) $(CFLAG) -c $< 

 
clean:
	-rm *.o
veryclean:
	-rm $(PROG) *.o 
