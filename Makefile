# This makefile is intended for the GNU C compiler.
# Your code must compile (with GCC) with the given CFLAGS.
# You may experiment with the MY_OPT variable to invoke additional compiler options
HOST = $(shell hostname)
BANG = $(shell hostname | grep ccom-bang | wc -c)
BANG-COMPUTE = $(shell hostname | grep compute | wc -c)
STAMPEDE = $(shell hostname | grep stampede | wc -c)
AMAZON = $(shell hostname | grep 'ip-' | wc -c)
JASONMAC = $(shell hostname | grep 'Mac' | wc -c)

ifneq ($(JASONMAC), 0)
#On OS X
CFLAGS += -I/usr/local/opt/openblas/include
LDLIBS += -lblas
CC = gcc
atlas := 1
multi := 0
#NO_BLAS = 1
else
ifneq ($(STAMPEDE), 0)
multi := 1
NO_BLAS = 0
# PUB = /home1/00660/tg458182/cse262-wi15
#include $(PUB)/Arch/arch.intel-mkl
else
ifneq ($(BANG), 0)
atlas := 1
multi := 0
NO_BLAS = 1
include $(PUB)/Arch/arch.gnu_c99.generic
else
ifneq ($(BANG-COMPUTE), 0)
atlas := 1
multi := 0
NO_BLAS = 1
include $(PUB)/Arch/arch.gnu_c99.generic
else
ifneq ($(AMAZON), 0)
atlas := 1
multi := 0
NO_BLAS = 1
include $(PUB)/Arch/arch.gnu_c99.generic
CFLAGS += -mfma
endif
endif
endif
endif
endif

# Added by Zhen Liang
# Additional Flags
# CFLAGS += -mtune=core2
CFLAGS += -march=native
CFLAGS += -ftree-vectorize
CFLAGS += -funroll-loops
CFLAG += -ffast-math

#WARNINGS += -Wall -pedantic
WARNINGS += -w -pedantic

# If you want to copy data blocks to contiguous storage
# This applies to the hand code version
ifeq ($(copy), 1)
    C++FLAGS += -DCOPY
    CFLAGS += -DCOPY
endif


# If you want to use restrict pointers, make restrict=1
# This applies to the hand code version
ifeq ($(restrict), 1)
    C++FLAGS += -D__RESTRICT
    CFLAGS += -D__RESTRICT
# ifneq ($(CARVER), 0)
#    C++FLAGS += -restrict
#     CFLAGS += -restrict
# endif
endif

ifeq ($(NO_BLAS), 1)
    C++FLAGS += -DNO_BLAS
    CFLAGS += -DNO_BLAS
endif


OPTIMIZATION = $(MY_OPT)

targets = benchmark-naive \
			benchmark-blocked \
			benchmark-blocked-final \
			benchmark-blocked-naive \
			benchmark-blas

objects = benchmark.o \
			dgemm-naive.o \
			dgemm-blocked.o \
			dgemm-blocked-final.o \
			dgemm-blocked-naive.o \
			dgemm-blas.o

UTIL   = wall_time.o cmdLine.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o  $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS) -mavx -mavx2 -mfma

benchmark-blocked : benchmark.o dgemm-blocked.o $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS) -pg -mavx -mavx2

benchmark-blocked-final : benchmark.o dgemm-blocked-final.o $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS) -pg -mavx -mavx2

benchmark-blocked-naive : benchmark.o dgemm-blocked-naive.o $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS) -pg -mavx -mavx2

benchmark-blas : benchmark.o dgemm-blas.o $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS) -mavx -mavx2 -mfma

%.o : %.c
	$(CC) -c $(CFLAGS) -O4 $<
#	$(CC) -c $(CFLAGS) $(OPTIMIZATION) $<


.PHONY : clean
clean:
	rm -f $(targets) $(objects) $(UTIL) core
