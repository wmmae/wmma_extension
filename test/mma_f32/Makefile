NVCC=nvcc

INCDIR=../../include

TEST_TF32=YES
TEST_SIMT=NO

NVCCFLAGS=-std=c++14 -I$(INCDIR) -Xcompiler="-fopenmp" --ptxas-options=-v -lcublas
NVCCFLAGS+=-gencode arch=compute_86,code=sm_86
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80

ifeq ($(TEST_TF32), YES)
NVCCFLAGS+=-DTEST_TF32
else
NVCCFLAGS+=-gencode arch=compute_75,code=sm_75
NVCCFLAGS+=-gencode arch=compute_70,code=sm_70
endif

ifeq ($(TEST_SIMT), YES)
NVCCFLAGS+=-DTEST_SIMT
endif

TARGET=mma.test elementwise.test vector.test matvec.test batch_gemm.test

all: $(TARGET)

%.test:%.cu
	$(NVCC) $< $(OBJS) $(NVCCFLAGS) -o $@

clean:
	rm -f *.test
