NVCC=nvcc

INCDIR=../../include

TEST_TF32=NO
TEST_SIMT=YES
SM_ARCH=Volta

NVCCFLAGS=-std=c++14 -I$(INCDIR) -Xcompiler="-fopenmp" --ptxas-options=-v -lcublas

ifeq ($(SM_ARCH), Ampere)
NVCCFLAGS+=-gencode arch=compute_86,code=sm_86
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80

ifeq ($(TEST_TF32), YES)
NVCCFLAGS+=-DTEST_TF32
endif

endif
ifeq ($(SM_ARCH), Turing)
NVCCFLAGS+=-gencode arch=compute_75,code=sm_75 -DSM_ARCH=75
endif
ifeq ($(SM_ARCH), Volta)
NVCCFLAGS+=-gencode arch=compute_70,code=sm_70 -DSM_ARCH=70
endif

ifeq ($(TEST_SIMT), YES)
NVCCFLAGS+=-DTEST_SIMT
endif

TARGET=batch_gemm.test

all: $(TARGET)

%.test:%.cu
	$(NVCC) $< $(OBJS) $(NVCCFLAGS) -o $@

clean:
	rm -f *.test
