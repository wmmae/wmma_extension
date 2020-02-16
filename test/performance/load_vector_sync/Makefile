TEST_ARCH=70
ROOT_DIR=../../../
NVCC=nvcc
NVCCFLAGS=-std=c++11 -I$(ROOT_DIR) -arch=sm_$(TEST_ARCH) -DCUDA_ARCH_SM=$(TEST_ARCH)

all: batched_direct_product.test

%.test : %.cu $(ROOT_DIR)/wmma_extension.hpp Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
