TEST_ARCH=80
ROOT_DIR=../../include
NVCC=nvcc
NVCCFLAGS=-std=c++17 -I$(ROOT_DIR) -arch=sm_$(TEST_ARCH) -DTEST_ARCH=$(TEST_ARCH) --extended-lambda
HEADERS=$(shell find ../../include -name '*.hpp')

TARGET=
TARGET+=add_eye.test
TARGET+=direct_product.test
TARGET+=foreach.test
TARGET+=foreach_ij.test
TARGET+=foreach_v.test
TARGET+=foreach_v_acc.test
TARGET+=gevm.test
TARGET+=wmma.load_vector.test
TARGET+=wmma.store_vector.test
TARGET+=print_fragment.test
TARGET+=fill.test
TARGET+=mma.test
TARGET+=vector.test
TARGET+=map.test
TARGET+=operators.test

all: $(TARGET)

%.test : %.cu Makefile $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f *.test
