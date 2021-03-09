TEST_ARCH=70
ROOT_DIR=../../include
NVCC=nvcc
NVCCFLAGS=-std=c++17 -I$(ROOT_DIR) -arch=sm_$(TEST_ARCH) -DTEST_ARCH=$(TEST_ARCH)
HEADERS=$(shell find ../../include -name '*.hpp')

TARGET=
TARGET+=add_eye.test
TARGET+=direct_product.test
TARGET+=foreach.test
TARGET+=foreach_v.test
TARGET+=foreach_v_acc.test
TARGET+=gevm.test
TARGET+=load_vector.test
TARGET+=store_vector.test
TARGET+=m8n8k4.test
TARGET+=print_fragment.test

all: $(TARGET)

%.test : %.cu Makefile $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
