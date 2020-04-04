TEST_ARCH=75
NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../../../ -arch=sm_$(TEST_ARCH)  -DTEST_ARCH=$(TEST_ARCH)
TARGET=store_vector-test


$(TARGET) : main.cu ../../../wmma_extension.hpp Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
