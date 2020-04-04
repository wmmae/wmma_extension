TEST_ARCH=70
NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../../../ -arch=sm_$(TEST_ARCH) -DTEST_ARCH=$(TEST_ARCH)
TARGET=gevm-test


$(TARGET) : main.cu ../../../wmma_extension.hpp Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
