TEST_ARCH=70
NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../../../include -arch=sm_$(TEST_ARCH) -DTEST_ARCH=$(TEST_ARCH)
TARGET=m8n8k4-test


$(TARGET) : main.cu ../../../include/wmma_extension.hpp Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
