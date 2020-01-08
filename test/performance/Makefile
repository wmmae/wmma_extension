TEST_ARCH=75
NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../../ -arch=sm_$(TEST_ARCH) -DCUDA_ARCH_SM=$(TEST_ARCH)
TARGET=wlv-performance.test


$(TARGET) : main.cu ../../wmma_extension.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
