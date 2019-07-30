TEST_ARCH=70
NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../ -arch=sm_$(TEST_ARCH) -DCUDA_ARCH=$(TEST_ARCH)0
TARGET=wlv-test


$(TARGET) : main.cu ../wmma_load_vector.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
