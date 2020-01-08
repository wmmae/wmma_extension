TEST_ARCH=70
NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../ -arch=sm_$(TEST_ARCH)
TARGET=wlv-test


$(TARGET) : main.cu ../wmma_extension.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
