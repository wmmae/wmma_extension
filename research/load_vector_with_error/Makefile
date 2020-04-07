TEST_ARCH=70
NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../../ -arch=sm_$(TEST_ARCH)
TARGET=load_vector_with_error


$(TARGET) : main.cu ../../wmma_extension.hpp Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
