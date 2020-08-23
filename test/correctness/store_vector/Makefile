TEST_ARCH=70
NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../../../include -arch=sm_$(TEST_ARCH) -DTEST_ARCH=$(TEST_ARCH)
HEADERS=$(shell find ../../../include -name '*.hpp')
TARGET=store_vector-test


$(TARGET) : main.cu Makefile $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
