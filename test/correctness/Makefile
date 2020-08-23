TEST_ARCH=75
NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../../include -arch=sm_$(TEST_ARCH)
TARGET=wlv-test
HEADERS=$(shell find ../../include -name '*.hpp')


$(TARGET) : main.cu Makefile $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
