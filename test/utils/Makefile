TEST_ARCH=80
ROOT_DIR=../../include
NVCC=nvcc
NVCCFLAGS=-std=c++17 -I$(ROOT_DIR) -arch=sm_$(TEST_ARCH) -DTEST_ARCH=$(TEST_ARCH) --extended-lambda
HEADERS=$(shell find ../../include -name '*.hpp')

TARGET=
TARGET+=cast.test cp_async.test

all: $(TARGET)

%.test : %.cu Makefile $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f *.test
