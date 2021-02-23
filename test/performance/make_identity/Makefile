include ../Makefile.common

all: batched_householder.test

%.test: %.cu $(ROOT_DIR)/wmma_extension/wmma_extension.hpp Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f *.test
