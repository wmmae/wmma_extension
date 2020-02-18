include ../Makefile.common

all: batched_householder.test

#batched_householder.test: batched_householder.cu $(ROOT_DIR)/wmma_extension.hpp Makefile
%.test: %.cu $(ROOT_DIR)/wmma_extension.hpp Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f *.test
