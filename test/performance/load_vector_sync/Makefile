include ../Makefile.common

all: batched_direct_product.test direct_product.test

%.test : %.cu $(ROOT_DIR)/wmma_extension/wmma_extension.hpp Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f *.test
