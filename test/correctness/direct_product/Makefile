include ../Makefile.common
TARGET=direct_product-test


$(TARGET) : main.cu ../Makefile.common $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
