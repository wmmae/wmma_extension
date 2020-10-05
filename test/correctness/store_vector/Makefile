include ../Makefile.common
TARGET=store_vector-test


$(TARGET) : main.cu ../Makefile.common $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
