include ../Makefile.common
TARGET=load_vector-test


$(TARGET) : main.cu ../Makefile.common $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
