include ../Makefile.common
TARGET=foreach_v-test


$(TARGET) : main.cu ../Makefile.common $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
