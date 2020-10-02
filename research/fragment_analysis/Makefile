SM=70
NVCC=nvcc
NVCCFLAGS=-std=c++11 -arch=sm_${SM} -DARCH=${SM}

TARGET=fragment_analysis.out

$(TARGET):main.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS)

clean:
	rm -f $(TARGET)
