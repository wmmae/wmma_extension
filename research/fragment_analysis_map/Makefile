SM=80
NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_${SM} -DARCH=${SM} -I../../include

TARGET=fragment_analysis.out

$(TARGET):main.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS)

clean:
	rm -f $(TARGET)
