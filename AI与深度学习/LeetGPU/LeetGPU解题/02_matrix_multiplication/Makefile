NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_75 -std=c++11
TARGET = matrix_multiplication
SOURCE = main.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
