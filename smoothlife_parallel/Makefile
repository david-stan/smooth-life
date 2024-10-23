# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O3 -Iinclude
LDFLAGS = -Llib -lfftw3 -lcufft -lcudart

# FFTW and CUDA libraries
LIBS = -lfftw3 -lcufft -lcudart

# Target executable
TARGET = smoothlife

# Source files
SRC_CXX = src/main.cpp src/smoothlife.cpp src/utils.cpp
SRC_CUDA = src/smoothlife_cuda.cu
OBJS = $(SRC_CXX:.cpp=.o) $(SRC_CUDA:.cu=.o)

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

# FFTW and CUDA installations
fftw:
	@echo "Installing FFTW..."
	sudo apt-get install libfftw3-dev

cuda:
	@echo "Installing CUDA..."
	sudo apt-get install nvidia-cuda-toolkit
