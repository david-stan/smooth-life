# Compiler and flags
CXX = g++                  # C++ compiler
NVCC = nvcc                # CUDA compiler

CXXFLAGS = -std=c++17 -O3 -Wall -g -I/usr/include -Iinclude  # C++ flags, include headers from include/
NVCCFLAGS = -Iinclude -O3          # CUDA flags
LDFLAGS = -L/usr/local/lib -lgsl -lfftw3 -lcufft -lcudart -lsfml-graphics -lsfml-window -lsfml-system
  # Linker flags, link with FFTW and CUDA libraries

# Target executable
TARGET = smoothlife              # Final executable name

# Source files
SRC_CXX = src/main.cpp src/smoothlife.cpp src/utils.cpp  														 # C++ source files
SRC_CUDA = cuda/smoothlife_cuda.cu                        														 # CUDA source files

OBJDIR = bin
OBJS = $(patsubst src/%.cpp,$(OBJDIR)/%.o,$(SRC_CXX)) $(patsubst cuda/%.cu,$(OBJDIR)/%.o,$(SRC_CUDA))             # Object files after compilation

# Build rules
all: $(OBJDIR) $(TARGET)

# Create bin directory if it doesn't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Linking step for the executable
$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Compile C++ source files into bin/
$(OBJDIR)/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files into bin/
$(OBJDIR)/%.o: cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJDIR)/*.o $(TARGET)

# FFTW and CUDA installation (optional)
fftw:
	@echo "Installing FFTW..."
	sudo apt-get install libfftw3-dev

cuda:
	@echo "Installing CUDA..."
	sudo apt-get install nvidia-cuda-toolkit

gsl:
	@echo "Installing GSL..."
	sudo apt-get install libgsl-dev