#include "../include/smoothlife.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <fftw3.h>

SmoothLife::SmoothLife(int grid_size, double radius) : grid_size(grid_size), radius(radius) {
    field.resize(grid_size, std::vector<double>(grid_size, 0.0));
}

void SmoothLife::initializeField() {
    initializeRandomCircles(field, grid_size, 10, 5, 30); // Example initialization
}

void SmoothLife::updateField() {
    // Call CUDA function to update the field
    applyCudaUpdates();
}

void SmoothLife::computeFFT() {
    // FFTW-based FFT computation for non-CUDA
    fftw_complex* input = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);
    fftw_complex* output = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);
    
    // Plan the FFT
    fftw_plan plan = fftw_plan_dft_2d(grid_size, grid_size, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    
    // Fill in data
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            input[i * grid_size + j][0] = field[i][j];
            input[i * grid_size + j][1] = 0.0; // Imaginary part
        }
    }
    
    // Execute FFT
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    
    fftw_free(input);
    fftw_free(output);
}
