#include "smoothlife.hpp"
#include "utils.hpp"
#include <iostream>
#include <fftw3.h>

SmoothLife::SmoothLife(int grid_size, double radius) : grid_size(grid_size), radius(radius) {
    field.resize(grid_size, std::vector<double>(grid_size, 0.0));
    // FFTW-based FFT computation for non-CUDA
    input_weights = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);
    output_weights = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);
    // Plan the FFT
    plan_fft2 = fftw_plan_dft_2d(grid_size, grid_size, input_weights, output_weights, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_ifft2 = fftw_plan_dft_2d(grid_size, grid_size, output_weights, input_weights, FFTW_BACKWARD, FFTW_ESTIMATE);
}

SmoothLife::~SmoothLife() {
    fftw_destroy_plan(plan_fft2);
    fftw_destroy_plan(plan_ifft2);
    
    fftw_free(input_weights);
    fftw_free(output_weights);
}

void SmoothLife::initializeField() {
    initializeRandomCircles(field, grid_size, 20, radius, radius + 1.0); // Example initialization
}

void SmoothLife::initializeWeights() {
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            input_weights[i * grid_size + j][0] = field[i][j]; // Real
            input_weights[i * grid_size + j][1] = 0.0; // Imaginary part
        }
    }
}

void SmoothLife::updateField() {
    // execute fft
    fftw_execute(plan_fft2);
}


void SmoothLife::updateFieldCUDA() {
    // Call CUDA function to update the field
    applyCudaUpdates();
}

void SmoothLife::printField() {
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            std::cout << field[i][j];
        }
        std::cout << std::endl;
    }
}
