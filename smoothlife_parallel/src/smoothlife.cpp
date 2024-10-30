#include "smoothlife.hpp"
#include "utils.hpp"
#include <iostream>
#include <fftw3.h>
#include <algorithm>
#include <SFML/Graphics.hpp>

double SmoothLife::B0 = 0.278;
double SmoothLife::B1 = 0.365;
double SmoothLife::D0 = 0.267;
double SmoothLife::D1 = 0.445;
double SmoothLife::alpha_M = 0.147;
double SmoothLife::alpha_N = 0.028;

SmoothLife::SmoothLife(int grid_size, double radius) : grid_size(grid_size), inner_radius(radius) {
    outer_radius = inner_radius * 3;
    field.resize(grid_size, std::vector<double>(grid_size, 0.0));
    initializeRandomCircles(field, grid_size, 30, radius, radius * 7); // initialization
    // FFTW-based FFT computation for non-CUDA
    input_weights = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            input_weights[i * grid_size + j][0] = field[i][j]; // Real
            input_weights[i * grid_size + j][1] = 0.0; // Imaginary part
        }
    }
    output_weights = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);

    disk_weights = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);
    annulus_weights = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);

    M_fourier = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);
    M_spatial = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);
    N_fourier = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);
    N_spatial = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);

    // Plan the FFT
    plan_fft2_field = fftw_plan_dft_2d(grid_size, grid_size, input_weights, output_weights, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_ifft2_M = fftw_plan_dft_2d(grid_size, grid_size, M_fourier, M_spatial, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan_ifft2_N = fftw_plan_dft_2d(grid_size, grid_size, N_fourier, N_spatial, FFTW_BACKWARD, FFTW_ESTIMATE);

    initializeWeightsDisk();
    initializeWeightsAnnulus();
}

SmoothLife::~SmoothLife() {
    fftw_destroy_plan(plan_fft2_field);
    fftw_destroy_plan(plan_ifft2_M);
    fftw_destroy_plan(plan_ifft2_N);
    
    fftw_free(input_weights);
    fftw_free(output_weights);

    fftw_free(disk_weights);
    fftw_free(annulus_weights);

    fftw_free(M_fourier);
    fftw_free(M_spatial);
    fftw_free(N_fourier);
    fftw_free(N_spatial);
}

// Logistic threshold function σ1
double SmoothLife::logistic_threshold(double x, double x0, double alpha) {
    return 1.0 / (1.0 + std::exp(-4.0 / alpha * (x - x0))); 
}

// Logistic interval function σ2
double SmoothLife::logistic_interval(double x, double a, double b, double alpha) {
    return logistic_threshold(x, a, alpha) * (1.0 - logistic_threshold(x, b, alpha));
}

// Linear interpolation function
double SmoothLife::lerp(double a, double b, double t) {
    return (1.0 - t) * a + t * b;
}

void SmoothLife::S(fftw_complex* M, fftw_complex* N, fftw_complex* output) {
    fftw_complex* kurac = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);
    fftw_complex* kurac1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * grid_size * grid_size);
    for (int i = 0; i < grid_size * grid_size; ++i) {
        // Extract the real parts of M and N
        double M_real = (1.0 / (2 * M_PI * std::sqrt(inner_radius))) * M[i][0];  // Real part of M
        double N_real = (1.0 / (8 * M_PI * std::sqrt(inner_radius))) * N[i][0];  // Real part of N
        kurac[i][0] = M_real;
        kurac1[i][0] = N_real;
        std::cout << M[i][0] << " ";
        // Compute aliveness using logistic_threshold on M
        double aliveness = logistic_threshold(M[i][0], 0.5, alpha_M);
        // std::cout << aliveness;
        // std::cout << aliveness << std::endl;
        // Interpolate between birth and death thresholds based on aliveness
        double threshold1 = lerp(B0, D0, aliveness);
        double threshold2 = lerp(B1, D1, aliveness);
        // std::cout << threshold1 << " -- " << threshold2 << std::endl;

        // Compute new aliveness using logistic_interval on N
        double new_aliveness = logistic_interval(N[i][0], threshold1, threshold2, alpha_N);
        // std::cout << new_aliveness << std::endl;
        // Clamp the result between 0 and 1 and store it in the real part of the output
        output[i][0] = std::clamp(new_aliveness, 0.0, 1.0);  // Only modify the real part
        output[i][1] = 0.0;  // Set the imaginary part to 0
    }
    visualize2DVector(kurac, grid_size);
    visualize2DVector(kurac1, grid_size);
}

void SmoothLife::initializeWeightsDisk() {
    int half_size = grid_size / 2;
    double grid_spacing = 1.0 / grid_size;

    // Iterate over all frequency components in Fourier space
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            // Compute the radial frequency rho (distance to the origin in frequency space)
            int kx = (i < half_size) ? i : i - grid_size;  // Handle periodicity
            int ky = (j < half_size) ? j : j - grid_size;
            double rho = std::sqrt(kx * kx + ky * ky) * grid_spacing;

            // Avoid division by zero at the origin (low frequency components)
            if (rho == 0.0) {
                disk_weights[i * grid_size + j][0] = M_PI * inner_radius * inner_radius;  // Area of the disk for the zero frequency
            } else {
                // Compute the Fourier transform (Bessel function J1)
                disk_weights[i * grid_size + j][0] = (std::sqrt(3 * inner_radius) / (4 * rho)) * gsl_sf_bessel_J1(2 * M_PI * inner_radius * rho);
            }
        }
    }
}

void SmoothLife::initializeWeightsAnnulus() {
    int half_size = grid_size / 2;
    double grid_spacing = 1.0 / grid_size;

    // Iterate over all frequency components in Fourier space
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            // Compute the radial frequency rho (distance to the origin in frequency space)
            int kx = (i < half_size) ? i : i - grid_size;  // Handle periodicity
            int ky = (j < half_size) ? j : j - grid_size;
            double rho = std::sqrt(kx * kx + ky * ky) * grid_spacing;

            // Avoid division by zero at the origin (low frequency components)
            if (rho == 0.0) {
                annulus_weights[i * grid_size + j][0] = M_PI * outer_radius * outer_radius - disk_weights[0][0];  // Area of the disk for the zero frequency
            } else {
                // Compute the Fourier transform (Bessel function J1)
                annulus_weights[i * grid_size + j][0] = std::sqrt(3 * outer_radius) / (4 * rho) * gsl_sf_bessel_J1(2 * M_PI * outer_radius * rho) - disk_weights[i * grid_size + j][0];
            }
        }
    }
}

void SmoothLife::complex_vector_multiplication(fftw_complex* signal, fftw_complex* filter, fftw_complex* result) {
    for (int i = 0; i < grid_size * grid_size; ++i) {
        double signal_real = signal[i][0];
        double signal_imag = signal[i][1];

        double filter_real = filter[i][0];
        double filter_imag = filter[i][1];

        // Perform complex multiplication
        result[i][0] = signal_real * filter_real - signal_imag * filter_imag;  // Real part of result
        result[i][1] = signal_real * filter_imag + signal_imag * filter_real;  // Imaginary part of result
    }
}

void SmoothLife::update() {
    // execute fft
    fftw_execute(plan_fft2_field);

    // M_f
    complex_vector_multiplication(output_weights, disk_weights, M_fourier);
    
    // N_f
    complex_vector_multiplication(output_weights, annulus_weights, N_fourier);

    fftw_execute(plan_ifft2_M);
    fftw_execute(plan_ifft2_N);

    for (int i = 0; i < grid_size * grid_size; i++) {
        M_spatial[i][0] /= (grid_size * grid_size);  // Normalize real part
        M_spatial[i][1] /= (grid_size * grid_size);  // Normalize imaginary part

        N_spatial[i][0] /= (grid_size * grid_size);
        N_spatial[i][1] /= (grid_size * grid_size);
    }
    visualize2DVector(M_spatial, grid_size);
    
    S(M_spatial, N_spatial, input_weights);

    
    // visualize2DVector(N_spatial, grid_size);

    visualize2DVector(input_weights, grid_size);

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            field[i][j] = input_weights[i * grid_size + j][0];
        }
    }
}


void SmoothLife::updateFieldCUDA() {
    // Call CUDA function to update the field
    applyCudaUpdates();
}

void SmoothLife::visualize2DVector(fftw_complex* vec, int grid_size) {
    sf::RenderWindow window(sf::VideoMode(grid_size, grid_size), "2D Vector Visualization");

    // Create an image to display the 2D vector
    sf::Image image;
    image.create(grid_size, grid_size, sf::Color::Black);

    // Set pixels in the image based on the 2D vector values
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            // Normalize the value to a grayscale range [0, 255]
            sf::Uint8 color = static_cast<sf::Uint8>(vec[i * grid_size + j][0] * 255.0);
            image.setPixel(j, i, sf::Color(color, color, color));
        }
    }

    // Load the image into a texture and then into a sprite for rendering
    sf::Texture texture;
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);

    // Main window loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }
}