#include <cuda_runtime.h>
#include <cufft.h>
#include "smoothlife.hpp"
#include "smoothlife_cuda.cuh"

#define BLOCK_SIZE 16 // threadsPerBlock(16, 16)
#define HALO 21

void precomputeCoefficients(int num_coefficients, float inner_radius, cufftComplex* h_disk_fft, cufftComplex* h_annulus_fft) {
    int half_size = num_coefficients / 2;
    float grid_spacing = 1.0 / num_coefficients;
    float outer_radius = 3 * inner_radius;

    // Iterate over all frequency components in Fourier space
    for (int i = 0; i < num_coefficients; ++i) {
        for (int j = 0; j < num_coefficients; ++j) {
            // Compute the radial frequency rho (distance to the origin in frequency space)
            int kx = (i < half_size) ? i : i - num_coefficients;  // Handle periodicity
            int ky = (j < half_size) ? j : j - num_coefficients;
            float rho = std::sqrt(kx * kx + ky * ky) * grid_spacing;

            // Avoid division by zero at the origin (low frequency components)
            if (rho == 0.0) {
                h_disk_fft[i * num_coefficients + j].x = M_PI * inner_radius * inner_radius;  // Area of the disk for the zero frequency
                h_annulus_fft[i * num_coefficients + j].x = M_PI * outer_radius * outer_radius - h_disk_fft[0].x;  // Area of the annulus for the zero frequency
            } else {
                // Compute the Fourier transform (Bessel function J1)
                float disk_coeff = (std::sqrt(3 * inner_radius) / (4 * rho)) * gsl_sf_bessel_J1(2 * M_PI * inner_radius * rho);
                if (disk_coeff > 0.066) {
                    h_disk_fft[i * num_coefficients + j].x = 0.0;
                } else {
                    h_disk_fft[i * num_coefficients + j].x = disk_coeff;
                }
                h_annulus_fft[i * num_coefficients + j].x = std::sqrt(3 * outer_radius) / (4 * rho) * gsl_sf_bessel_J1(2 * M_PI * outer_radius * rho) - h_disk_fft[i * num_coefficients + j].x;
            }
            h_disk_fft[i * num_coefficients + j].y = 0; // Imaginary parts
            h_annulus_fft[i * num_coefficients + j].y = 0;
        }
    }
}

__device__ void complex_vector_multiplication(int grid_size, cufftComplex* signal, cufftComplex* filter, cufftComplex* result) {
    for (int i = 0; i < grid_size * grid_size; ++i) {
        float signal_real = signal[i].x;
        float signal_imag = signal[i].y;

        float filter_real = filter[i].x;
        float filter_imag = filter[i].y;

        // Perform complex multiplication
        result[i].x = signal_real * filter_real - signal_imag * filter_imag;  // Real part
        result[i].y = signal_real * filter_imag + signal_imag * filter_real;  // Imaginary part
    }
}


void SmoothLife::applyCudaUpdate() {
    float* d_field_real;
    cudaMemcpy(d_field_real, &field[0][0], sizeof(double) * grid_size * grid_size, cudaMemcpyHostToDevice);

    cufftComplex* d_field_fft;  
    cufftHandle plan; 
    cudaMalloc((void**)&d_field_fft, sizeof(double) * grid_size * grid_size); 
    cufftPlan2d(&plan, grid_size, grid_size, CUFFT_R2C);
    cufftExecR2C(plan, d_field_real, d_field_fft);

    cufftComplex* h_disk_coeffs = new cufftComplex[grid_size * grid_size];
    cufftComplex* h_annulus_coeffs = new cufftComplex[grid_size * grid_size];

    // take Hermitian symmetry of a real-valued signal into account AND individual block + halo dims
    float num_coeffs = (BLOCK_SIZE + 2 * HALO) * ((BLOCK_SIZE + 2 * HALO) / 2 + 1);
    precomputeCoefficients(num_coeffs, inner_radius, h_disk_coeffs, h_annulus_coeffs);
    
    cufftComplex* d_disk_coeffs;
    cufftComplex* d_annulus_coeffs;
    cudaMalloc(&d_disk_coeffs, sizeof(cufftComplex) * num_coeffs);
    cudaMalloc(&d_annulus_coeffs, sizeof(cufftComplex) * num_coeffs);
    cudaMemcpy(d_disk_coeffs, h_disk_coeffs, sizeof(cufftComplex) * num_coeffs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_annulus_coeffs, h_annulus_coeffs, sizeof(cufftComplex) * num_coeffs, cudaMemcpyHostToDevice);

    // No longer needed after the cudaMemcpy
    delete[] h_disk_coeffs;
    delete[] h_annulus_coeffs;

    int reduced_grid_size = grid_size / 2 + 1; // hermitian
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((reduced_grid_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (reduced_grid_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
                   
    smooth_life_kernel<<<numBlocks, threadsPerBlock>>>(d_field_fft, d_disk_coeffs, d_annulus_coeffs, reduced_grid_size);
    
    // cudaMemcpy(&field[0][0], d_field, sizeof(double) * grid_size * grid_size, cudaMemcpyDeviceToHost);
    // cudaFree(d_field);
}

// CUDA Kernel for SmoothLife with Shared Memory
__global__ void smooth_life_kernel(cufftComplex* d_field_fft, cufftComplex* d_disk_coeffs, cufftComplex* d_annulus_coeffs, int grid_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + HALO;
    int ty = threadIdx.y + HALO;

    // Allocate shared memory for the block, including the halo region
    __shared__ cufftComplex shared_grid[BLOCK_SIZE + 2 * HALO][BLOCK_SIZE + 2 * HALO];

    // Load cells and neighbors into shared memory
    if (x < grid_size && y < grid_size) {
        shared_grid[ty][tx] = d_field_fft[y * grid_size + x];

        // Load the halo cells (neighbors)
        if (threadIdx.x < HALO) {
            shared_grid[ty][tx - HALO] = d_field_fft[y * grid_size + ((x - HALO + grid_size) % grid_size)];
            shared_grid[ty][tx + BLOCK_SIZE] = d_field_fft[y * grid_size + ((x + BLOCK_SIZE) % grid_size)];
        }
        if (threadIdx.y < HALO) {
            shared_grid[ty - HALO][tx] = d_field_fft[((y - HALO + grid_size) % grid_size) * grid_size + x];
            shared_grid[ty + BLOCK_SIZE][tx] = d_field_fft[((y + BLOCK_SIZE) % grid_size) * grid_size + x];
        }
        // Load corners of halo (optional but ensures full coverage)
        if (threadIdx.x < HALO && threadIdx.y < HALO) {
            shared_grid[ty - HALO][tx - HALO] = d_field_fft[((y - HALO + grid_size) % grid_size) * grid_size + ((x - HALO + grid_size) % grid_size)];
            shared_grid[ty - HALO][tx + BLOCK_SIZE] = d_field_fft[((y - HALO + grid_size) % grid_size) * grid_size + ((x + BLOCK_SIZE) % grid_size)];
            shared_grid[ty + BLOCK_SIZE][tx - HALO] = d_field_fft[((y + BLOCK_SIZE) % grid_size) * grid_size + ((x - HALO + grid_size) % grid_size)];
            shared_grid[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = d_field_fft[((y + BLOCK_SIZE) % grid_size) * grid_size + ((x + BLOCK_SIZE) % grid_size)];
        }
    }
    __syncthreads();

    // Ensure we're within grid bounds and calculate M and N
    // if (x < grid_size && y < grid_size) {
    //     double M = shared_grid[ty][tx];  // Current cell's aliveness
    //     double N = 0.0;

    //     // Compute neighbor density (N) using shared memory
    //     for (int i = -HALO; i <= HALO; ++i) {
    //         for (int j = -HALO; j <= HALO; ++j) {
    //             if (i != 0 || j != 0) {
    //                 N += shared_grid[ty + i][tx + j];
    //             }
    //         }
    //     }
    //     N /= 8.0;  // Example for average of neighbors in a 3x3 grid

    //     // Apply SmoothLife transition
    //     double new_value = smooth_life_transition(M, N, alpha_M, alpha_N, B0, B1, D0, D1);

    //     // Write the result to the output grid
    //     new_grid[y * grid_size + x] = new_value;
    // }
}
