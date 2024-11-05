#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include "smoothlife.hpp"
#include "smoothlife_cuda.cuh"

#define BLOCK_SIZE 16 // threadsPerBlock(16, 16)
#define HALO 12

__device__ void complex_vector_multiplication(cufftDoubleComplex* signal, cufftDoubleComplex* filter, cufftDoubleComplex* result) {
    float signal_real = (*signal).x;
    float signal_imag = (*signal).y;

    float filter_real = (*filter).x;
    float filter_imag = (*filter).y;

    // Perform complex multiplication
    (*result).x = signal_real * filter_real - signal_imag * filter_imag;  // Real part
    (*result).y = signal_real * filter_imag + signal_imag * filter_real;  // Imaginary part
}


void SmoothLife::applyCudaUpdate() {
    cudaDeviceReset();

    // take Hermitian symmetry of a real-valued signal into account AND individual block + halo dims
    size_t num_coeffs = grid_size * (grid_size / 2 + 1);
    size_t num_coeffs_filter = (BLOCK_SIZE + 2 * HALO) * (BLOCK_SIZE + 2 * HALO);

    cufftDoubleReal* d_field_real;
    cufftDoubleComplex* d_field_fft;
    cufftDoubleComplex* d_field_fft_M;
    cufftDoubleComplex* d_field_fft_N;
    
    cufftDoubleReal* d_field_real_output;

    cudaMalloc(&d_field_real, sizeof(cufftDoubleReal) * grid_size * grid_size);
    cudaMalloc(&d_field_fft, sizeof(cufftDoubleComplex) * num_coeffs);

    cudaMalloc(&d_field_fft_M, sizeof(cufftDoubleComplex) * num_coeffs);
    cudaMalloc(&d_field_fft_N, sizeof(cufftDoubleComplex) * num_coeffs);
    cudaMalloc(&d_field_real_output, sizeof(cufftDoubleReal) * grid_size * grid_size);


    cudaMemcpy(d_field_real, input_weights_real, sizeof(double) * grid_size * grid_size, cudaMemcpyHostToDevice);

    cufftHandle plan; 
     
    cufftPlan2d(&plan, grid_size, grid_size, CUFFT_D2Z);
    cufftExecD2Z(plan, d_field_real, d_field_fft);

    std::cout << "ejoo123456" << std::endl;

    cufftDoubleComplex* h_disk_coeffs = new cufftDoubleComplex[num_coeffs_filter];
    cufftDoubleComplex* h_annulus_coeffs = new cufftDoubleComplex[num_coeffs_filter];

    for (int i = 0; i < num_coeffs_filter; i++) {
        h_disk_coeffs[i].x = disk_weights_cuda[i][0];
        h_disk_coeffs[i].y = 0;
        h_annulus_coeffs[i].x = annulus_weights_cuda[i][0];
        h_annulus_coeffs[i].y = 0;
    }

    cufftDoubleComplex* d_disk_coeffs;
    cufftDoubleComplex* d_annulus_coeffs;
    cudaMalloc(&d_disk_coeffs, sizeof(cufftDoubleComplex) * num_coeffs);
    cudaMalloc(&d_annulus_coeffs, sizeof(cufftDoubleComplex) * num_coeffs);

    cudaMemcpy(d_disk_coeffs, h_disk_coeffs, sizeof(cufftDoubleComplex) * num_coeffs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_annulus_coeffs, h_annulus_coeffs, sizeof(cufftDoubleComplex) * num_coeffs, cudaMemcpyHostToDevice);

    // No longer needed after the cudaMemcpy
    delete[] h_disk_coeffs;
    delete[] h_annulus_coeffs;

    std::cout << "Izdrzao" << std::endl;

    int reduced_grid_size = grid_size / 2 + 1; // hermitian
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    int numBlocksX = (reduced_grid_size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int numBlocksY = (grid_size + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 numBlocks(numBlocksX, numBlocksY);
    // dim3 numBlocks((reduced_grid_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                (reduced_grid_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
                   
    size_t height = grid_size;
    size_t width = reduced_grid_size;

    smooth_life_kernel<<<numBlocks, threadsPerBlock>>>(d_field_fft, d_disk_coeffs, d_annulus_coeffs, d_field_fft_M, d_field_fft_N, height, width);

    std::cout << "Jjss" << std::endl;
    
    cufftDoubleComplex* h_field_fft_M = new cufftDoubleComplex[num_coeffs];
    cufftDoubleComplex* h_field_fft_N = new cufftDoubleComplex[num_coeffs];
    cudaMemcpy(h_field_fft_M, d_field_fft_M, sizeof(cufftDoubleComplex) * num_coeffs, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_field_fft_N, d_field_fft_N, sizeof(cufftDoubleComplex) * num_coeffs, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < num_coeffs; i++) {
        std::cout << h_field_fft_M[i].x;
        std::cout << h_field_fft_N[i].x;
    }
    // cudaMemcpy(&field[0][0], d_field, sizeof(double) * grid_size * grid_size, cudaMemcpyDeviceToHost);
    // cudaFree(d_field);
}

// CUDA Kernel for SmoothLife with Shared Memory
__global__ void smooth_life_kernel(cufftDoubleComplex* d_field_fft, cufftDoubleComplex* d_disk_coeffs, cufftDoubleComplex* d_annulus_coeffs, 
                                    cufftDoubleComplex* d_field_fft_M, cufftDoubleComplex* d_field_fft_N, size_t height, size_t width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;      // global
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + HALO;                        // local
    int ty = threadIdx.y + HALO;

    // Allocate shared memory for the block, including the halo region
    __shared__ cufftDoubleComplex shared_grid[BLOCK_SIZE + 2 * HALO][BLOCK_SIZE + 2 * HALO];

    // Load cells and neighbors into shared memory
    if (x < width && y < height) {
        shared_grid[ty][tx] = d_field_fft[y * height + x];

        // Load the halo cells (neighbors)
        if (threadIdx.x < HALO) {
            shared_grid[ty][tx - HALO] = d_field_fft[y * height + ((x - HALO + width) % width)];
            shared_grid[ty][tx + BLOCK_SIZE] = d_field_fft[y * height + ((x + BLOCK_SIZE) % width)];
        }
        if (threadIdx.y < HALO) {
            shared_grid[ty - HALO][tx] = d_field_fft[((y - HALO + height) % height) * height + x];
            shared_grid[ty + BLOCK_SIZE][tx] = d_field_fft[((y + BLOCK_SIZE) % height) * height + x];
        }
        // Load corners of halo (optional but ensures full coverage)
        // if (threadIdx.x < HALO && threadIdx.y < HALO) {
        //     shared_grid[ty - HALO][tx - HALO] = d_field_fft[((y - HALO + height) % height) * height + ((x - HALO + width) % width)];
        //     shared_grid[ty - HALO][tx + BLOCK_SIZE] = d_field_fft[((y - HALO + height) % height) * height + ((x + BLOCK_SIZE) % width)];
        //     shared_grid[ty + BLOCK_SIZE][tx - HALO] = d_field_fft[((y + BLOCK_SIZE) % height) * height + ((x - HALO + width) % width)];
        //     shared_grid[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = d_field_fft[((y + BLOCK_SIZE) % height) * height + ((x + BLOCK_SIZE) % width)];
        // }
    }
    __syncthreads();

    // Ensure we're within grid bounds and calculate M and N
    if (x < width && y < height) {
        
        cufftDoubleComplex* shared_value = &(shared_grid[ty][tx]);
        cufftDoubleComplex* filter_value_disk = &(d_disk_coeffs[y * (BLOCK_SIZE + 2 * HALO) + x]);
        cufftDoubleComplex* filter_value_annulus = &(d_annulus_coeffs[y * (BLOCK_SIZE + 2 * HALO) + x]);
    
        // M_f
        complex_vector_multiplication(shared_value, filter_value_disk, d_field_fft_M);
        // N_f
        complex_vector_multiplication(shared_value, filter_value_annulus, d_field_fft_N);
    }
}
