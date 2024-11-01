#include <cuda_runtime.h>
#include <cufft.h>
#include "../include/smoothlife.hpp"

#define HALO 21

// Example CUDA kernel
__global__ void update_kernel(double* field, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < grid_size && idy < grid_size) {
        int index = idy * grid_size + idx;
        // Perform update logic here
        field[index] *= 0.99; // Example update rule
    }
}

void SmoothLife::applyCudaUpdates() {
    double* d_field;
    cudaMalloc((void**)&d_field, sizeof(double) * grid_size * grid_size);
    cudaMemcpy(d_field, &field[0][0], sizeof(double) * grid_size * grid_size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((grid_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (grid_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
                   
    update_kernel<<<numBlocks, threadsPerBlock>>>(d_field, grid_size);
    
    cudaMemcpy(&field[0][0], d_field, sizeof(double) * grid_size * grid_size, cudaMemcpyDeviceToHost);
    cudaFree(d_field);
}

// CUDA Kernel for SmoothLife with Shared Memory
__global__ void smooth_life_kernel(double* grid, double* new_grid, int grid_size, double alpha_M, double alpha_N, double B0, double B1, double D0, double D1) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + HALO;
    int ty = threadIdx.y + HALO;

    // Allocate shared memory for the block, including the halo region
    __shared__ double shared_grid[BLOCK_SIZE + 2 * HALO][BLOCK_SIZE + 2 * HALO];

    // Load cells and neighbors into shared memory
    if (x < grid_size && y < grid_size) {
        shared_grid[ty][tx] = grid[y * grid_size + x];

        // Load the halo cells (neighbors)
        if (threadIdx.x < HALO) {
            shared_grid[ty][tx - HALO] = grid[y * grid_size + ((x - HALO + grid_size) % grid_size)];
            shared_grid[ty][tx + BLOCK_SIZE] = grid[y * grid_size + ((x + BLOCK_SIZE) % grid_size)];
        }
        if (threadIdx.y < HALO) {
            shared_grid[ty - HALO][tx] = grid[((y - HALO + grid_size) % grid_size) * grid_size + x];
            shared_grid[ty + BLOCK_SIZE][tx] = grid[((y + BLOCK_SIZE) % grid_size) * grid_size + x];
        }
        // Load corners of halo (optional but ensures full coverage)
        if (threadIdx.x < HALO && threadIdx.y < HALO) {
            shared_grid[ty - HALO][tx - HALO] = grid[((y - HALO + grid_size) % grid_size) * grid_size + ((x - HALO + grid_size) % grid_size)];
            shared_grid[ty - HALO][tx + BLOCK_SIZE] = grid[((y - HALO + grid_size) % grid_size) * grid_size + ((x + BLOCK_SIZE) % grid_size)];
            shared_grid[ty + BLOCK_SIZE][tx - HALO] = grid[((y + BLOCK_SIZE) % grid_size) * grid_size + ((x - HALO + grid_size) % grid_size)];
            shared_grid[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = grid[((y + BLOCK_SIZE) % grid_size) * grid_size + ((x + BLOCK_SIZE) % grid_size)];
        }
    }
    __syncthreads();

    // Ensure we're within grid bounds and calculate M and N
    if (x < grid_size && y < grid_size) {
        double M = shared_grid[ty][tx];  // Current cell's aliveness
        double N = 0.0;

        // Compute neighbor density (N) using shared memory
        for (int i = -HALO; i <= HALO; ++i) {
            for (int j = -HALO; j <= HALO; ++j) {
                if (i != 0 || j != 0) {
                    N += shared_grid[ty + i][tx + j];
                }
            }
        }
        N /= 8.0;  // Example for average of neighbors in a 3x3 grid

        // Apply SmoothLife transition
        double new_value = smooth_life_transition(M, N, alpha_M, alpha_N, B0, B1, D0, D1);

        // Write the result to the output grid
        new_grid[y * grid_size + x] = new_value;
    }
}