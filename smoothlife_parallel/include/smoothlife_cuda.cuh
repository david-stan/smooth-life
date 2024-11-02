#include <cuda_runtime.h>

__global__ void smooth_life_kernel(cufftComplex* d_field_fft, cufftComplex* d_disk_coeffs, cufftComplex* d_annulus_coeffs, int grid_size);