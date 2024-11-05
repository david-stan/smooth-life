#include <cuda_runtime.h>

__global__ void smooth_life_kernel(cufftDoubleComplex* d_field_fft,
                                   cufftDoubleComplex* d_disk_coeffs,
                                   cufftDoubleComplex* d_annulus_coeffs,
                                   cufftDoubleComplex* d_field_fft_M,
                                   cufftDoubleComplex* d_field_fft_N, size_t height, size_t width);