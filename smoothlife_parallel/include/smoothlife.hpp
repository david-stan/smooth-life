#ifndef SMOOTHLIFE_HPP
#define SMOOTHLIFE_HPP

#include <vector>
#include <fftw3.h>

class SmoothLife {
public:
    SmoothLife(int grid_size, double radius);
    ~SmoothLife();
    void initializeField();
    void initializeWeights();
    void printField();
    void updateField();
    void updateFieldCUDA();
    
private:
    int grid_size;
    double radius;
    std::vector<std::vector<double>> field;

    fftw_complex* input_weights;
    fftw_complex* output_weights;

    fftw_plan plan_fft2;
    fftw_plan plan_ifft2;

    // CUDA-accelerated functions
    void applyCudaUpdates();
    void applyCudaFFT();
};

#endif // SMOOTHLIFE_HPP
