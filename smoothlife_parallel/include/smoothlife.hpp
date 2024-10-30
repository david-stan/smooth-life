#ifndef SMOOTHLIFE_HPP
#define SMOOTHLIFE_HPP

#include <vector>
#include <fftw3.h>
#include <gsl/gsl_sf_bessel.h>

class SmoothLife {
public:
    SmoothLife(int grid_size, double radius);
    ~SmoothLife();
    
    void update();
    void updateFieldCUDA();
    void visualize2DVector(fftw_complex* vector, int grid_size);

    static double B0;
    static double B1;
    static double D0;
    static double D1;
    static double alpha_M;
    static double alpha_N;
    
private:
    int grid_size;
    double inner_radius, outer_radius;
    std::vector<std::vector<double>> field;

    fftw_complex* input_weights;
    fftw_complex* output_weights;

    fftw_complex* disk_weights;
    fftw_complex* annulus_weights;

    fftw_complex* M_fourier;
    fftw_complex* M_spatial;
    fftw_complex* N_fourier;
    fftw_complex* N_spatial;

    fftw_plan plan_fft2_field;
    fftw_plan plan_ifft2_M;
    fftw_plan plan_ifft2_N;

    void initializeWeightsDisk();
    void initializeWeightsAnnulus();

    double logistic_threshold(double x, double x0, double alpha);
    double logistic_interval(double x, double a, double b, double alpha);
    double lerp(double a, double b, double t);
    void S(fftw_complex* M, fftw_complex* N, fftw_complex* output);
    void complex_vector_multiplication(fftw_complex* signal, fftw_complex* filter, fftw_complex* result);

    // CUDA-accelerated functions
    void applyCudaUpdates();
    void applyCudaFFT();
};

#endif // SMOOTHLIFE_HPP
