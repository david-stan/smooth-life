#ifndef SMOOTHLIFE_HPP
#define SMOOTHLIFE_HPP

#include <vector>

class SmoothLife {
public:
    SmoothLife(int grid_size, double radius);
    void initializeField();
    void updateField();
    void computeFFT();
    
private:
    int grid_size;
    double radius;
    std::vector<std::vector<double>> field;

    // CUDA-accelerated functions
    void applyCudaUpdates();
    void applyCudaFFT();
};

#endif // SMOOTHLIFE_HPP
