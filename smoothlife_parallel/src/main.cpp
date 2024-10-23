#include "smoothlife.hpp"
#include <iostream>

int main() {
    int grid_size = 256;
    double radius = 5.0;

    SmoothLife smoothlife(grid_size, radius);
    
    smoothlife.initializeField();
    smoothlife.updateField();
    
    std::cout << "Field updated!" << std::endl;
    
    return 0;
}
