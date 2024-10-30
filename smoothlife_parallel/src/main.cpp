#include "smoothlife.hpp"
#include <iostream>

int main() {
    int grid_size = 64;
    double radius = 1.0;

    SmoothLife smoothlife(grid_size, radius);
    
    smoothlife.initializeField();
    // smoothlife.updateField();
    smoothlife.printField();
    
    std::cout << "Field updated!" << std::endl;
    
    return 0;
}
