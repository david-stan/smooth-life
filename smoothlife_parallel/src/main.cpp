#include "smoothlife.hpp"
#include <iostream>

int main() {
    int grid_size = 256;
    double radius = 4.4;

    SmoothLife smoothlife(grid_size, radius);
    
    smoothlife.update();
    std::cout << "Field updated!" << std::endl;
    
    return 0;
}
