#include "smoothlife.hpp"
#include <iostream>

int main() {
    int grid_size = 128;
    double radius = 3.0;

    SmoothLife smoothlife(grid_size, radius);
    
    smoothlife.printField();
    smoothlife.update();
    std::cout << "Field updated!" << std::endl;
    smoothlife.printField();
    
    return 0;
}
