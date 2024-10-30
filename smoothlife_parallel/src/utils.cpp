#include "utils.hpp"

void initializeRandomCircles(std::vector<std::vector<double>>& field, int grid_size, int num_circles, double min_radius, double max_radius)
{
    // Seed the random number generator
    std::srand(std::time(0));

    // Iterate over the number of circles to generate
    for (int n = 0; n < num_circles; ++n) {
        // Randomly generate the circle's center
        int center_x = std::rand() % grid_size;
        int center_y = std::rand() % grid_size;
        
        // Randomly generate the radius of the circle
        double radius = min_radius + (std::rand() / (RAND_MAX / (max_radius - min_radius)));

        // Iterate over the entire grid and set values within the circle to 1
        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                // Compute the distance from the current point (i, j) to the circle center (center_x, center_y)
                double distance = std::sqrt((i - center_x) * (i - center_x) + (j - center_y) * (j - center_y));

                // If the distance is less than the radius, set the field value to 1
                if (distance <= radius) {
                    field[i][j] = 1.0;
                }
            }
        }
    }
}