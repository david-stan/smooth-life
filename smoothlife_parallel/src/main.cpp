#include "smoothlife.hpp"
#include <iostream>
#include <SFML/Graphics.hpp>

int main() {
    int grid_size = 512;
    double radius = 5;

    SmoothLife smoothlife(grid_size, radius);
    // smoothlife.update();
    smoothlife.applyCudaUpdate();

    // Create an SFML window
    sf::RenderWindow window(sf::VideoMode(grid_size, grid_size), "SFML Animation");

    // Create a texture and sprite to display the grid
    sf::Texture texture;
    texture.create(grid_size, grid_size);
    sf::Sprite sprite(texture);

    // Main loop for the animation
    while (window.isOpen()) {
        // Handle events (e.g., close window)
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Update the grid for the next frame
        smoothlife.update();

        // Create an image to represent the grid
        sf::Image image;
        image.create(grid_size, grid_size, sf::Color::Black);

        // Set pixels based on the grid state
        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                // Normalize the value to a grayscale range [0, 255]
                sf::Uint8 color = static_cast<sf::Uint8>(smoothlife.getBuffer()[i * grid_size + j][0] * 255.0);
                image.setPixel(j, i, sf::Color(color, color, color));
            }
        }

        // Update the texture with the new image
        texture.update(image);

        // Clear, draw, and display the window
        window.clear();
        window.draw(sprite);
        window.display();

        // Control frame rate (e.g., 60 frames per second)
        sf::sleep(sf::milliseconds(40));
    }
    
    return 0;
}
