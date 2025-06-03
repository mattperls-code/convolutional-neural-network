#ifndef IMAGE_OPERATIONS_HPP
#define IMAGE_OPERATIONS_HPP

#include "../lib/matrix.hpp"

class ImageOperations
{
    public:
        static Matrix pad(const Matrix& image, float value, int leftPadding, int rightPadding, int topPadding, int bottomPadding);
        static Matrix pad(const Matrix& image, float value, int padding);
        static Matrix pad(const Matrix& image, int padding);

        static Matrix convolution(const Matrix& image, const Matrix& kernel, int stride);

        static Matrix minPool(const Matrix& image, const Shape& window, int stride);
        static Matrix maxPool(const Matrix& image, const Shape& window, int stride);
        static Matrix avgPool(const Matrix& image, const Shape& window, int stride);
        
        static Matrix resize(const Matrix& image, const Shape& newSize);
        static Matrix crop(const Matrix& image, int x, int y, const Shape& newSize, float replaceValue);
        static Matrix translate(const Matrix& image, int x, int y, float replaceValue);
        static Matrix rotate(const Matrix& image, int theta, float replaceValue);
};

#endif