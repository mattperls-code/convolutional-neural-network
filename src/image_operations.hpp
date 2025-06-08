#ifndef IMAGE_OPERATIONS_HPP
#define IMAGE_OPERATIONS_HPP

#include "tensor.hpp"

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
        static Matrix crop(const Matrix& image, int x, int y, const Shape& newSize, float fillValue);
        static Matrix translate(const Matrix& image, int x, int y, float fillValue);
        static Matrix rotate(const Matrix& image, float theta, float fillValue);

        static Tensor rgbToGreyscale(const Tensor& image);
        static Tensor greyscaleToRgb(const Tensor& image);

        static Tensor pngToTensor(const std::string& pngFilePath);
        static void tensorToPng(const std::string& outputFilePath, const Tensor& tensor);
};

#endif