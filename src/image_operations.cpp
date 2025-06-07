#include "image_operations.hpp"

#include <lodepng/lodepng.h>

Matrix ImageOperations::pad(const Matrix& image, float value, int leftPadding, int rightPadding, int topPadding, int bottomPadding)
{
    if (leftPadding == 0 && rightPadding == 0 && topPadding == 0 && bottomPadding == 0) return image;

    if (leftPadding < 0 || rightPadding < 0 || topPadding < 0 || bottomPadding < 0) throw std::runtime_error("ImageOperations pad: padding cannot be negative");

    Matrix output(Shape(image.rowCount() + topPadding + bottomPadding, image.colCount() + leftPadding + rightPadding), value);

    for (int r = 0;r<image.rowCount();r++) {
        for (int c = 0;c<image.colCount();c++) {
            output.set(r + topPadding, c + leftPadding, image.get(r, c));
        }
    }

    return output;
};

Matrix ImageOperations::pad(const Matrix& image, float value, int padding)
{
    return pad(image, value, padding, padding, padding, padding);
};

Matrix ImageOperations::pad(const Matrix& image, int padding)
{
    return pad(image, 0.0, padding);
};

Matrix ImageOperations::convolution(const Matrix& image, const Matrix& kernel, int stride)
{
    if (image.empty()) throw std::runtime_error("ImageOperations convolution: image is empty");
    if (kernel.empty()) throw std::runtime_error("ImageOperations convolution: kernel is empty");

    if (image.rowCount() < kernel.rowCount() || image.colCount() < kernel.colCount()) throw std::runtime_error("ImageOperations convolution: kernel dimension larger than image dimension");

    if (stride < 1) throw std::runtime_error("ImageOperations convolution: stride must be at least 1");

    Matrix output(Shape(
        1 + (image.rowCount() - kernel.rowCount()) / stride,
        1 + (image.colCount() - kernel.colCount()) / stride
    ));

    for (int outputR = 0;outputR<output.rowCount();outputR++) {
        auto imageR = outputR * stride;

        for (int outputC = 0;outputC<output.colCount();outputC++) {
            auto imageC = outputC * stride;

            auto sum = 0.0;

            for (int kernelR = 0;kernelR<kernel.rowCount();kernelR++) {
                for (int kernelC = 0;kernelC<kernel.colCount();kernelC++) {
                    sum += image.get(imageR + kernelR, imageC + kernelC) * kernel.get(kernelR, kernelC);
                }
            }

            output.set(outputR, outputC, sum);
        }
    }

    return output;
};

Matrix ImageOperations::minPool(const Matrix& image, const Shape& window, int stride)
{
    if (image.empty()) throw std::runtime_error("ImageOperations minPool: image is empty");
    if (window.rows == 0 || window.cols == 0) throw std::runtime_error("ImageOperations minPool: window is empty");

    if (image.rowCount() < window.rows || image.colCount() < window.cols) throw std::runtime_error("ImageOperations minPool: window dimension larger than image dimension");

    if (stride < 1) throw std::runtime_error("ImageOperations minPool: stride must be at least 1");

    Matrix output(Shape(
        1 + (image.rowCount() - window.rows) / stride,
        1 + (image.colCount() - window.cols) / stride
    ));

    for (int outputR = 0;outputR<output.rowCount();outputR++) {
        auto imageR = outputR * stride;

        for (int outputC = 0;outputC<output.colCount();outputC++) {
            auto imageC = outputC * stride;

            auto windowMin = image.get(imageR, imageC);

            for (int windowR = 0;windowR<window.rows;windowR++) {
                for (int windowC = 0;windowC<window.cols;windowC++) {
                    auto pixel = image.get(imageR + windowR, imageC + windowC);

                    if (pixel < windowMin) windowMin = pixel;
                }
            }

            output.set(outputR, outputC, windowMin);
        }
    }

    return output;
};

Matrix ImageOperations::maxPool(const Matrix& image, const Shape& window, int stride)
{
    if (image.empty()) throw std::runtime_error("ImageOperations maxPool: image is empty");
    if (window.rows == 0 || window.cols == 0) throw std::runtime_error("ImageOperations maxPool: window is empty");

    if (image.rowCount() < window.rows || image.colCount() < window.cols) throw std::runtime_error("ImageOperations maxPool: window dimension larger than image dimension");

    if (stride < 1) throw std::runtime_error("ImageOperations maxPool: stride must be at least 1");

    Matrix output(Shape(
        1 + (image.rowCount() - window.rows) / stride,
        1 + (image.colCount() - window.cols) / stride
    ));

    for (int outputR = 0;outputR<output.rowCount();outputR++) {
        auto imageR = outputR * stride;

        for (int outputC = 0;outputC<output.colCount();outputC++) {
            auto imageC = outputC * stride;

            auto windowMax = image.get(imageR, imageC);

            for (int windowR = 0;windowR<window.rows;windowR++) {
                for (int windowC = 0;windowC<window.cols;windowC++) {
                    auto pixel = image.get(imageR + windowR, imageC + windowC);

                    if (pixel > windowMax) windowMax = pixel;
                }
            }

            output.set(outputR, outputC, windowMax);
        }
    }

    return output;
};

Matrix ImageOperations::avgPool(const Matrix& image, const Shape& window, int stride)
{
    if (image.empty()) throw std::runtime_error("ImageOperations avgPool: image is empty");
    if (window.rows == 0 || window.cols == 0) throw std::runtime_error("ImageOperations avgPool: window is empty");

    if (image.rowCount() < window.rows || image.colCount() < window.cols) throw std::runtime_error("ImageOperations avgPool: window dimension larger than image dimension");

    if (stride < 1) throw std::runtime_error("ImageOperations avgPool: stride must be at least 1");

    Matrix output(Shape(
        1 + (image.rowCount() - window.rows) / stride,
        1 + (image.colCount() - window.cols) / stride
    ));

    int windowArea = window.rows * window.cols;

    for (int outputR = 0;outputR<output.rowCount();outputR++) {
        auto imageR = outputR * stride;

        for (int outputC = 0;outputC<output.colCount();outputC++) {
            auto imageC = outputC * stride;

            auto windowSum = 0.0;

            for (int windowR = 0;windowR<window.rows;windowR++) {
                for (int windowC = 0;windowC<window.cols;windowC++) {
                    windowSum += image.get(imageR + windowR, imageC + windowC);
                }
            }

            output.set(outputR, outputC, windowSum / windowArea);
        }
    }

    return output;
};

Matrix ImageOperations::resize(const Matrix& image, const Shape& newSize)
{
    if (image.empty()) throw std::runtime_error("ImageOperations resize: image is empty");
    if (newSize.rows == 0 || newSize.cols == 0) throw std::runtime_error("ImageOperations resize: newSize is empty");

    Matrix output(newSize);

    auto rScalar = (float) (image.rowCount() - 1) / (newSize.rows - 1);
    auto cScalar = (float) (image.colCount() - 1) / (newSize.cols - 1);

    for (int r = 0;r<output.rowCount();r++) {
        auto imageR = round(rScalar * r);

        for (int c = 0;c<output.colCount();c++) {
            auto imageC = round(cScalar * c);

            output.set(r, c, image.get(imageR, imageC));
        }
    }

    return output;
};

Matrix ImageOperations::crop(const Matrix& image, int x, int y, const Shape& cropWindow, float fillValue)
{
    if (image.empty()) throw std::runtime_error("ImageOperations crop: image is empty");
    if (cropWindow.rows == 0 || cropWindow.cols == 0) throw std::runtime_error("ImageOperations crop: cropWindow is empty");

    Matrix output(cropWindow, fillValue);

    for (int r = 0;r<cropWindow.rows;r++) {
        auto imageR = r + y;
        if (imageR < 0) continue;
        if (imageR >= image.rowCount()) break;

        for (int c = 0;c<cropWindow.cols;c++) {
            auto imageC = c + x;
            if (imageC < 0) continue;
            if (imageC >= image.colCount()) break;

            output.set(r, c, image.get(imageR, imageC));
        }
    }

    return output;
};

Matrix ImageOperations::translate(const Matrix& image, int x, int y, float fillValue)
{
    if (image.empty()) throw std::runtime_error("ImageOperations translate: image is empty");
    
    Matrix output(image.shape(), fillValue);

    for (int r = 0;r<output.rowCount();r++) {
        int imageR = r - y;
        if (imageR < 0) continue;
        if (imageR >= image.rowCount()) break;

        for (int c = 0;c<output.colCount();c++) {
            int imageC = c - x;
            if (imageC < 0) continue;
            if (imageC >= image.colCount()) break;

            output.set(r, c, image.get(imageR, imageC));
        }
    }

    return output;
};

Matrix ImageOperations::rotate(const Matrix& image, float theta, float fillValue)
{
    if (image.empty()) throw std::runtime_error("ImageOperations rotate: image is empty");

    Matrix output(image.shape(), fillValue);

    auto cosTheta = cos(theta);
    auto sinTheta = sin(theta);

    for (int r = 0;r<output.rowCount();r++) {
        for (int c = 0;c<output.colCount();c++) {
            auto deltaX = c + 0.5 - 0.5 * output.colCount();
            auto deltaY = r + 0.5 - 0.5 * output.rowCount();

            auto rotatedX = deltaX * cosTheta - deltaY * sinTheta;
            auto rotatedY = deltaX * sinTheta + deltaY * cosTheta;

            auto imageR = round(rotatedY - 0.5 + 0.5 * output.rowCount());
            auto imageC = round(rotatedX - 0.5 + 0.5 * output.colCount());

            if (imageR >= 0 && imageC >= 0 && imageR < image.rowCount() && imageC < image.colCount()) output.set(r, c, image.get(imageR, imageC));
        }
    }

    return output;
};

Tensor ImageOperations::pngToTensor(const std::string& pngFilePath)
{
    std::vector<unsigned char> imageData;

    unsigned int imageWidth, imageHeight;

    auto error = lodepng::decode(imageData, imageWidth, imageHeight, pngFilePath);

    if (error) throw std::runtime_error("ImageOperations pngToTensor: error opening " + pngFilePath + ", " + lodepng_error_text(error));

    Tensor output(Dimensions(3, Shape(imageHeight, imageWidth)));

    for (int r = 0;r<imageHeight;r++) {
        for (int c = 0;c<imageWidth;c++) {
            auto offset = 4 * (r * imageWidth + c);

            for (int i = 0;i<3;i++) output.set(r, c, i, (float) imageData[offset + i] / 256);
        }
    }

    return output;
};