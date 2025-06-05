#include <catch2/catch_all.hpp>
#include <iostream>

#include "../src/image_operations.hpp"

#include "util.hpp"

TEST_CASE("IMAGE OPERATIONS") {
    SECTION("PAD") {
        Matrix input({
            { 0.0, 0.3, -0.4 },
            { 0.5, 0.1, 0.2 }
        });

        Matrix expectedOutput({
            { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
            { 1.0, 0.0, 0.3, -0.4, 1.0, 1.0 },
            { 1.0, 0.5, 0.1, 0.2, 1.0, 1.0 }
        });

        auto observedOutput = ImageOperations::pad(input, 1.0, 1, 2, 1, 0);

        REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput, 0.0));
    }

    SECTION("CONVOLUTION") {
        Matrix inputImage({
            { 0.1, 0.2, 0.3, 0.4, 0.5 },
            { 0.6, 0.7, 0.8, 0.9, 1.0 },
            { 1.1, 1.2, 1.3, 1.4, 1.5 },
            { 1.6, 1.7, 1.8, 1.9, 2.0 },
        });

        Matrix inputKernel({
            { 0.0, 1.0 },
            { -1.0, 0.5 }
        });

        auto inputStride = 2;

        Matrix expectedOutput({
            { -0.05, 0.05 },
            { 0.45, 0.55 }
        });

        auto observedOutput = ImageOperations::convolution(inputImage, inputKernel, inputStride);

        REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput, 0.0));
    }

    SECTION("MIN POOL") {
        Matrix inputImage({
            { -1.0, 0.5, 0.4 },
            { 0.0, -0.3, 0.1 }
        });

        Shape inputWindow(1, 2);

        auto inputStride = 1;

        Matrix expectedOutput({
            { -1.0, 0.4 },
            { -0.3, -0.3 }
        });

        auto observedOutput = ImageOperations::minPool(inputImage, inputWindow, inputStride);

        REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput, 0.0));
    }

    SECTION("MAX POOL") {
        Matrix inputImage({
            { -1.0, 0.5, 0.4 },
            { 0.0, -0.3, 0.1 }
        });

        Shape inputWindow(2, 1);

        auto inputStride = 2;

        Matrix expectedOutput(std::vector<std::vector<float>>({{ 0.0, 0.4 }}));

        auto observedOutput = ImageOperations::maxPool(inputImage, inputWindow, inputStride);

        REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput, 0.0));
    }

    SECTION("AVG POOL") {
        Matrix inputImage({
            { -1.0, 0.5, 0.4 },
            { 0.0, -0.3, 0.1 }
        });

        Shape inputWindow(2, 2);

        auto inputStride = 1;

        Matrix expectedOutput(std::vector<std::vector<float>>({{ -0.2, 0.175 }}));

        auto observedOutput = ImageOperations::avgPool(inputImage, inputWindow, inputStride);

        REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput, 0.0));
    }

    SECTION("RESIZE DOWN") {
        Matrix inputImage({
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });
    
        Shape inputNewSize(3, 4);
    
        Matrix expectedOutput({
            { 1.0, 1.0, 2.0, 2.0 },
            { 3.0, 3.0, 4.0, 4.0 },
            { 3.0, 3.0, 4.0, 4.0 }
        });
    
        auto observedOutput = ImageOperations::resize(inputImage, inputNewSize);
    
        REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput, 0.0));
    }

    SECTION("RESIZE UP") {
        Matrix inputImage({
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });
    
        Shape inputNewSize(5, 2);
    
        Matrix expectedOutput({
            { 1.0, 3.0 },
            { 1.0, 3.0 },
            { 4.0, 6.0 },
            { 4.0, 6.0 },
            { 4.0, 6.0 },
        });
    
        auto observedOutput = ImageOperations::resize(inputImage, inputNewSize);
    
        REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput, 0.0));
    }
    
    SECTION("CROP") {
        Matrix inputImage({
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
    
        auto inputX = 1;
        auto inputY = 1;
        Shape inputNewSize(2, 3);
        auto inputFillValue = -1.0;
    
        Matrix expectedOutput({
            { 5.0, 6.0, -1.0 },
            { 8.0, 9.0, -1.0 }
        });
    
        auto observedOutput = ImageOperations::crop(inputImage, inputX, inputY, inputNewSize, inputFillValue);
    
        REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput, 0.0));
    }

    SECTION("TRANSLATE") {
        Matrix inputImage({
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });
    
        auto inputX = 1;
        auto inputY = -1;
        auto inputFillValue = 0.0;
    
        Matrix expectedOutput({
            { 0.0, 3.0 },
            { 0.0, 0.0 }
        });
    
        auto observedOutput = ImageOperations::translate(inputImage, inputX, inputY, inputFillValue);
    
        REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput, 0.0));
    }

    SECTION("ROTATE") {
        Matrix inputImage({
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
    
        auto inputTheta = M_PI_2;
        auto inputFillValue = 0.0;
    
        Matrix expectedOutput({
            { 3.0, 6.0, 9.0 },
            { 2.0, 5.0, 8.0 },
            { 1.0, 4.0, 7.0 }
        });
    
        auto observedOutput = ImageOperations::rotate(inputImage, inputTheta, inputFillValue);
    
        REQUIRE(matricesAreApproxEqual(observedOutput, expectedOutput, 0.0));
    }
}