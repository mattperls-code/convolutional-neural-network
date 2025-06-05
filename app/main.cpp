#include <iostream>

#include "../src/convolutional_neural_network.hpp"

int main()
{
    ConvolutionalNeuralNetwork cnn(
        Dimensions(3, Shape(24, 24)),
        {
            new ConvolutionLayerParameters(16, Shape(6, 6), 2, 4),
            new PoolLayerParameters(MAX, Shape(4, 4), 2, 0),
            new ActivationLayerParameters(RELU)
        },
        {
            HiddenLayerParameters(32, TANH),
            HiddenLayerParameters(24, TANH),
            HiddenLayerParameters(8, LINEAR)
        },
        SOFTMAX,
        CATEGORICAL_CROSS_ENTROPY
    );

    // cnn.initializeRandomLayerParameters();

    std::cout << cnn.toString() << std::endl;
};