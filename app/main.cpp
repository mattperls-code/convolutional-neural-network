#include <iostream>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/memory.hpp>

#include "../src/convolutional_neural_network.hpp"

int main()
{
    auto input = ImageOperations::pngToTensor("./app/img10x10.png");

    Matrix expectedOutput({{ 1.0 }, { 2.0 }, { -3.0 }});

    TensorDataPoint trainingDataPoint(input, expectedOutput);
    
    ConvolutionalNeuralNetwork cnn(
        Dimensions(3, Shape(10, 10)),
        {
            new PoolLayerParameters(AVG, Shape(2, 2), 1, 0),
            new ConvolutionLayerParameters(4, Shape(2, 2), 2, 1),
            new ActivationLayerParameters(TANH)
        },
        {
            HiddenLayerParameters(3, LINEAR)
        },
        IDENTITY,
        SUM_SQUARED_ERROR
    );

    auto loaded = cnn.load("./results/training/test.json");

    if (!loaded) {
        cnn.initializeRandomFeatureLayerParameters();
        cnn.initializeRandomHiddenLayerParameters(-0.5, 0.5, -0.5, 0.5);
    }

    for (int i = 0;i<20;i++) {
        cnn.train(trainingDataPoint, 0.005);

        std::cout << "Output after " << (i + 1) << " rounds of training: " << cnn.getNormalizedOutput().toString() << std::endl;
    }

    cnn.save("./results/training/test.json");

    return 0;
};