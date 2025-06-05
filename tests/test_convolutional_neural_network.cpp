#include <catch2/catch_all.hpp>
#include <iostream>

#include "../src/convolutional_neural_network.hpp"

#include "util.hpp"

TEST_CASE("CONVOLUTIONAL NEURAL NETWORK") {
    ConvolutionalNeuralNetwork cnn(
        Dimensions(3, Shape(7, 7)),
        {
            new PoolLayerParameters(MAX, Shape(2, 2), 1, 0),
            new ConvolutionLayerParameters({
                Tensor({
                    Matrix({
                        { 0.0, 1.0 },
                        { 1.0, 1.0}
                    }),
                    Matrix({
                        { -0.5, 0.0 },
                        { 0.5, 1.0 }
                    }),
                    Matrix({
                        { -1.0, -1.0 },
                        { 0.0, 0.0}
                    }),
                }),
                Tensor({
                    Matrix({
                        { 0.0, 0.5 },
                        { -0.5, 0.0 }
                    }),
                    Matrix({
                        { 1.0, -1.0 },
                        { -0.5, 0.5}
                    }),
                    Matrix({
                        { 0.0, 0.5 },
                        { -0.5, 0.5 }
                    }),
                })
            }, 2, 1),
            new ActivationLayerParameters(RELU, std::vector<float>({ 0.3, 0.6 }), std::vector<float>({ 0.1, -0.2 }))
        },
        {
            HiddenLayerParameters(6, TANH),
            HiddenLayerParameters(4, LINEAR)
        },
        SOFTMAX,
        CATEGORICAL_CROSS_ENTROPY
    );

    Tensor input({
        Matrix({
            { 0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.2 },
            { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0 },
            { 0.4, 0.6, 0.8, 1.0, 0.0, 0.2, 0.4 },
            { 0.6, 0.8, 1.0, 0.0, 0.2, 0.4, 0.6 },
            { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0 },
            { 1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 },
            { 0.4, 0.6, 0.8, 1.0, 0.0, 0.2, 0.4 },
        }),
        Matrix({
            { 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 1.0 },
            { 0.8, 1.0, 0.0, 0.2, 0.4, 0.6, 0.8 },
            { 0.6, 0.4, 0.2, 0.0, 1.0, 0.8, 0.6 },
            { 0.4, 0.2, 0.0, 1.0, 0.8, 0.6, 0.4 },
            { 0.8, 0.6, 0.4, 0.2, 0.0, 1.0, 0.8 },
            { 0.6, 0.8, 1.0, 0.0, 0.2, 0.4, 0.6 },
            { 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 1.0 }
        }),
        Matrix({
            { 0.2, 0.0, 1.0, 0.8, 0.6, 0.4, 0.2 },
            { 0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.2 },
            { 0.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0 },
            { 0.8, 1.0, 0.0, 0.2, 0.4, 0.6, 0.8 },
            { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0 },
            { 1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 },
            { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0 }
        })
    });

    std::cout << cnn.toString() << std::endl << std::endl;
    
    prettyPrintTensor(input);
};