#include "../src/convolutional_neural_network.hpp"

int main()
{
    ConvolutionalNeuralNetwork cnn(
        Dimensions(3, Shape(128, 128)),
        {
            std::make_unique<FeatureLayerParameters>(ConvolutionLayerParameters(16, Shape(8, 8), 4, 2)),
            std::make_unique<FeatureLayerParameters>(PoolLayerParameters(MAX, Shape(5, 5), 2, 0)),
            std::make_unique<FeatureLayerParameters>(ActivationLayerParameters(RELU, true))
        },
        {
            HiddenLayerParameters(32, TANH),
            HiddenLayerParameters(24, TANH),
            HiddenLayerParameters(8, LINEAR)
        },
        SOFTMAX,
        CATEGORICAL_CROSS_ENTROPY
    );

    cnn.initializeRandomLayerParameters();
};