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
                    })
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
                    })
                })
            }, 2, 1),
            new ActivationLayerParameters(RELU, std::vector<float>({ 0.3, 0.6 }), std::vector<float>({ 0.1, -0.2 }))
        },
        {
            HiddenLayerParameters(
                TANH,
                Matrix({
                    { 0.1, -0.3, 0.8, 0.6, -0.5, -0.2, 1.1, 0.2, 1.1, -0.8, -0.6, -0.1, -0.5, -0.3, 0.8, 0.8, 0.1, -0.3, 0.8, 0.6, -0.5, -0.2, 1.1, 0.2, 1.1, -0.8, -0.6, -0.1, -0.5, -0.3, 0.8, 0.8 },
                    { 0.1, -0.3, 0.8, 0.6, -0.5, -0.2, 1.1, 0.2, 1.1, -0.8, -0.6, -0.1, -0.5, -0.3, 0.8, 0.8, 0.1, -0.3, 0.8, 0.6, -0.5, -0.2, 1.1, 0.2, 1.1, -0.8, -0.6, -0.1, -0.5, -0.3, 0.8, 0.8 },
                    { 0.1, -0.3, 0.8, 0.6, -0.5, -0.2, 1.1, 0.2, 1.1, -0.8, -0.6, -0.1, -0.5, -0.3, 0.8, 0.8, 0.1, -0.3, 0.8, 0.6, -0.5, -0.2, 1.1, 0.2, 1.1, -0.8, -0.6, -0.1, -0.5, -0.3, 0.8, 0.8 },
                    { 0.1, -0.3, 0.8, 0.6, -0.5, -0.2, 1.1, 0.2, 1.1, -0.8, -0.6, -0.1, -0.5, -0.3, 0.8, 0.8, 0.1, -0.3, 0.8, 0.6, -0.5, -0.2, 1.1, 0.2, 1.1, -0.8, -0.6, -0.1, -0.5, -0.3, 0.8, 0.8 },
                }),
                Matrix({{ 0.1 }, { -0.4 }, { 0.3 }, { 0.3 }})
            ),
            HiddenLayerParameters(
                LINEAR,
                Matrix({
                    { 0.7, 0.4, -0.3, -0.4 },
                    { 0.5, -0.5, -0.6, -0.9 },
                    { -0.4, 0.0, 1.2, -0.4 }
                }),
                Matrix({{ 0.4 }, { -0.3 }, { -0.7 }})
            )
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

    Matrix expectedOutput({{ 0.01 }, { 0.01 }, { 0.97 }});

    cnn.calculateFeedForwardOutput(input);

    SECTION("FEEDFORWARD") {
        Tensor expectedPoolOutput({
            Matrix({
                { 0.4, 0.6, 0.8, 1.0, 1.0, 1.0 },
                { 0.6, 0.8, 1.0, 1.0, 1.0, 1.0 },
                { 0.8, 1.0, 1.0, 1.0, 0.4, 0.6 },
                { 0.8, 1.0, 1.0, 0.8, 1.0, 1.0 },
                { 1.0, 0.4, 0.6, 0.8, 1.0, 1.0 },
                { 1.0, 0.8, 1.0, 1.0, 0.8, 1.0 }
            }),
            Matrix({
                { 1.0, 1.0, 0.6, 0.4, 0.6, 1.0 },
                { 1.0, 1.0, 0.2, 1.0, 1.0, 0.8 },
                { 0.6, 0.4, 1.0, 1.0, 1.0, 0.8 },
                { 0.8, 0.6, 1.0, 1.0, 1.0, 1.0 },
                { 0.8, 1.0, 1.0, 0.2, 1.0, 1.0 },
                { 1.0, 1.0, 1.0, 0.4, 0.4, 1.0 }
            }),
            Matrix({
                { 0.4, 1.0, 1.0, 1.0, 1.0, 0.4 },
                { 1.0, 1.0, 0.8, 1.0, 1.0, 0.2 },
                { 1.0, 1.0, 0.8, 0.6, 0.6, 0.8 },
                { 1.0, 1.0, 0.6, 0.8, 1.0, 1.0 },
                { 1.0, 0.4, 0.6, 0.8, 1.0, 1.0 },
                { 1.0, 0.4, 0.6, 0.8, 1.0, 1.0 }
            })
        });

        auto observedPoolOutput = cnn.getFeatureLayerStates()[0]->output;

        REQUIRE(tensorsAreApproxEqual(observedPoolOutput, expectedPoolOutput, 0.0));

        Tensor expectedConvolutionOutput({
            Matrix::add(Matrix::add(
                ImageOperations::convolution(
                    ImageOperations::pad(expectedPoolOutput.getMatrix(0), 1),
                    Matrix({
                        { 0.0, 1.0 },
                        { 1.0, 1.0}
                    }),
                    2
                ),
                ImageOperations::convolution(
                    ImageOperations::pad(expectedPoolOutput.getMatrix(1), 1),
                    Matrix({
                        { -0.5, 0.0 },
                        { 0.5, 1.0 }
                    }),
                    2
                )),
                ImageOperations::convolution(
                    ImageOperations::pad(expectedPoolOutput.getMatrix(2), 1),
                    Matrix({
                        { -1.0, -1.0 },
                        { 0.0, 0.0}
                    }),
                    2
                )
            ),
            Matrix::add(Matrix::add(
                ImageOperations::convolution(
                    ImageOperations::pad(expectedPoolOutput.getMatrix(0), 1),
                    Matrix({
                        { 0.0, 0.5 },
                        { -0.5, 0.0 }
                    }),
                    2
                ),
                ImageOperations::convolution(
                    ImageOperations::pad(expectedPoolOutput.getMatrix(1), 1),
                    Matrix({
                        { 1.0, -1.0 },
                        { -0.5, 0.5}
                    }),
                    2
                )),
                ImageOperations::convolution(
                    ImageOperations::pad(expectedPoolOutput.getMatrix(2), 1),
                    Matrix({
                        { 0.0, 0.5 },
                        { -0.5, 0.5 }
                    }),
                    2
                )
            )
        });

        auto observedConvolutionOutput = cnn.getFeatureLayerStates()[1]->output;

        REQUIRE(tensorsAreApproxEqual(observedConvolutionOutput, expectedConvolutionOutput, 0.0));

        Tensor expectedActivationOutput({
            evaluateUnaryActivationFunction(RELU, Matrix::add(
                Matrix::scalarProduct(observedConvolutionOutput.getMatrix(0), 0.3),
                Matrix(observedConvolutionOutput.getDimensions().shape, 0.1)
            )),
            evaluateUnaryActivationFunction(RELU, Matrix::add(
                Matrix::scalarProduct(observedConvolutionOutput.getMatrix(1), 0.6),
                Matrix(observedConvolutionOutput.getDimensions().shape, -0.2)
            ))
        });

        auto observedActivationOutput = cnn.getFeatureLayerStates()[2]->output;

        REQUIRE(tensorsAreApproxEqual(observedActivationOutput, expectedActivationOutput, 1e-4));
    }

    auto neuralNetworkLossPartials = cnn.getNeuralNetwork().calculateLossPartials(expectedOutput);

    auto convolutionalNetworkLossPartials = cnn.calculateLossPartials(neuralNetworkLossPartials);

    SECTION("BACK PROPAGATION") {
        SECTION("ACTIVATION LAYER") {
            auto state = dynamic_cast<ActivationLayerState*>(cnn.getFeatureLayerStates()[2].get());

            Tensor expectedDLossWrtWeightedAndBiased({
                Matrix({
                    {  0.00850, -0.02551,  0.06802,  0.05101 },
                    { -0.04251, -0.01700,  0.09352,  0.01700 },
                    {  0.09352, -0.06802, -0.05101, -0.00850 },
                    { -0.04251,  0.00000,  0.00000,  0.00000 }
                }),
                Matrix({
                    {  0.00850,  0.00000,  0.00000,  0.00000 },
                    { -0.04251, -0.01700,  0.09352,  0.00000 },
                    {  0.09352,  0.00000, -0.05101,  0.00000 },
                    {  0.00000, -0.02551,  0.06802,  0.06802 }
                })
            });

            auto observedDLossWrtWeightedAndBiased = state->dLossWrtWeightedAndBiased;

            REQUIRE(tensorsAreApproxEqual(observedDLossWrtWeightedAndBiased, expectedDLossWrtWeightedAndBiased, 1e-4));

            std::vector<float> expectedDLossWrtWeights = { 0.237198, 0.149643 };

            auto observedDLossWrtWeights = state->dLossWrtWeights;

            REQUIRE(matricesAreApproxEqual(Matrix({ observedDLossWrtWeights }), Matrix({ expectedDLossWrtWeights }), 1e-4));

            std::vector<float> expectedDLossWrtBias = { 0.07651, 0.19555 };

            auto observedDLossWrtBias = state->dLossWrtBias;

            REQUIRE(matricesAreApproxEqual(Matrix({ observedDLossWrtBias }), Matrix({ expectedDLossWrtBias }), 1e-4));

            Tensor expectedDLossWrtInput({
                Matrix({
                    {  0.002550, -0.007653,  0.020406,  0.015303 },
                    { -0.012753, -0.005100,  0.028056,  0.005100 },
                    {  0.028056, -0.020406, -0.015303, -0.002550 },
                    { -0.012753,  0.000000,  0.000000,  0.000000 }
                }),
                Matrix({
                    {  0.005100,  0.000000,  0.000000,  0.000000 },
                    { -0.025506, -0.010200,  0.056112,  0.000000 },
                    {  0.056112,  0.000000, -0.030606,  0.000000 },
                    {  0.000000, -0.015306,  0.040812,  0.040812 }
                })
            });

            auto observedDLossWrtInput = state->dLossWrtInput;

            REQUIRE(tensorsAreApproxEqual(observedDLossWrtInput, expectedDLossWrtInput, 1e-4));
        }

        SECTION("CONVOLUTION LAYER") {
            auto state = dynamic_cast<ConvolutionLayerState*>(cnn.getFeatureLayerStates()[1].get());
            
            std::vector<Tensor> expectedDLossWrtkernels = {
                Tensor({
                    Matrix({{ -0.00612, -0.010702 }, { 0.034186, 0.011748 }}),
                    Matrix({{ -0.003056, -0.011722 }, { 0.019894, 0.012254 }}),
                    Matrix({{ -0.01122, -0.001006 }, { 0.011742, 0.0143 }})
                }),
                Tensor({
                    Matrix({{ 0.092842, 0.06223 }, { 0.021422, 0.019376 }}),
                    Matrix({{ 0.057134, 0.043862 }, { 0.045908, 0.049982 }}),
                    Matrix({{ 0.08876, 0.07957 }, { -0.001022, 0.027536 }})
                })
            };

            auto observedDLossWrtKernels = state->dLossWrtKernels;
            
            REQUIRE(expectedDLossWrtkernels.size() == observedDLossWrtKernels.size());

            for (int i = 0;i<expectedDLossWrtkernels.size();i++) REQUIRE(tensorsAreApproxEqual(observedDLossWrtKernels[i], expectedDLossWrtkernels[i], 1e-4));

            Tensor expectedDLossWrtInput({
                Matrix({
                    { 0.00255, -0.00765, -0.00765, 0.02041, 0.02041, 0.0153 },
                    { -0.025505, 0.0, -0.0102, 0.0, 0.056115, 0.0 },
                    { -0.01275, 0.0, -0.0051, 0.000005, 0.02806, 0.0051 },
                    { 0.056115, 0.0, -0.02041, 0.0, -0.030605, 0.0 },
                    { 0.02806, -0.02041, -0.02041, 0.000005, -0.0153, -0.00255 },
                    { -0.01275, 0.0, -0.00765, 0.0, 0.020405, 0.0 }
                }),
                Matrix({
                    { 0.0051, -0.003825, -0.00765, 0.010205, 0.02041, 0.00765 },
                    { 0.02551, -0.00765, 0.0102, 0.04208, -0.05611, -0.00255 },
                    { -0.025505, 0.00255, -0.0102, -0.014025, 0.056115, 0.00255 },
                    { -0.05611, 0.010205, 0.0, -0.02296, 0.03061, 0.001275 },
                    { 0.056115, -0.010205, -0.02041, 0.007655, -0.030605, -0.001275 },
                    { 0.0, -0.0153, 0.0153, 0.04081, -0.04081, 0.04081 }
                }),
                Matrix({
                    { 0.00255, 0.0, 0.0, 0.0, 0.0, 0.0 },
                    { -0.000005, 0.0051, 0.0, -0.02806, -0.000005, -0.0051 },
                    { -0.012755, 0.0051, -0.0051, -0.028055, 0.028055, 0.0 },
                    { -0.000005, 0.02041, 0.02041, 0.0153, -0.000005, 0.00255 },
                    { 0.028055, 0.0, 0.0, 0.015305, -0.015305, 0.0 },
                    { 0.01275, 0.0, -0.00765, 0.0, 0.020405, 0.0 }
                })
            });

            auto observedDLossWrtInput = state->dLossWrtInput;

            REQUIRE(tensorsAreApproxEqual(observedDLossWrtInput, expectedDLossWrtInput, 1e-4));
        }

        SECTION("POOL LAYER") {
            Tensor poolInput({
                Matrix({
                    { 0.5, 1.0, 1.0, 0.0, -1.0 },
                    { 1.0, 1.0, 0.0, 0.0, 0.0 },
                    { 0.5, 0.0, -0.5, -0.5, 0.5 },
                    { 0.0, 0.0, 1.0, 1.0, 0.5 },
                    { 1.0, 0.5, -0.5, 0.5, 0.0 }
                }),
                Matrix({
                    { 0.0, 0.5, -1.0, -1.0, -0.5 },
                    { -1.0, 1.0, 1.0, 0.5, 0.0 },
                    { 0.0, 0.0, 1.0, 0.5, 0.5 },
                    { -0.5, -0.5, 0.5, 0.0, 1.0 },
                    { 1.0, -0.5, 0.0, 0.0, -0.5 }
                })
            });

            Tensor poolDLossWrtOutput({
                Matrix({
                    { 1.0, 0.5, 1.0 },
                    { -0.5, 0.5, 0.0 },
                    { 0.5, 0.5, 1.0 }
                }),
                Matrix({
                    { 0.5, 0.5, 1.0 },
                    { 1.0, 1.0, -1.0 },
                    { -0.5, -1.0, -1.0 }
                })
            });

            SECTION("MIN | MAX") {
                PoolLayerParameters poolLayerParameters(MAX, Shape(2, 2), 2, 1);
                
                auto poolLayerState = std::make_unique<PoolLayerState>();

                poolLayerParameters.runFeedforward(poolInput, poolLayerState.get());

                poolLayerParameters.calculateLossPartials(poolDLossWrtOutput, poolLayerState.get());

                Tensor expectedDLossWrtInput({
                    Matrix({
                        { 1.0, 0.5, 0.0, 0.0, 0.0 },
                        { -0.5, 0.5, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.5, 1.0, 0.0 },
                        { 0.5, 0.0, 0.0, 0.0, 0.0 }
                    }),
                    Matrix({
                        { 0.0, 0.5, 0.0, 0.0, 0.0 },
                        { 0.0, 1.0, 0.0, -1.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, -1.0, 0.0, -1.0 },
                        { -0.5, 0.0, 0.0, 0.0, 0.0 }
                    })
                });

                auto observedDLossWrtInput = poolLayerState->dLossWrtInput;

                REQUIRE(tensorsAreApproxEqual(observedDLossWrtInput, expectedDLossWrtInput, 1e-6));
            }

            SECTION("AVG") {
                PoolLayerParameters poolLayerParameters(AVG, Shape(2, 2), 2, 1);
                
                auto poolLayerState = std::make_unique<PoolLayerState>();

                poolLayerParameters.runFeedforward(poolInput, poolLayerState.get());

                poolLayerParameters.calculateLossPartials(poolDLossWrtOutput, poolLayerState.get());
                
                Tensor expectedDLossWrtInput = Tensor::scalarProduct(Tensor({
                    Matrix({
                        { 1.0, 0.5, 0.5, 1.0, 1.0 },
                        { -0.5, 0.5, 0.5, 0.0, 0.0 },
                        { -0.5, 0.5, 0.5, 0.0, 0.0 },
                        { 0.5, 0.5, 0.5, 1.0, 1.0 },
                        { 0.5, 0.5, 0.5, 1.0, 1.0 }
                    }),
                    Matrix({
                        { 0.5, 0.5, 0.5, 1.0, 1.0 },
                        { 1.0, 1.0, 1.0, -1.0, -1.0 },
                        { 1.0, 1.0, 1.0, -1.0, -1.0 },
                        { -0.5, -1.0, -1.0, -1.0, -1.0 },
                        { -0.5, -1.0, -1.0, -1.0, -1.0 }
                    })
                }), 0.25);

                auto observedDLossWrtInput = poolLayerState->dLossWrtInput;

                REQUIRE(tensorsAreApproxEqual(observedDLossWrtInput, expectedDLossWrtInput, 1e-6));
            }
        }
    }
};