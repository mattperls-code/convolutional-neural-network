#ifndef CONVOLUTIONAL_NEURAL_NETWORK_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_HPP

#include <variant>
#include <memory>

#include "../lib/neural_network.hpp"

#include "tensor.hpp"

enum FeatureLayerType
{
    CONVOLUTION,
    POOL,
    ACTIVATION
};

class FeatureLayerState
{
    public:
        FeatureLayerType type;

        Tensor input;
        Tensor output;

        Tensor dLossWrtOutput;
        Tensor dLossWrtInput;

        virtual ~FeatureLayerState() = default;
};

class ConvolutionLayerState : public FeatureLayerState
{
    public:
        std::vector<Tensor> dLossWrtKernel;
};

class PoolLayerState : public FeatureLayerState {};

class ActivationLayerState : public FeatureLayerState
{
    public:
        Tensor weightedAndBiased;

        Tensor dLossWrtWeightedAndBiased;
        std::vector<float> dLossWrtBias;
        std::vector<float> dLossWrtWeights;
};

class FeatureLayerParameters
{
    public:
        FeatureLayerType type;

        virtual void randomizeParameters(const Dimensions& inputDimensions, std::mt19937& rng) = 0;

        virtual Dimensions getOutputputDimensions(const Dimensions& inputDimensions) const = 0;

        virtual ~FeatureLayerParameters() = default;
};

class ConvolutionLayerParameters : public FeatureLayerParameters
{
    private:
        int kernelCount;
        Shape kernelShape;

        std::vector<Tensor> kernels;

        int stride;
        int padding;

    public:
        ConvolutionLayerParameters(int kernelCount, const Shape& kernelShape, int stride, int padding);
        ConvolutionLayerParameters(const std::vector<Tensor>& kernels, int stride, int padding);
        
        void randomizeParameters(const Dimensions& inputDimensions, std::mt19937& rng) override;

        Dimensions getOutputputDimensions(const Dimensions& inputDimensions) const override;
};

enum PoolMode
{
    MIN,
    MAX,
    AVG
};

class PoolLayerParameters : public FeatureLayerParameters
{
    private:
        PoolMode mode;

        Shape window;

        int stride;
        int padding;

    public:
        PoolLayerParameters(PoolMode mode, const Shape& window, int stride, int padding);
        
        void randomizeParameters(const Dimensions& inputDimensions, std::mt19937& rng) override;

        Dimensions getOutputputDimensions(const Dimensions& inputDimensions) const override;
};

class ActivationLayerParameters : public FeatureLayerParameters
{
    private:
        UnaryActivationFunction unaryActivationFunction;
        bool applyWeightsAndBias;

        std::vector<float> weights;
        std::vector<float> bias;

    public:
        ActivationLayerParameters(UnaryActivationFunction unaryActivationFunction, bool applyWeightsAndBias);
        ActivationLayerParameters(UnaryActivationFunction unaryActivationFunction, bool applyWeightsAndBias, std::vector<float> weights, std::vector<float> bias);

        void randomizeParameters(const Dimensions& inputDimensions, std::mt19937& rng) override;

        Dimensions getOutputputDimensions(const Dimensions& inputDimensions) const override;
};

class FeatureLayerLossPartials
{
    public:
        FeatureLayerType type;

        virtual ~FeatureLayerLossPartials() = default;
};

class ConvolutionLayerLossPartials : public FeatureLayerLossPartials
{
    public:
        std::vector<Tensor> kernels;
};

class PoolLayerLossPartials : public FeatureLayerLossPartials {};

class ActivationLayerLossPartials : public FeatureLayerLossPartials
{
    public:
        std::vector<float> weights;
        std::vector<float> bias;
};

class ConvolutionalNetworkLossPartials
{
    public:
        Tensor inputLayerLossPartials;
        std::vector<std::unique_ptr<FeatureLayerLossPartials>> featureLayersLossPartials;

        ConvolutionalNetworkLossPartials(const Tensor& inputLayerLossPartials, const std::vector<std::unique_ptr<FeatureLayerLossPartials>>& featureLayersLossPartials): inputLayerLossPartials(inputLayerLossPartials), featureLayersLossPartials(featureLayersLossPartials) {};

        void add(const ConvolutionalNetworkLossPartials& other);

        void scalarMultiply(float scalar);
};

class TensorDataPoint
{
    public:
        Tensor input;
        Matrix expectedOutput;

        TensorDataPoint(const Tensor& input, const Matrix& expectedOutput): input(input), expectedOutput(expectedOutput) {};
};

class ConvolutionalNeuralNetwork
{
    private:
        Dimensions inputLayerDimensions;
        std::vector<std::unique_ptr<FeatureLayerState>> featureLayerStates;
        std::vector<std::unique_ptr<FeatureLayerParameters>> featureLayerParameters;
        NeuralNetwork neuralNetwork;

        void runFeatureLayerFeedforward(int featureLayerIndex, const Tensor& input);

        // should only be called after feedforward has run
        void calculateFeatureLayerLossPartials(int featureLayerIndex, const Tensor& dLossWrtOutput);
        ConvolutionalNetworkLossPartials calculateConvolutionalNetworkLossPartials(const Matrix& expectedOutput);

    public:
        ConvolutionalNeuralNetwork(const Dimensions& inputLayerDimensions, std::vector<std::unique_ptr<FeatureLayerParameters>> featureLayerParameters, std::vector<HiddenLayerParameters> hiddenLayerParameters, NormalizationFunction outputNormalizationFunction, LossFunction lossFunction);

        Dimensions getInputLayerDimensions();
        std::vector<std::unique_ptr<FeatureLayerState>> getFeatureLayerStates();
        std::vector<std::unique_ptr<FeatureLayerParameters>> getFeatureLayerParameters();
        NeuralNetwork getNeuralNetwork();
        Matrix getNormalizedOutput();

        void initializeRandomLayerParameters();
        void initializeRandomLayerParameters(float minInitialWeight, float maxInitialWeight, float minInitialBias, float maxInitialBias);

        Matrix calculateFeedForwardOutput(const Tensor& input);

        float calculateLoss(const Tensor& input, const Matrix& expectedOutput);

        ConvolutionalNetworkLossPartials train(TensorDataPoint trainingDataPoint, float learningRate);

        void batchTrain(std::vector<TensorDataPoint> trainingDataBatch, float learningRate);

        std::string toString();
};

#endif