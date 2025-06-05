#ifndef CONVOLUTIONAL_NEURAL_NETWORK_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_HPP

#include <memory>

#include "../lib/neural_network.hpp"

#include "image_operations.hpp"
#include "tensor.hpp"

enum FeatureLayerType
{
    CONVOLUTION,
    POOL,
    ACTIVATION
};

class FeatureLayerLossPartials
{
    public:
        FeatureLayerType type;

        FeatureLayerLossPartials(FeatureLayerType type): type(type) {};

        virtual void add(const FeatureLayerLossPartials* other) = 0;

        virtual void scalarMultiply(float scalar) = 0;

        virtual ~FeatureLayerLossPartials() = default;
};

class ConvolutionLayerLossPartials : public FeatureLayerLossPartials
{
    public:
        std::vector<Tensor> kernels;

        ConvolutionLayerLossPartials(std::vector<Tensor> kernels): FeatureLayerLossPartials(CONVOLUTION), kernels(kernels) {};

        void add(const FeatureLayerLossPartials* other) override;

        void scalarMultiply(float scalar) override;
};

class PoolLayerLossPartials : public FeatureLayerLossPartials {
    public:
        PoolLayerLossPartials(): FeatureLayerLossPartials(POOL) {};

        void add(const FeatureLayerLossPartials* other) override;

        void scalarMultiply(float scalar) override;
};

class ActivationLayerLossPartials : public FeatureLayerLossPartials
{
    public:
        std::vector<float> weights;
        std::vector<float> bias;

        ActivationLayerLossPartials(std::vector<float> weights, std::vector<float> bias): FeatureLayerLossPartials(ACTIVATION), weights(weights), bias(bias) {};

        void add(const FeatureLayerLossPartials* other) override;

        void scalarMultiply(float scalar) override;
};

class FeatureLayerState
{
    public:
        FeatureLayerType type;

        Tensor input;
        Tensor output;

        Tensor dLossWrtOutput;
        Tensor dLossWrtInput;
        
        FeatureLayerState(FeatureLayerType type): type(type) {};

        virtual FeatureLayerLossPartials* getLossPartials() const = 0;

        virtual std::string toString() const = 0;

        virtual ~FeatureLayerState() = default;
};

class ConvolutionLayerState : public FeatureLayerState
{
    public:
        Tensor paddedInput;

        std::vector<Tensor> dLossWrtKernels;

        ConvolutionLayerState(): FeatureLayerState(CONVOLUTION) {};

        FeatureLayerLossPartials* getLossPartials() const override;

        std::string toString() const override;
};

class PoolLayerState : public FeatureLayerState
{
    public:
        Tensor paddedInput;

        PoolLayerState(): FeatureLayerState(POOL) {};

        FeatureLayerLossPartials* getLossPartials() const override;

        std::string toString() const override;
};

class ActivationLayerState : public FeatureLayerState
{
    public:
        Tensor weightedAndBiased;

        Tensor dLossWrtWeightedAndBiased;
        std::vector<float> dLossWrtBias;
        std::vector<float> dLossWrtWeights;

        ActivationLayerState(): FeatureLayerState(POOL) {};

        FeatureLayerLossPartials* getLossPartials() const override;

        std::string toString() const override;
};

class FeatureLayerParameters
{
    public:
        FeatureLayerType type;

        virtual void randomizeParameters(const Dimensions& inputDimensions, std::mt19937& rng) = 0;

        virtual Dimensions getOutputDimensions(const Dimensions& inputDimensions) const = 0;

        virtual void runFeedforward(const Tensor& input, FeatureLayerState* state) const = 0;
        virtual void calculateLossPartials(const Tensor& dLossWrtOutput, FeatureLayerState* state) const = 0;

        virtual void applyLossPartials(const FeatureLayerLossPartials* lossPartials) = 0;

        virtual std::string toString() const = 0;

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

        Dimensions getOutputDimensions(const Dimensions& inputDimensions) const override;

        void runFeedforward(const Tensor& input, FeatureLayerState* state) const override;
        void calculateLossPartials(const Tensor& dLossWrtOutput, FeatureLayerState* state) const override;

        void applyLossPartials(const FeatureLayerLossPartials* lossPartials) override;

        std::string toString() const override;
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
        decltype(ImageOperations::minPool)* poolOperation;

        Shape window;

        int stride;
        int padding;

    public:
        PoolLayerParameters(PoolMode mode, const Shape& window, int stride, int padding);
        
        void randomizeParameters(const Dimensions& inputDimensions, std::mt19937& rng) override;

        Dimensions getOutputDimensions(const Dimensions& inputDimensions) const override;
        
        void runFeedforward(const Tensor& input, FeatureLayerState* state) const override;
        void calculateLossPartials(const Tensor& dLossWrtOutput, FeatureLayerState* state) const override;

        void applyLossPartials(const FeatureLayerLossPartials* lossPartials) override;

        std::string toString() const override;
};

class ActivationLayerParameters : public FeatureLayerParameters
{
    private:
        UnaryActivationFunction unaryActivationFunction;

        std::vector<float> weights;
        std::vector<float> bias;

    public:
        ActivationLayerParameters(UnaryActivationFunction unaryActivationFunction);
        ActivationLayerParameters(UnaryActivationFunction unaryActivationFunction, std::vector<float> weights, std::vector<float> bias);

        void randomizeParameters(const Dimensions& inputDimensions, std::mt19937& rng) override;

        Dimensions getOutputDimensions(const Dimensions& inputDimensions) const override;
        
        void runFeedforward(const Tensor& input, FeatureLayerState* state) const override;
        void calculateLossPartials(const Tensor& dLossWrtOutput, FeatureLayerState* state) const override;

        void applyLossPartials(const FeatureLayerLossPartials* lossPartials) override;

        std::string toString() const override;
};

class ConvolutionalNetworkLossPartials
{
    public:
        Tensor inputLayerLossPartials;
        std::vector<std::unique_ptr<FeatureLayerLossPartials>> featureLayersLossPartials;
        NetworkLossPartials neuralNetworkLossPartials;

        ConvolutionalNetworkLossPartials(const Tensor& inputLayerLossPartials, std::vector<FeatureLayerLossPartials*> featureLayersLossPartials, const NetworkLossPartials& neuralNetworkLossPartials);

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

    public:
        ConvolutionalNeuralNetwork(const Dimensions& inputLayerDimensions, std::vector<FeatureLayerParameters*> featureLayerParameters, std::vector<HiddenLayerParameters> hiddenLayerParameters, NormalizationFunction outputNormalizationFunction, LossFunction lossFunction);

        Dimensions getInputLayerDimensions();
        const std::vector<std::unique_ptr<FeatureLayerState>>& getFeatureLayerStates();
        const std::vector<std::unique_ptr<FeatureLayerParameters>>& getFeatureLayerParameters();
        NeuralNetwork getNeuralNetwork();
        Matrix getNormalizedOutput();

        void initializeRandomLayerParameters();
        void initializeRandomLayerParameters(float minInitialWeight, float maxInitialWeight, float minInitialBias, float maxInitialBias);

        Matrix calculateFeedForwardOutput(const Tensor& input);

        float calculateLoss(const Tensor& input, const Matrix& expectedOutput);

        ConvolutionalNetworkLossPartials calculateLossPartials(TensorDataPoint dataPoint);
        ConvolutionalNetworkLossPartials calculateBatchLossPartials(std::vector<TensorDataPoint> dataBatch);

        void applyLossPartials(ConvolutionalNetworkLossPartials lossPartials);

        void train(TensorDataPoint trainingDataPoint, float learningRate);
        void batchTrain(std::vector<TensorDataPoint> trainingDataBatch, float learningRate);

        std::string toString();
};

#endif