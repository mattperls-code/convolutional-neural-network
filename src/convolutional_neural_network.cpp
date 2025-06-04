#include "convolutional_neural_network.hpp"

ConvolutionLayerParameters::ConvolutionLayerParameters(int kernelCount, const Shape& kernelShape, int stride, int padding)
{
    this->type = CONVOLUTION;

    this->kernelCount = kernelCount;
    this->kernelShape = kernelShape;

    this->stride = stride;
    this->padding = padding;
};

ConvolutionLayerParameters::ConvolutionLayerParameters(const std::vector<Tensor>& kernels, int stride, int padding)
{
    this->type = CONVOLUTION;

    if (kernels.empty()) throw std::runtime_error("ConvolutionLayerParameters: kernels is empty");

    // in case of random parameters generated after explicit kernels initialization
    this->kernelCount = kernels.size();
    this->kernelShape = kernels[0].getDimensions().shape;

    this->kernels = kernels;

    this->stride = stride;
    this->padding = padding;
};

void ConvolutionLayerParameters::randomizeParameters(const Dimensions& inputDimensions, std::mt19937& rng)
{
    std::uniform_real_distribution<float> kernelPixelValueDistribution(-1.0, 1.0);

    this->kernels.resize(this->kernelCount);

    for (int i = 0;i<this->kernelCount;i++) this->kernels[i] = Tensor(Dimensions(inputDimensions.depth, this->kernelShape), rng, kernelPixelValueDistribution);
};

Dimensions ConvolutionLayerParameters::getOutputputDimensions(const Dimensions& inputDimensions) const
{
    return Dimensions(
        this->kernelCount,
        Shape(
            (inputDimensions.shape.rows + 2 * this->padding - this->kernelShape.rows) / this->stride + 1,
            (inputDimensions.shape.cols + 2 * this->padding - this->kernelShape.cols) / this->stride + 1
        )
    );
};

PoolLayerParameters::PoolLayerParameters(PoolMode mode, const Shape& window, int stride, int padding)
{
    this->type = POOL;

    this->mode = mode;

    this->window = window;

    this->stride = stride;
    this->padding = padding;
};

void PoolLayerParameters::randomizeParameters(const Dimensions& inputDimensions, std::mt19937& rng) {};

Dimensions PoolLayerParameters::getOutputputDimensions(const Dimensions& inputDimensions) const
{
    return Dimensions(
        inputDimensions.depth,
        Shape(
            (inputDimensions.shape.rows + 2 * this->padding - this->window.rows) / this->stride + 1,
            (inputDimensions.shape.cols + 2 * this->padding - this->window.cols) / this->stride + 1
        )
    );
};

ActivationLayerParameters::ActivationLayerParameters(UnaryActivationFunction unaryActivationFunction, bool applyWeightsAndBias)
{
    this->type = ACTIVATION;

    this->unaryActivationFunction = unaryActivationFunction;
    this->applyWeightsAndBias = applyWeightsAndBias;
};

ActivationLayerParameters::ActivationLayerParameters(UnaryActivationFunction unaryActivationFunction, bool applyWeightsAndBias, std::vector<float> weights, std::vector<float> bias)
{
    this->type = ACTIVATION;

    this->unaryActivationFunction = unaryActivationFunction;
    this->applyWeightsAndBias = applyWeightsAndBias;

    this->weights = weights;
    this->bias = bias;
};

void ActivationLayerParameters::randomizeParameters(const Dimensions& inputDimensions, std::mt19937& rng)
{
    // TODO: eventually build in a proper way to specify this distribution, a bit of a pain with all the subclassing though

    std::uniform_real_distribution<float> initialWeightDistribution(-1.0, 1.0);
    std::uniform_real_distribution<float> initialBiasDistribution(-0.2, 0.2);

    this->weights.resize(inputDimensions.depth);
    this->bias.resize(inputDimensions.depth);

    for (int i = 0;i<inputDimensions.depth;i++) {
        this->weights[i] = initialWeightDistribution(rng);
        this->bias[i] = initialBiasDistribution(rng);
    }
};

Dimensions ActivationLayerParameters::getOutputputDimensions(const Dimensions& inputDimensions) const
{
    return inputDimensions;
};