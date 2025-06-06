#include "convolutional_neural_network.hpp"

// feature layer loss partials

void ConvolutionLayerLossPartials::add(const FeatureLayerLossPartials* other)
{
    if (other->type != CONVOLUTION) throw std::runtime_error("ConvolutionLayerLossPartials add: other is wrong type");

    auto otherConvolutionLayerParameters = *static_cast<const ConvolutionLayerLossPartials*>(other);

    if (this->kernels.size() != otherConvolutionLayerParameters.kernels.size()) throw std::runtime_error("ConvolutionLayerLossPartials add: other has different number of kernels");

    for (int i = 0;i<this->kernels.size();i++) this->kernels[i] = Tensor::add(this->kernels[i], otherConvolutionLayerParameters.kernels[i]);
};

void ConvolutionLayerLossPartials::scalarMultiply(float scalar)
{
    for (auto& kernel : this->kernels) kernel = Tensor::scalarProduct(kernel, scalar);
};

void PoolLayerLossPartials::add(const FeatureLayerLossPartials* other)
{
    if (other->type != POOL) throw std::runtime_error("PoolLayerLossPartials add: other is wrong type");
};

void PoolLayerLossPartials::scalarMultiply(float) {};

void ActivationLayerLossPartials::add(const FeatureLayerLossPartials* other)
{
    if (other->type != ACTIVATION) throw std::runtime_error("ActivationLayerLossPartials add: other kernels is wrong size");

    auto otherActivationLayerLossPartials = *static_cast<const ActivationLayerLossPartials*>(other);

    if (this->weights.size() != otherActivationLayerLossPartials.weights.size()) throw std::runtime_error("ActivationLayerLossPartials add: other weights is wrong size");
    if (this->bias.size() != otherActivationLayerLossPartials.bias.size()) throw std::runtime_error("ActivationLayerLossPartials add: other bias is wrong size");

    for (int i = 0;i<this->weights.size();i++) this->weights[i] += otherActivationLayerLossPartials.weights[i];
    for (int i = 0;i<this->bias.size();i++) this->bias[i] += otherActivationLayerLossPartials.bias[i];
};

void ActivationLayerLossPartials::scalarMultiply(float scalar)
{
    for (int i = 0;i<this->weights.size();i++) this->weights[i] *= scalar;
    for (int i = 0;i<this->bias.size();i++) this->bias[i] *= scalar;
};

// feature layer state

FeatureLayerLossPartials* ConvolutionLayerState::getLossPartials() const
{
    return new ConvolutionLayerLossPartials(this->dLossWrtKernels);
};

std::string ConvolutionLayerState::toString() const
{
    std::string output;

    output += "\t\tinput: " + this->input.toString() + "\n";
    output += "\t\tpaddedInput: " + this->paddedInput.toString() + "\n";
    output += "\t\toutput: " + this->output.toString() + "\n";

    output += "\n";

    output += "\t\tdLossWrtOutput: " + this->dLossWrtOutput.toString() + "\n";

    output += "\t\tdLossWrtKernels:\n";
    for (auto dLossWrtKernel : this->dLossWrtKernels) output += "\t\t" + dLossWrtKernel.toString() + "\n";

    output += "\t\tdLossWrtInput: " + this->dLossWrtInput.toString() + "\n";

    return output;
};

FeatureLayerLossPartials* PoolLayerState::getLossPartials() const
{
    return new PoolLayerLossPartials();
};

std::string PoolLayerState::toString() const
{
    std::string output;

    output += "\t\tinput: " + this->input.toString() + "\n";
    output += "\t\toutput: " + this->output.toString() + "\n";

    output += "\n";

    output += "\t\tdLossWrtOutput: " + this->dLossWrtOutput.toString() + "\n";
    output += "\t\tdLossWrtInput: " + this->dLossWrtInput.toString() + "\n";

    return output;
};

FeatureLayerLossPartials* ActivationLayerState::getLossPartials() const
{
    return new ActivationLayerLossPartials(this->dLossWrtWeights, this->dLossWrtBias);
};

std::string ActivationLayerState::toString() const
{
    std::string output;

    output += "\t\tinput: " + this->input.toString() + "\n";
    output += "\t\tweightedAndBiased: " + this->weightedAndBiased.toString() + "\n";
    output += "\t\toutput: " + this->output.toString() + "\n";

    output += "\n";

    output += "\t\tdLossWrtOutput: " + this->dLossWrtOutput.toString() + "\n";
    output += "\t\tdLossWrtWeightedAndBiased: " + this->dLossWrtWeightedAndBiased.toString() + "\n";
    output += "\t\tdLossWrtWeights: " + Matrix({ this->dLossWrtWeights }).toString() + "\n";
    output += "\t\tdLossWrtBias: " + Matrix({ this->dLossWrtBias }).toString() + "\n";
    output += "\t\tdLossWrtInput: " + this->dLossWrtInput.toString() + "\n";

    return output;
};

// feature layer parameters

ConvolutionLayerParameters::ConvolutionLayerParameters(int kernelCount, const Shape& kernelShape, int stride, int padding)
{
    if (kernelCount < 1) throw std::runtime_error("ConvolutionLayerParameters constructor: layer must have at least one kernel");

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
    if (inputDimensions.depth == 0) throw std::runtime_error("ConvolutionLayerParameters randomizeParameters: inputDimensions cannot be empty");

    std::uniform_real_distribution<float> kernelPixelValueDistribution(-1.0, 1.0);

    this->kernels.resize(this->kernelCount);

    for (int i = 0;i<this->kernelCount;i++) this->kernels[i] = Tensor(Dimensions(inputDimensions.depth, this->kernelShape), rng, kernelPixelValueDistribution);
};

Dimensions ConvolutionLayerParameters::getOutputDimensions(const Dimensions& inputDimensions) const
{
    return Dimensions(
        this->kernelCount,
        Shape(
            (inputDimensions.shape.rows + 2 * this->padding - this->kernelShape.rows) / this->stride + 1,
            (inputDimensions.shape.cols + 2 * this->padding - this->kernelShape.cols) / this->stride + 1
        )
    );
};

void ConvolutionLayerParameters::runFeedforward(const Tensor& input, FeatureLayerState* state) const
{
    if (state->type != CONVOLUTION) throw std::runtime_error("ConvolutionLayerParameters runFeedforward: state is wrong type");

    auto convolutionLayerState = static_cast<ConvolutionLayerState*>(state);

    auto inputDepth = input.getDimensions().depth;

    if (inputDepth != this->kernels[0].getDimensions().depth) throw std::runtime_error("ConvolutionLayerParameters runFeedforward: kernel depth misalignment");

    convolutionLayerState->input = input;

    std::vector<Matrix> paddedInputMatrices(inputDepth);

    for (int i = 0;i<inputDepth;i++) paddedInputMatrices[i] = ImageOperations::pad(input.getMatrix(i), this->padding);

    convolutionLayerState->paddedInput = Tensor(paddedInputMatrices);

    std::vector<Matrix> convolutionOutput(this->kernelCount);

    for (int i = 0;i<this->kernelCount;i++) {
        Matrix kernelOutput = ImageOperations::convolution(paddedInputMatrices[0], this->kernels[i].getMatrix(0), this->stride);

        for (int j = 1;j<paddedInputMatrices.size();j++) kernelOutput = Matrix::add(kernelOutput, ImageOperations::convolution(paddedInputMatrices[j], this->kernels[i].getMatrix(j), this->stride));

        convolutionOutput[i] = kernelOutput;
    }

    convolutionLayerState->output = Tensor(convolutionOutput);
};

void ConvolutionLayerParameters::calculateLossPartials(const Tensor& dLossWrtOutput, FeatureLayerState* state) const
{
    if (state->type != CONVOLUTION) throw std::runtime_error("ConvolutionLayerParameters calculateLossPartials: state is wrong type");

    auto convolutionLayerState = static_cast<ConvolutionLayerState*>(state);

    auto inputDepth = convolutionLayerState->input.getDimensions().depth;
    auto outputDepth = convolutionLayerState->output.getDimensions().depth;

    convolutionLayerState->dLossWrtOutput = dLossWrtOutput;
    convolutionLayerState->dLossWrtKernels = std::vector<Tensor>(this->kernelCount, this->kernels[0].getDimensions());
    convolutionLayerState->dLossWrtInput = Tensor(convolutionLayerState->input.getDimensions());

    for (int inputChannel = 0;inputChannel<inputDepth;inputChannel++) {
        auto inputMatrix = convolutionLayerState->paddedInput.getMatrix(inputChannel);

        for (int outputChannel = 0;outputChannel<outputDepth;outputChannel++) {
            auto upsampledDLossWrtOutputChannel = Matrix::upsample(convolutionLayerState->dLossWrtOutput.getMatrix(outputChannel), this->stride - 1);
            
            convolutionLayerState->dLossWrtKernels[outputChannel].dangerouslyGetData()[inputChannel] = ImageOperations::convolution(inputMatrix, upsampledDLossWrtOutputChannel, 1);

            auto kernel = this->kernels[outputChannel].getMatrix(inputChannel);

            convolutionLayerState->dLossWrtInput.dangerouslyGetData()[inputChannel] = Matrix::add(
                convolutionLayerState->dLossWrtInput.getMatrix(inputChannel),
                ImageOperations::crop(
                    ImageOperations::convolution(
                        ImageOperations::pad(
                            upsampledDLossWrtOutputChannel,
                            0.0,
                            kernel.colCount() - 1,
                            kernel.colCount() - 1,
                            kernel.rowCount() - 1,
                            kernel.rowCount() - 1
                        ),
                        Matrix::flipped(kernel),
                        1
                    ),
                    this->padding,
                    this->padding,
                    convolutionLayerState->input.getDimensions().shape,
                    0.0
                )
            );
        }
    }
};

void ConvolutionLayerParameters::applyLossPartials(const FeatureLayerLossPartials* lossPartials)
{
    if (lossPartials->type != CONVOLUTION) throw std::runtime_error("ConvolutionLayerParameters applyLossPartials: lossPartials is wrong type");

    auto convolutionLayerLossPartials = *static_cast<const ConvolutionLayerLossPartials*>(lossPartials);

    if (this->kernelCount != convolutionLayerLossPartials.kernels.size()) throw std::runtime_error("ConvolutionLayerParameters applyLossPartials: lossPartials has different number of kernels");

    for (int i = 0;i<this->kernelCount;i++) this->kernels[i] = Tensor::add(this->kernels[i], convolutionLayerLossPartials.kernels[i]);
};

std::string ConvolutionLayerParameters::toString() const
{
    std::string output;

    output += "\t\tstride: " + std::to_string(this->stride) + "\n";
    output += "\t\tpadding: " + std::to_string(this->padding) + "\n";
    output += "\t\tkernelCount: " + std::to_string(this->kernelCount) + "\n";
    output += "\t\tkernelShape: " + this->kernelShape.toString() + "\n";

    output += "\t\tkernels:\n";
    for (auto kernel : this->kernels) output += "\t\t\t\t" + kernel.toString() + "\n";

    return output;
};

PoolLayerParameters::PoolLayerParameters(PoolMode mode, const Shape& window, int stride, int padding)
{
    this->type = POOL;

    this->mode = mode;

    switch (this->mode) {
        case MIN:
            this->poolOperation = ImageOperations::minPool;
            break;
        case MAX:
            this->poolOperation = ImageOperations::maxPool;
            break;
        case AVG:
            this->poolOperation = ImageOperations::avgPool;
            break;
        default:
            throw std::runtime_error("PoolLayerParameters constructor: unhandled mode");
    }

    this->window = window;

    this->stride = stride;
    this->padding = padding;
};

void PoolLayerParameters::randomizeParameters(const Dimensions&, std::mt19937&) {};

Dimensions PoolLayerParameters::getOutputDimensions(const Dimensions& inputDimensions) const
{
    return Dimensions(
        inputDimensions.depth,
        Shape(
            (inputDimensions.shape.rows + 2 * this->padding - this->window.rows) / this->stride + 1,
            (inputDimensions.shape.cols + 2 * this->padding - this->window.cols) / this->stride + 1
        )
    );
};

void PoolLayerParameters::runFeedforward(const Tensor& input, FeatureLayerState* state) const
{
    if (state->type != POOL) throw std::runtime_error("PoolLayerParameters runFeedforward: state is wrong type");

    auto poolLayerState = static_cast<PoolLayerState*>(state);

    poolLayerState->input = input;

    auto inputDepth = input.getDimensions().depth;
    
    std::vector<Matrix> paddedInputMatrices(inputDepth);

    for (int i = 0;i<inputDepth;i++) paddedInputMatrices[i] = ImageOperations::pad(input.getMatrix(i), this->padding);

    poolLayerState->paddedInput = Tensor(paddedInputMatrices);

    std::vector<Matrix> poolOutput(inputDepth);

    for (int i = 0;i<inputDepth;i++) poolOutput[i] = this->poolOperation(paddedInputMatrices[i], this->window, this->stride);

    poolLayerState->output = Tensor(poolOutput);
};

void PoolLayerParameters::calculateLossPartials(const Tensor& dLossWrtOutput, FeatureLayerState* state) const
{
    if (state->type != POOL) throw std::runtime_error("PoolLayerParameters calculateLossPartials: state is wrong type");

    auto poolLayerState = static_cast<PoolLayerState*>(state);

    poolLayerState->dLossWrtOutput = dLossWrtOutput;
    poolLayerState->dLossWrtInput = Tensor(poolLayerState->input.getDimensions());

    for (int channel = 0;channel<poolLayerState->input.getDimensions().depth;channel++) {
        auto paddedInputMatrix = poolLayerState->paddedInput.getMatrix(channel);
        auto dLossWrtOutputChannel = poolLayerState->dLossWrtOutput.getMatrix(channel);

        auto& dLossWrtInputChannel = poolLayerState->dLossWrtInput.dangerouslyGetData()[channel];

        for (int outputR = 0;outputR<dLossWrtOutputChannel.rowCount();outputR++) {
            for (int outputC = 0;outputC<dLossWrtOutputChannel.colCount();outputC++) {
                auto dLossWrtOutputPixel = dLossWrtOutputChannel.get(outputR, outputC);

                auto windowStartInputRow = outputR * this->stride;
                auto windowStartInputCol = outputC * this->stride;

                if (this->mode == MIN) {
                    auto minWindowRow = windowStartInputRow;
                    auto minWindowCol = windowStartInputCol;
                    auto minWindowValue = paddedInputMatrix.get(minWindowRow, minWindowCol);

                    for (int windowR = 0;windowR<this->window.rows;windowR++) {
                        for (int windowC = 0;windowC<this->window.cols;windowC++) {
                            auto inputRow = windowStartInputRow + windowR;
                            auto inputCol = windowStartInputCol + windowC;
                            auto inputValue = paddedInputMatrix.get(inputRow, inputCol);

                            if (inputValue < minWindowValue) {
                                minWindowRow = inputRow;
                                minWindowCol = inputCol;
                                minWindowValue = inputValue;
                            }
                        }
                    }

                    int unpaddedRow = minWindowRow - this->padding;
                    int unpaddedCol = minWindowCol - this->padding;

                    if (unpaddedRow >= 0 && unpaddedCol >= 0 && unpaddedRow < dLossWrtInputChannel.rowCount() && unpaddedCol < dLossWrtInputChannel.colCount()) {
                        dLossWrtInputChannel.set(unpaddedRow, unpaddedCol, dLossWrtInputChannel.get(unpaddedRow, unpaddedCol) + dLossWrtOutputPixel);
                    }
                }
                else if (this->mode == MAX) {
                    auto maxWindowRow = windowStartInputRow;
                    auto maxWindowCol = windowStartInputCol;
                    auto maxWindowValue = paddedInputMatrix.get(maxWindowRow, maxWindowCol);

                    for (int windowR = 0;windowR<this->window.rows;windowR++) {
                        for (int windowC = 0;windowC<this->window.cols;windowC++) {
                            auto inputRow = windowStartInputRow + windowR;
                            auto inputCol = windowStartInputCol + windowC;
                            auto inputValue = paddedInputMatrix.get(inputRow, inputCol);

                            if (inputValue > maxWindowValue) {
                                maxWindowRow = inputRow;
                                maxWindowCol = inputCol;
                                maxWindowValue = inputValue;
                            }
                        }
                    }

                    int unpaddedRow = maxWindowRow - this->padding;
                    int unpaddedCol = maxWindowCol - this->padding;
                    
                    if (unpaddedRow >= 0 && unpaddedCol >= 0 && unpaddedRow < dLossWrtInputChannel.rowCount() && unpaddedCol < dLossWrtInputChannel.colCount()) {
                        dLossWrtInputChannel.set(unpaddedRow, unpaddedCol, dLossWrtInputChannel.get(unpaddedRow, unpaddedCol) + dLossWrtOutputPixel);
                    }
                }
                else if (this->mode == AVG) {
                    auto dLossWrtWindowPixel = dLossWrtOutputPixel / this->window.area();

                    for (int windowR = 0;windowR<this->window.rows;windowR++) {
                        for (int windowC = 0;windowC<this->window.cols;windowC++) {
                            auto inputRow = windowStartInputRow + windowR - this->padding;
                            auto inputCol = windowStartInputCol + windowC - this->padding;

                            if (inputRow >= 0 && inputCol >= 0 && inputRow < dLossWrtInputChannel.rowCount() && inputCol < dLossWrtInputChannel.colCount()) {
                                dLossWrtInputChannel.set(inputRow, inputCol, dLossWrtInputChannel.get(inputRow, inputCol) + dLossWrtWindowPixel);
                            }
                        }
                    }
                }
                else throw std::runtime_error("PoolLayerParameters calculateLossPartials: unhandled mode");
            }
        }
    }
};

void PoolLayerParameters::applyLossPartials(const FeatureLayerLossPartials* lossPartials)
{
    if (lossPartials->type != POOL) throw std::runtime_error("PoolLayerParameters applyLossPartials: lossPartials is wrong type");
};

std::string PoolLayerParameters::toString() const
{
    std::string output;

    output += "\t\twindow: " + this->window.toString() + "\n";
    output += "\t\tstride: " + std::to_string(this->stride) + "\n";
    output += "\t\tpadding: " + std::to_string(this->padding) + "\n";
    output += "\t\tmode: " + std::to_string(this->mode) + "\n";

    return output;
};

ActivationLayerParameters::ActivationLayerParameters(UnaryActivationFunction unaryActivationFunction)
{
    this->type = ACTIVATION;

    this->unaryActivationFunction = unaryActivationFunction;
};

ActivationLayerParameters::ActivationLayerParameters(UnaryActivationFunction unaryActivationFunction, std::vector<float> weights, std::vector<float> bias)
{
    this->type = ACTIVATION;

    this->unaryActivationFunction = unaryActivationFunction;

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

Dimensions ActivationLayerParameters::getOutputDimensions(const Dimensions& inputDimensions) const
{
    return inputDimensions;
};

void ActivationLayerParameters::runFeedforward(const Tensor& input, FeatureLayerState* state) const
{
    if (state->type != ACTIVATION) throw std::runtime_error("ActivationLayerParameters runFeedforward: state is wrong type");

    auto activationLayerState = static_cast<ActivationLayerState*>(state);

    auto inputDepth = input.getDimensions().depth;

    if (this->weights.size() != inputDepth) throw std::runtime_error("ActivationLayerParameters runFeedforward: input is the wrong depth");

    activationLayerState->input = input;

    activationLayerState->weightedAndBiased = Tensor(input);
    activationLayerState->output = Tensor(input.getDimensions());
    
    for (int i = 0;i<inputDepth;i++) {
        auto& matrix = activationLayerState->weightedAndBiased.dangerouslyGetData()[i];

        for (auto& value : matrix.dangerouslyGetData()) {
            value *= this->weights[i];
            value += this->bias[i];
        }

        activationLayerState->output.dangerouslyGetData()[i] = evaluateUnaryActivationFunction(this->unaryActivationFunction, activationLayerState->weightedAndBiased.getMatrix(i));
    }
};

void ActivationLayerParameters::calculateLossPartials(const Tensor& dLossWrtOutput, FeatureLayerState* state) const
{
    if (state->type != ACTIVATION) throw std::runtime_error("ActivationLayerParameters calculateLossPartials: state is wrong type");

    auto activationLayerState = static_cast<ActivationLayerState*>(state);

    auto inputDepth = activationLayerState->input.getDimensions().depth;

    activationLayerState->dLossWrtOutput = dLossWrtOutput;
    activationLayerState->dLossWrtWeightedAndBiased.dangerouslyGetData().resize(dLossWrtOutput.getDimensions().depth);
    activationLayerState->dLossWrtWeights = std::vector<float>(this->weights.size());
    activationLayerState->dLossWrtBias = std::vector<float>(this->bias.size());
    activationLayerState->dLossWrtInput = Tensor(activationLayerState->input.getDimensions());

    for (int channel = 0;channel<inputDepth;channel++) {
        auto dLossWrtOutputChannel = activationLayerState->dLossWrtOutput.getMatrix(channel);

        auto& dLossWrtWeightedAndBiasedChannel = activationLayerState->dLossWrtWeightedAndBiased.dangerouslyGetData()[channel];

        auto weightedAndBiasedMatrix = activationLayerState->weightedAndBiased.getMatrix(channel);
        auto outputMatrix = activationLayerState->output.getMatrix(channel);

        dLossWrtWeightedAndBiasedChannel = Matrix::hadamardProduct(
            dLossWrtOutputChannel,
            unaryActivationFunctionDerivative(this->unaryActivationFunction, weightedAndBiasedMatrix, outputMatrix)
        );

        for (int r = 0;r<dLossWrtWeightedAndBiasedChannel.rowCount();r++) {
            for (int c = 0;c<dLossWrtWeightedAndBiasedChannel.colCount();c++) {
                auto dLossWrtWeightedAndBiasedPixel = dLossWrtWeightedAndBiasedChannel.get(r, c);
                auto inputPixel = activationLayerState->input.get(r, c, channel);

                activationLayerState->dLossWrtWeights[channel] += dLossWrtWeightedAndBiasedPixel * inputPixel;
                activationLayerState->dLossWrtBias[channel] += dLossWrtWeightedAndBiasedPixel;
            }
        }

        activationLayerState->dLossWrtInput.dangerouslyGetData()[channel] = Matrix::scalarProduct(dLossWrtWeightedAndBiasedChannel, this->weights[channel]);
    }
};

void ActivationLayerParameters::applyLossPartials(const FeatureLayerLossPartials* lossPartials)
{
    if (lossPartials->type != ACTIVATION) throw std::runtime_error("ActivationLayerParameters applyLossPartials: lossPartials is wrong type");

    auto activationLayerLossPartials = *static_cast<const ActivationLayerLossPartials*>(lossPartials);

    if (this->weights.size() != activationLayerLossPartials.weights.size()) throw std::runtime_error("ActivationLayerParameters applyLossPartials: lossPartials weights are the wrong size");

    for (int i = 0;i<this->weights.size();i++) {
        this->weights[i] += activationLayerLossPartials.weights[i];
        this->bias[i] += activationLayerLossPartials.bias[i];
    }
};

std::string ActivationLayerParameters::toString() const
{
    std::string output;
    
    output += "\t\tweights: " + Matrix({ this->weights }).toString() + "\n";
    output += "\t\tbias: " + Matrix({ this->bias }).toString() + "\n";

    return output;
};

// convolutional network loss partials

ConvolutionalNetworkLossPartials::ConvolutionalNetworkLossPartials(const Tensor& inputLayerLossPartials, std::vector<FeatureLayerLossPartials*> featureLayersLossPartials, const NetworkLossPartials& neuralNetworkLossPartials)
{
    this->inputLayerLossPartials = inputLayerLossPartials;

    this->featureLayersLossPartials.clear();

    for (auto featureLayerLossPartials : featureLayersLossPartials) this->featureLayersLossPartials.emplace_back(featureLayerLossPartials);

    this->neuralNetworkLossPartials = neuralNetworkLossPartials;
};

void ConvolutionalNetworkLossPartials::add(const ConvolutionalNetworkLossPartials& other)
{
    if (this->inputLayerLossPartials.getDimensions() != other.inputLayerLossPartials.getDimensions()) throw std::runtime_error("ConvolutionalNetworkLossPartials add: other inputLayerLossPartials has wrong dimensions");
    
    if (this->featureLayersLossPartials.size() != other.featureLayersLossPartials.size()) throw std::runtime_error("ConvolutionalNetworkLossPartials add: other featureLayersLossPartials is a different size");

    this->inputLayerLossPartials = Tensor::add(this->inputLayerLossPartials, other.inputLayerLossPartials);

    for (int i = 0;i<this->featureLayersLossPartials.size();i++) this->featureLayersLossPartials[i]->add(other.featureLayersLossPartials[i].get());

    this->neuralNetworkLossPartials.add(other.neuralNetworkLossPartials);
};

void ConvolutionalNetworkLossPartials::scalarMultiply(float scalar)
{
    this->inputLayerLossPartials = Tensor::scalarProduct(this->inputLayerLossPartials, scalar);

    for (auto& featureLayerLossPartials : this->featureLayersLossPartials) featureLayerLossPartials->scalarMultiply(scalar);

    this->neuralNetworkLossPartials.scalarMultiply(scalar);
};

// convolutional neural network

ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(const Dimensions& inputLayerDimensions, std::vector<FeatureLayerParameters*> featureLayersParameters, std::vector<HiddenLayerParameters> hiddenLayerParameters, NormalizationFunction outputNormalizationFunction, LossFunction lossFunction)
{
    this->inputLayerDimensions = inputLayerDimensions;

    if (featureLayersParameters.empty()) throw std::runtime_error("ConvolutionalNeuralNetwork constructor: featureLayerParameters is empty");

    auto featureLayersOutputDimensions = inputLayerDimensions;
    
    for (auto featureLayerParameters : featureLayersParameters) {
        this->featureLayerParameters.emplace_back(featureLayerParameters);
        
        switch (featureLayerParameters->type) {
            case CONVOLUTION:
                this->featureLayerStates.emplace_back(new ConvolutionLayerState());
                break;
            case POOL:
                this->featureLayerStates.emplace_back(new PoolLayerState());
                break;
            case ACTIVATION:
                this->featureLayerStates.emplace_back(new ActivationLayerState());
                break;
            default:
                throw std::runtime_error("ConvolutionalNeuralNetwork constructor: unhandled featureLayerParameters type");
        }

        featureLayersOutputDimensions = featureLayerParameters->getOutputDimensions(featureLayersOutputDimensions);
    }

    this->neuralNetwork = NeuralNetwork(
        featureLayersOutputDimensions.volume(),
        hiddenLayerParameters,
        outputNormalizationFunction,
        lossFunction
    );
};

Dimensions ConvolutionalNeuralNetwork::getInputLayerDimensions()
{
    return this->inputLayerDimensions;
};

const std::vector<std::unique_ptr<FeatureLayerState>>& ConvolutionalNeuralNetwork::getFeatureLayerStates()
{
    return this->featureLayerStates;
};

const std::vector<std::unique_ptr<FeatureLayerParameters>>& ConvolutionalNeuralNetwork::getFeatureLayerParameters()
{
    return this->featureLayerParameters;
};

NeuralNetwork ConvolutionalNeuralNetwork::getNeuralNetwork()
{
    return this->neuralNetwork;
};

Matrix ConvolutionalNeuralNetwork::getNormalizedOutput()
{
    return this->neuralNetwork.getNormalizedOutput();
};

void ConvolutionalNeuralNetwork::initializeRandomHiddenLayerParameters()
{
    this->neuralNetwork.initializeRandomHiddenLayerParameters();
}

void ConvolutionalNeuralNetwork::initializeRandomHiddenLayerParameters(float minInitialWeight, float maxInitialWeight, float minInitialBias, float maxInitialBias)
{
    this->neuralNetwork.initializeRandomHiddenLayerParameters(minInitialWeight, maxInitialWeight, minInitialBias, maxInitialBias);
}

void ConvolutionalNeuralNetwork::initializeRandomFeatureLayerParameters()
{
    std::random_device rd;
    std::mt19937 rng(rd());

    auto featureLayersOutputDimensions = inputLayerDimensions;

    for (auto& featureLayerParameters : this->featureLayerParameters) {
        featureLayerParameters->randomizeParameters(featureLayersOutputDimensions, rng);

        featureLayersOutputDimensions = featureLayerParameters->getOutputDimensions(featureLayersOutputDimensions);
    }
};

Matrix ConvolutionalNeuralNetwork::calculateFeedForwardOutput(const Tensor& input)
{
    if (input.getDimensions() != this->inputLayerDimensions) throw std::runtime_error("ConvolutionalNeuralNetwork calculateFeedforwardOutput: input is the wrong dimensions");

    this->featureLayerParameters[0]->runFeedforward(input, this->featureLayerStates[0].get());

    for (int i = 1;i<this->featureLayerStates.size();i++) this->featureLayerParameters[i]->runFeedforward(this->featureLayerStates[i - 1]->output, this->featureLayerStates[i].get());

    return this->neuralNetwork.calculateFeedForwardOutput(this->featureLayerStates.back()->output.getColumnVector());
};

float ConvolutionalNeuralNetwork::calculateLoss(const Tensor& input, const Matrix& expectedOutput)
{
    auto predictedValues = this->calculateFeedForwardOutput(input);

    return evaluateLossFunction(this->neuralNetwork.getLossFunction(), predictedValues, expectedOutput);
};

ConvolutionalNetworkLossPartials ConvolutionalNeuralNetwork::calculateLossPartials(NetworkLossPartials neuralNetworkLossPartials)
{
    auto dLossWrtOutput = Tensor::fromColumnVector(neuralNetworkLossPartials.inputLayerLossPartials, this->featureLayerStates.back()->output.getDimensions());

    this->featureLayerParameters.back()->calculateLossPartials(dLossWrtOutput, this->featureLayerStates.back().get());

    for (int i = this->featureLayerParameters.size() - 2;i>=0;i--) this->featureLayerParameters[i]->calculateLossPartials(this->featureLayerStates[i + 1]->dLossWrtInput, this->featureLayerStates[i].get());

    std::vector<FeatureLayerLossPartials*> featureLayerLossPartials(this->featureLayerStates.size());

    for (int i = 0;i<this->featureLayerStates.size();i++) featureLayerLossPartials[i] = this->featureLayerStates[i]->getLossPartials();

    auto inputLayerLossPartials = this->featureLayerStates[0]->dLossWrtInput;

    return ConvolutionalNetworkLossPartials(inputLayerLossPartials, featureLayerLossPartials, neuralNetworkLossPartials);
};

ConvolutionalNetworkLossPartials ConvolutionalNeuralNetwork::calculateLossPartials(TensorDataPoint dataPoint)
{
    this->calculateFeedForwardOutput(dataPoint.input);

    auto neuralNetworkLossPartials = this->neuralNetwork.calculateLossPartials(dataPoint.expectedOutput);
    
    return this->calculateLossPartials(neuralNetworkLossPartials);
};

ConvolutionalNetworkLossPartials ConvolutionalNeuralNetwork::calculateBatchLossPartials(std::vector<TensorDataPoint> dataBatch)
{
    if (dataBatch.empty()) throw std::runtime_error("ConvolutionalNeuralNetwork calculateBatchLossPartials: dataBatch is empty");

    auto averageLossPartials = this->calculateLossPartials(dataBatch[0]);

    for (int i = 1;i<dataBatch.size();i++) averageLossPartials.add(this->calculateLossPartials(dataBatch[i]));
    
    averageLossPartials.scalarMultiply(1.0 / dataBatch.size());
    
    return averageLossPartials;
};

void ConvolutionalNeuralNetwork::applyLossPartials(ConvolutionalNetworkLossPartials lossPartials)
{
    if (this->featureLayerParameters.size() != lossPartials.featureLayersLossPartials.size()) throw std::runtime_error("ConvolutionalNeuralNetwork applyLossPartials: different number of feature layers");

    this->neuralNetwork.applyLossPartials(lossPartials.neuralNetworkLossPartials);

    for (int i = 0;i<lossPartials.featureLayersLossPartials.size();i++) this->featureLayerParameters[i]->applyLossPartials(lossPartials.featureLayersLossPartials[i].get());
};

void ConvolutionalNeuralNetwork::train(TensorDataPoint trainingDataPoint, float learningRate)
{
    auto parameterAdjustements = this->calculateLossPartials(trainingDataPoint);
    parameterAdjustements.scalarMultiply(-learningRate);

    this->applyLossPartials(std::move(parameterAdjustements));
};

void ConvolutionalNeuralNetwork::batchTrain(std::vector<TensorDataPoint> trainingDataBatch, float learningRate)
{
    auto parameterAdjustements = this->calculateBatchLossPartials(trainingDataBatch);
    parameterAdjustements.scalarMultiply(-learningRate);

    this->applyLossPartials(std::move(parameterAdjustements));
};

std::string ConvolutionalNeuralNetwork::toString()
{
    std::string output;

    output += "<CNN> {\n";

    output += "\tInput Layer (" + this->inputLayerDimensions.toString() + ")\n\n";

    auto featureLayersOutputDimensions = this->inputLayerDimensions;

    for (int i = 0;i<this->featureLayerParameters.size();i++) {
        const auto& parameters = this->featureLayerParameters[i];
        const auto& state = this->featureLayerStates[i];

        auto layerInputDimensions = featureLayersOutputDimensions;
        featureLayersOutputDimensions = parameters->getOutputDimensions(featureLayersOutputDimensions);

        output += "\tFeature Layer (" + layerInputDimensions.toString() + ", type " + std::to_string(parameters->type) + ")\n";

        output += parameters->toString() + "\n";

        if (state.get() == nullptr) continue;

        output += "\n";

        output += state->toString();
    }

    output += "}\n";

    output += "<NN> " + this->neuralNetwork.toString();

    return output;
};