#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/memory.hpp>

#include "../src/convolutional_neural_network.hpp"
#include "../src/image_operations.hpp"

ConvolutionalNeuralNetwork cnn;

void initModel()
{
    cnn = ConvolutionalNeuralNetwork(
        Dimensions(1, Shape(28, 28)),
        {
            new ConvolutionLayerParameters(8, Shape(3, 3), 1, 1),
            new PoolLayerParameters(AVG, Shape(2, 2), 2, 0),
            new ActivationLayerParameters(TANH),
            new ConvolutionLayerParameters(16, Shape(3, 3), 1, 1),
            new PoolLayerParameters(AVG, Shape(2, 2), 2, 0)
        },
        {
            HiddenLayerParameters(64, RELU),
            HiddenLayerParameters(10, LINEAR)
        },
        SOFTMAX,
        CATEGORICAL_CROSS_ENTROPY
    );

    cnn.initializeRandomFeatureLayerParameters();
    cnn.initializeRandomHiddenLayerParameters(-0.1, 0.1, -0.1, 0.1);
};

void cli_load()
{
    std::cout << "Saved Model File Path? ";

    std::string savedModelFilePath;

    std::cin >> savedModelFilePath;

    std::cout << std::endl;

    bool loaded = cnn.load(savedModelFilePath);

    if (loaded) std::cout << "Successfully Loaded";

    else std::cout << "An Error Occurred";

    std::cout << std::endl << std::endl;
};

void cli_save()
{
    std::cout << "Saved Model File Path? ";

    std::string savedModelFilePath;

    std::cin >> savedModelFilePath;

    std::cout << std::endl;

    bool saved = cnn.save(savedModelFilePath);

    if (saved) std::cout << "Successfully Saved";

    else std::cout << "An Error Occurred";

    std::cout << std::endl << std::endl;
};

void cli_train()
{
    std::cout << "Starting Training" << std::endl;

    std::cout << "How Many Cycles? ";

    unsigned int cycleCount;

    std::cin >> cycleCount;

    std::cout << std::endl;

    std::cout << "Initializing Training Data" << std::endl;

    std::vector<TensorDataPoint> trainingData;
    trainingData.reserve(60000);

    std::ifstream trainingDataFile("./app/mnist/training.csv");

    auto firstLine = true;
    std::string line;

    while (getline(trainingDataFile, line)) {
        if (firstLine) {
            firstLine = false;
            continue;
        }

        trainingData.emplace_back(Tensor(Dimensions(1, Shape(28, 28))), Matrix(Shape(10, 1)));

        std::stringstream lineStream(line);

        char label;

        lineStream >> label;

        auto labelIndex = label - '0';
        trainingData.back().expectedOutput.set(labelIndex, 0, 1.0);

        lineStream.ignore();
        
        std::string pixelValueStr;

        int pixelIndex = 0;

        while (getline(lineStream, pixelValueStr, ',')) {
            trainingData.back().input.dangerouslyGetData()[0].dangerouslyGetData()[pixelIndex] = std::stof(pixelValueStr) / 256.0;
            pixelIndex++;
        }
    }
    
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(trainingData.begin(), trainingData.end(), rng);
    
    std::cout << "Training Data Initialized" << std::endl;

    int trainingBatchSize = 1200;
    int batches = 60000 / trainingBatchSize;

    for (int i = 0;i<cycleCount;i++) {
        int batchIndex = i % batches;

        auto batchLoss = cnn.batchTrain(
            std::vector<TensorDataPoint>(
                trainingData.begin() + trainingBatchSize * batchIndex,
                trainingData.begin() + trainingBatchSize * (batchIndex + 1)
            ),
            0.1
        );

        std::cout << "Finished Cycle " << i << " with batch loss of " << batchLoss << std::endl;
    }

    std::cout << "Finished Training" << std::endl << std::endl;
};

void cli_test()
{
    std::cout << "Initializing Testing Data" << std::endl;

    std::vector<TensorDataPoint> testingData;
    testingData.reserve(60000);

    std::ifstream testingDataFile("./app/mnist/testing.csv");

    auto firstLine = true;
    std::string line;

    while (getline(testingDataFile, line)) {
        if (firstLine) {
            firstLine = false;
            continue;
        }

        testingData.emplace_back(Tensor(Dimensions(1, Shape(28, 28))), Matrix(Shape(10, 1)));

        std::stringstream lineStream(line);

        char label;

        lineStream >> label;

        auto labelIndex = label - '0';
        testingData.back().expectedOutput.set(labelIndex, 0, 1.0);

        lineStream.ignore();
        
        std::string pixelValueStr;

        int pixelIndex = 0;

        while (getline(lineStream, pixelValueStr, ',')) {
            testingData.back().input.dangerouslyGetData()[0].dangerouslyGetData()[pixelIndex] = std::stof(pixelValueStr) / 256.0;
            pixelIndex++;
        }
    }

    std::cout << "Testing Data Initialized" << std::endl;

    std::cout << "Calculating Average Testing Data Loss" << std::endl;

    auto averageLoss = 0.0;

    for (auto testDataPoint : testingData) averageLoss += cnn.calculateLoss(testDataPoint.input, testDataPoint.expectedOutput);

    averageLoss /= testingData.size();

    std::cout << "Average Testing Data Loss: " << averageLoss << std::endl << std::endl;

    return; // forces compiler to eval before early return
};

void cli_drawing()
{
    std::cout << "PNG Drawing File Path? ";

    std::string pngDrawingFilePath;

    std::cin >> pngDrawingFilePath;

    std::cout << std::endl;

    auto inputImage = ImageOperations::rgbToGreyscale(ImageOperations::pngToTensor(pngDrawingFilePath));

    for (auto& channel : inputImage.dangerouslyGetData()) channel = ImageOperations::resize(channel, Shape(28, 28));

    auto results = cnn.calculateFeedForwardOutput(inputImage);

    std::cout << "Output: " << results.toString() << std::endl << std::endl;
};

Tensor formatActivationMap(const Matrix& activationMap)
{
    Tensor output(Dimensions(3, activationMap.shape()));

    for (int r = 0;r<activationMap.shape().rows;r++) {
        for (int c = 0;c<activationMap.shape().cols;c++) {
            auto pixelValue = activationMap.get(r, c);

            if (pixelValue < 0) {
                auto pixelR = -255 * pixelValue;

                if (pixelR > 255) pixelR = 255;

                output.set(r, c, 0, pixelR);
            }
            else {
                auto pixelG = 255 * pixelValue;

                if (pixelG > 255) pixelG = 255;

                output.set(r, c, 1, pixelG);
            }
        }
    }

    return output;
};

void cli_analyze()
{
    std::cout << "Analysis Output Directory? ";

    std::string analysisOutputDirectory;

    std::cin >> analysisOutputDirectory;

    std::cout << "Generating Analysis" << std::endl;

    for (int i = 0;i<10;i++) {
        auto digitImageFilePath = "results/drawings/digit" + std::to_string(i) + ".png";

        auto digitImage = ImageOperations::rgbToGreyscale(ImageOperations::pngToTensor(digitImageFilePath));

        for (auto& channel : digitImage.dangerouslyGetData()) channel = ImageOperations::resize(channel, Shape(28, 28));

        cnn.calculateFeedForwardOutput(digitImage);

        for (int j = 0;j<cnn.getFeatureLayerStates().size();j++) {
            auto layerOutput = cnn.getFeatureLayerStates()[j]->output;

            for (int k = 0;k<layerOutput.getDimensions().depth;k++) {
                auto channelOutput = layerOutput.getMatrix(k);
    
                ImageOperations::tensorToPng(
                    analysisOutputDirectory + "/layer" + std::to_string(j) + "/channel" + std::to_string(k) + "/digit" + std::to_string(i) + ".png",
                    formatActivationMap(channelOutput)
                );
            }
        }
    }

    std::cout << "Generated Analysis" << std::endl << std::endl;
};

int main()
{
    initModel();

    std::cout << std::endl << "Welcome to the Convolutional Neural Network demo!" << std::endl << std::endl;

    while (true) {
        std::string command;

        std::cout << "Enter a command (LOAD, SAVE, TRAIN, TEST, DRAWING, ANALYZE, EXIT): ";

        std::cin >> command;

        std::cout << std::endl;

        if (command == "LOAD") cli_load();

        else if (command == "SAVE") cli_save();

        else if (command == "TRAIN") cli_train();

        else if (command == "TEST") cli_test();

        else if (command == "DRAWING") cli_drawing();

        else if (command == "ANALYZE") cli_analyze();

        else if (command == "EXIT") break;

        else std::cout << "Command \"" << command << "\" is unrecognized." << std::endl << std::endl;
    }

    return 0;
};