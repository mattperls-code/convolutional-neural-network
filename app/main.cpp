#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/memory.hpp>

#include "../src/convolutional_neural_network.hpp"

ConvolutionalNeuralNetwork cnn;

void initModel()
{
    cnn = ConvolutionalNeuralNetwork(
        Dimensions(1, Shape(28, 28)),
        {
            new PoolLayerParameters(AVG, Shape(2, 2), 2, 0),
            new ConvolutionLayerParameters(8, Shape(5, 5), 2, 0),
            new ActivationLayerParameters(RELU),
            new ConvolutionLayerParameters(16, Shape(7, 7), 1, 2),
            new PoolLayerParameters(MAX, Shape(2, 2), 1, 0)
        },
        {
            HiddenLayerParameters(24, SIGMOID),
            HiddenLayerParameters(10, LINEAR)
        },
        SOFTMAX,
        CATEGORICAL_CROSS_ENTROPY
    );

    cnn.initializeRandomFeatureLayerParameters();
    cnn.initializeRandomHiddenLayerParameters();
};

void cli_load()
{
    std::cout << "TODO: load" << std::endl << std::endl;
};

void cli_save()
{
    std::cout << "TODO: save" << std::endl << std::endl;
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

    int trainingBatchSize = 1000;
    int batches = 60000 / trainingBatchSize;

    for (int i = 0;i<cycleCount;i++) {
        int batchIndex = 0; // i % batches;

        auto testDataPoint = trainingData[0];

        auto lossBefore = cnn.calculateLoss(testDataPoint.input, testDataPoint.expectedOutput);

        std::cout << "loss: " << lossBefore << std::endl;

        cnn.batchTrain(
            std::vector<TensorDataPoint>(
                trainingData.begin() + trainingBatchSize * batchIndex,
                trainingData.begin() + trainingBatchSize * (batchIndex + 1)
            ),
            0.1
        );

        auto lossAfter = cnn.calculateLoss(testDataPoint.input, testDataPoint.expectedOutput);

        std::cout << "Finished Cycle " << i << std::endl;
    }

    std::cout << "Finished Training" << std::endl << std::endl;
};

void cli_test()
{
    std::cout << "TODO: test" << std::endl << std::endl;
};

void cli_draw()
{
    std::cout << "TODO: draw" << std::endl << std::endl;
};

int main()
{
    initModel();

    std::cout << std::endl << "Welcome to the Convolutional Neural Network demo!" << std::endl << std::endl;

    while (true) {
        std::string command;

        std::cout << "Enter a command (LOAD, SAVE, TRAIN, TEST, DRAW, EXIT): ";

        std::cin >> command;

        std::cout << std::endl;

        if (command == "LOAD") cli_load();

        else if (command == "SAVE") cli_save();

        else if (command == "TRAIN") cli_train();

        else if (command == "TEST") cli_test();

        else if (command == "DRAW") cli_draw();

        else if (command == "EXIT") break;

        else std::cout << "Command \"" << command << "\" is unrecognized." << std::endl << std::endl;
    }

    return 0;
};