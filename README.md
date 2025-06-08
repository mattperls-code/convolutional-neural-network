# Convolutional Neural Network

A general purpose image processing model built without any math or machine learning libraries.

The ```ConvolutionalNeuralNetwork``` class manages initializing, saving, and retrieving network parameters, calculating feedforward output, determining loss on known inputs, single and batch training using gradient descent, and storing network state for analysis or chaining.

```cpp
/* Create Convolutional Neural Network Instance */

ConvolutionalNeuralNetwork myCNN(
    Dimensions(1, Shape(28, 28)), // input image dimensions: 1 channel of size 28x28
    {
        // convolution layer with 8 kernels of size 3x3 using stride=1, padding=1
        new ConvolutionLayerParameters(8, Shape(3, 3), 1, 1),

        // relu activation layer
        new ActivationLayerParameters(RELU),

        // convolutional layer with 16 kernels of size 3x3 using stride=1, padding=1
        new ConvolutionLayerParameters(16, Shape(3, 3), 1, 1),

        // max pool layer with window size 2x2 using stride=2, padding=0
        new PoolLayerParameters(MAX, Shape(2, 2), 2, 0)
    },
    {
        HiddenLayerParameters(64, RELU), // hidden layer with 64 nodes using relu activation

        HiddenLayerParameters(10, LINEAR) // output layer with 10 nodes using linear activation
    },
    SOFTMAX, // normalization function
    CATEGORICAL_CROSS_ENTROPY // loss function
);
```

```cpp
/* Initialize Parameters */

myCNN.initializeRandomFeatureLayerParameters();

// initial weight range -0.1 to 0.1, initial bias range -0.1 to 0.1
myCNN.initializeRandomHiddenLayerParameters(-0.1, 0.1, -0.1, 0.1);
```

```cpp
/* Load Parameters From A File */

myCNN.load("path/to/learnedParameters.json");

/* Save Parameters To A File */

myCNN.save("path/to/learnedParameters.json");
```

```cpp
/* Use TensorDataPoint For Training */

std::vector<TensorDataPoint> trainingDataPoints;

trainingDataPoints.emplace_back(
    Tensor(/* ...28x28x1 data... */), // input
    Matrix(/* ...10x1 data... */) // expected output
);

// ... add other training data ...
```

```cpp
/* Train Using A Single TensorDataPoint */

float learningRate = 0.1;

myCNN.train(trainingDataPoints[0], learningRate);
```

```cpp
/* Train Using A Batch */

float learningRate = 0.05;

myCNN.batchTrain(trainingDataPoints, learningRate);
```

```cpp
/* Make A Prediction */

Tensor input = Tensor(/* ...28x28x1 data... */ );

Matrix output = myCNN.calculateFeedForwardOutput(input);
```

```cpp
/* Calculate The Loss For A Known Point */

Matrix expectedOutput = Matrix(/* ...10x1 data... */);

float loss = myCNN.calculateLoss(input, expectedOutput);
```

## MNIST Digit Classification Example

![Digit 0](results/drawings/digit0.png)
![Digit 1](results/drawings/digit1.png)
![Digit 2](results/drawings/digit2.png)
![Digit 3](results/drawings/digit3.png)
![Digit 4](results/drawings/digit4.png)
![Digit 5](results/drawings/digit5.png)
![Digit 6](results/drawings/digit6.png)
![Digit 7](results/drawings/digit7.png)
![Digit 8](results/drawings/digit8.png)
![Digit 9](results/drawings/digit9.png)

### Network Structure and Initialization

```cpp
ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork(
    Dimensions(1, Shape(28, 28)),
    {
        new ConvolutionLayerParameters(8, Shape(3, 3), 1, 1),
        new ActivationLayerParameters(RELU),
        new ConvolutionLayerParameters(16, Shape(3, 3), 1, 1),
        new PoolLayerParameters(MAX, Shape(2, 2), 2, 0)
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
```

### Training Progression

The model was trained in randomly partitioned batches of 1200 images on a dataset of 60k images.

The loss and accuracy were calculated against a separate testing dataset of 10k images.

<table border="1">
  <tr>
    <th>Epoch</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th>
  </tr>
  <tr>
    <td>Loss</td><td>4.06674</td><td>0.53085</td><td>0.28210</td><td>0.23416</td><td>0.17990</td><td>0.15610</td>
  </tr>
  <tr>
    <td>Accuracy</td><td>0.1494</td><td>0.8453</td><td>0.9162</td><td>0.9296</td><td>0.9451</td><td>0.9518</td>
  </tr>
</table>

### Feature Analysis

Visualizing the outputs of different layers gives insight into how certain features of the data are used.

Each channel roughly represents one feature that a convolutional kernel identifies.

To explore this, we can look at snapshots of the output channels throughout the feedforward process:

### Layer 1 (Convolution)

Each channel can be identified as roughly one type of feature (vertical line, bottom left corner, etc).

| Digit   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---------|---|---|---|---|---|---|---|---|---|---|
| Channel 0 | ![](results/analysis/layer0/channel0/digit0.png) | ![](results/analysis/layer0/channel0/digit1.png) | ![](results/analysis/layer0/channel0/digit2.png) | ![](results/analysis/layer0/channel0/digit3.png) | ![](results/analysis/layer0/channel0/digit4.png) | ![](results/analysis/layer0/channel0/digit5.png) | ![](results/analysis/layer0/channel0/digit6.png) | ![](results/analysis/layer0/channel0/digit7.png) | ![](results/analysis/layer0/channel0/digit8.png) | ![](results/analysis/layer0/channel0/digit9.png) |
| Channel 1 | ![](results/analysis/layer0/channel1/digit0.png) | ![](results/analysis/layer0/channel1/digit1.png) | ![](results/analysis/layer0/channel1/digit2.png) | ![](results/analysis/layer0/channel1/digit3.png) | ![](results/analysis/layer0/channel1/digit4.png) | ![](results/analysis/layer0/channel1/digit5.png) | ![](results/analysis/layer0/channel1/digit6.png) | ![](results/analysis/layer0/channel1/digit7.png) | ![](results/analysis/layer0/channel1/digit8.png) | ![](results/analysis/layer0/channel1/digit9.png) |
| Channel 2 | ![](results/analysis/layer0/channel2/digit0.png) | ![](results/analysis/layer0/channel2/digit1.png) | ![](results/analysis/layer0/channel2/digit2.png) | ![](results/analysis/layer0/channel2/digit3.png) | ![](results/analysis/layer0/channel2/digit4.png) | ![](results/analysis/layer0/channel2/digit5.png) | ![](results/analysis/layer0/channel2/digit6.png) | ![](results/analysis/layer0/channel2/digit7.png) | ![](results/analysis/layer0/channel2/digit8.png) | ![](results/analysis/layer0/channel2/digit9.png) |
| Channel 3 | ![](results/analysis/layer0/channel3/digit0.png) | ![](results/analysis/layer0/channel3/digit1.png) | ![](results/analysis/layer0/channel3/digit2.png) | ![](results/analysis/layer0/channel3/digit3.png) | ![](results/analysis/layer0/channel3/digit4.png) | ![](results/analysis/layer0/channel3/digit5.png) | ![](results/analysis/layer0/channel3/digit6.png) | ![](results/analysis/layer0/channel3/digit7.png) | ![](results/analysis/layer0/channel3/digit8.png) | ![](results/analysis/layer0/channel3/digit9.png) |
| Channel 4 | ![](results/analysis/layer0/channel4/digit0.png) | ![](results/analysis/layer0/channel4/digit1.png) | ![](results/analysis/layer0/channel4/digit2.png) | ![](results/analysis/layer0/channel4/digit3.png) | ![](results/analysis/layer0/channel4/digit4.png) | ![](results/analysis/layer0/channel4/digit5.png) | ![](results/analysis/layer0/channel4/digit6.png) | ![](results/analysis/layer0/channel4/digit7.png) | ![](results/analysis/layer0/channel4/digit8.png) | ![](results/analysis/layer0/channel4/digit9.png) |
| Channel 5 | ![](results/analysis/layer0/channel5/digit0.png) | ![](results/analysis/layer0/channel5/digit1.png) | ![](results/analysis/layer0/channel5/digit2.png) | ![](results/analysis/layer0/channel5/digit3.png) | ![](results/analysis/layer0/channel5/digit4.png) | ![](results/analysis/layer0/channel5/digit5.png) | ![](results/analysis/layer0/channel5/digit6.png) | ![](results/analysis/layer0/channel5/digit7.png) | ![](results/analysis/layer0/channel5/digit8.png) | ![](results/analysis/layer0/channel5/digit9.png) |
| Channel 6 | ![](results/analysis/layer0/channel6/digit0.png) | ![](results/analysis/layer0/channel6/digit1.png) | ![](results/analysis/layer0/channel6/digit2.png) | ![](results/analysis/layer0/channel6/digit3.png) | ![](results/analysis/layer0/channel6/digit4.png) | ![](results/analysis/layer0/channel6/digit5.png) | ![](results/analysis/layer0/channel6/digit6.png) | ![](results/analysis/layer0/channel6/digit7.png) | ![](results/analysis/layer0/channel6/digit8.png) | ![](results/analysis/layer0/channel6/digit9.png) |
| Channel 7 | ![](results/analysis/layer0/channel7/digit0.png) | ![](results/analysis/layer0/channel7/digit1.png) | ![](results/analysis/layer0/channel7/digit2.png) | ![](results/analysis/layer0/channel7/digit3.png) | ![](results/analysis/layer0/channel7/digit4.png) | ![](results/analysis/layer0/channel7/digit5.png) | ![](results/analysis/layer0/channel7/digit6.png) | ![](results/analysis/layer0/channel7/digit7.png) | ![](results/analysis/layer0/channel7/digit8.png) | ![](results/analysis/layer0/channel7/digit9.png) |

### Layer 2 (Relu Activation)

Looks like most of the layers got zeroed out? Either unlucky initial bias + relu or an odd training choice. Gives insight into which activations may be dangerous early on

| Digit   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---------|---|---|---|---|---|---|---|---|---|---|
| Channel 0 | ![](results/analysis/layer1/channel0/digit0.png) | ![](results/analysis/layer1/channel0/digit1.png) | ![](results/analysis/layer1/channel0/digit2.png) | ![](results/analysis/layer1/channel0/digit3.png) | ![](results/analysis/layer1/channel0/digit4.png) | ![](results/analysis/layer1/channel0/digit5.png) | ![](results/analysis/layer1/channel0/digit6.png) | ![](results/analysis/layer1/channel0/digit7.png) | ![](results/analysis/layer1/channel0/digit8.png) | ![](results/analysis/layer1/channel0/digit9.png) |
| Channel 1 | ![](results/analysis/layer1/channel1/digit0.png) | ![](results/analysis/layer1/channel1/digit1.png) | ![](results/analysis/layer1/channel1/digit2.png) | ![](results/analysis/layer1/channel1/digit3.png) | ![](results/analysis/layer1/channel1/digit4.png) | ![](results/analysis/layer1/channel1/digit5.png) | ![](results/analysis/layer1/channel1/digit6.png) | ![](results/analysis/layer1/channel1/digit7.png) | ![](results/analysis/layer1/channel1/digit8.png) | ![](results/analysis/layer1/channel1/digit9.png) |
| Channel 2 | ![](results/analysis/layer1/channel2/digit0.png) | ![](results/analysis/layer1/channel2/digit1.png) | ![](results/analysis/layer1/channel2/digit2.png) | ![](results/analysis/layer1/channel2/digit3.png) | ![](results/analysis/layer1/channel2/digit4.png) | ![](results/analysis/layer1/channel2/digit5.png) | ![](results/analysis/layer1/channel2/digit6.png) | ![](results/analysis/layer1/channel2/digit7.png) | ![](results/analysis/layer1/channel2/digit8.png) | ![](results/analysis/layer1/channel2/digit9.png) |
| Channel 3 | ![](results/analysis/layer1/channel3/digit0.png) | ![](results/analysis/layer1/channel3/digit1.png) | ![](results/analysis/layer1/channel3/digit2.png) | ![](results/analysis/layer1/channel3/digit3.png) | ![](results/analysis/layer1/channel3/digit4.png) | ![](results/analysis/layer1/channel3/digit5.png) | ![](results/analysis/layer1/channel3/digit6.png) | ![](results/analysis/layer1/channel3/digit7.png) | ![](results/analysis/layer1/channel3/digit8.png) | ![](results/analysis/layer1/channel3/digit9.png) |
| Channel 4 | ![](results/analysis/layer1/channel4/digit0.png) | ![](results/analysis/layer1/channel4/digit1.png) | ![](results/analysis/layer1/channel4/digit2.png) | ![](results/analysis/layer1/channel4/digit3.png) | ![](results/analysis/layer1/channel4/digit4.png) | ![](results/analysis/layer1/channel4/digit5.png) | ![](results/analysis/layer1/channel4/digit6.png) | ![](results/analysis/layer1/channel4/digit7.png) | ![](results/analysis/layer1/channel4/digit8.png) | ![](results/analysis/layer1/channel4/digit9.png) |
| Channel 5 | ![](results/analysis/layer1/channel5/digit0.png) | ![](results/analysis/layer1/channel5/digit1.png) | ![](results/analysis/layer1/channel5/digit2.png) | ![](results/analysis/layer1/channel5/digit3.png) | ![](results/analysis/layer1/channel5/digit4.png) | ![](results/analysis/layer1/channel5/digit5.png) | ![](results/analysis/layer1/channel5/digit6.png) | ![](results/analysis/layer1/channel5/digit7.png) | ![](results/analysis/layer1/channel5/digit8.png) | ![](results/analysis/layer1/channel5/digit9.png) |
| Channel 6 | ![](results/analysis/layer1/channel6/digit0.png) | ![](results/analysis/layer1/channel6/digit1.png) | ![](results/analysis/layer1/channel6/digit2.png) | ![](results/analysis/layer1/channel6/digit3.png) | ![](results/analysis/layer1/channel6/digit4.png) | ![](results/analysis/layer1/channel6/digit5.png) | ![](results/analysis/layer1/channel6/digit6.png) | ![](results/analysis/layer1/channel6/digit7.png) | ![](results/analysis/layer1/channel6/digit8.png) | ![](results/analysis/layer1/channel6/digit9.png) |
| Channel 7 | ![](results/analysis/layer1/channel7/digit0.png) | ![](results/analysis/layer1/channel7/digit1.png) | ![](results/analysis/layer1/channel7/digit2.png) | ![](results/analysis/layer1/channel7/digit3.png) | ![](results/analysis/layer1/channel7/digit4.png) | ![](results/analysis/layer1/channel7/digit5.png) | ![](results/analysis/layer1/channel7/digit6.png) | ![](results/analysis/layer1/channel7/digit7.png) | ![](results/analysis/layer1/channel7/digit8.png) | ![](results/analysis/layer1/channel7/digit9.png) |

### Layer 3 (Convolution)

Again we see many distinct features for a given kernel.

| Digit     | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-----------|---|---|---|---|---|---|---|---|---|---|
| Channel 0  | ![](results/analysis/layer2/channel0/digit0.png) | ![](results/analysis/layer2/channel0/digit1.png) | ![](results/analysis/layer2/channel0/digit2.png) | ![](results/analysis/layer2/channel0/digit3.png) | ![](results/analysis/layer2/channel0/digit4.png) | ![](results/analysis/layer2/channel0/digit5.png) | ![](results/analysis/layer2/channel0/digit6.png) | ![](results/analysis/layer2/channel0/digit7.png) | ![](results/analysis/layer2/channel0/digit8.png) | ![](results/analysis/layer2/channel0/digit9.png) |
| Channel 1  | ![](results/analysis/layer2/channel1/digit0.png) | ![](results/analysis/layer2/channel1/digit1.png) | ![](results/analysis/layer2/channel1/digit2.png) | ![](results/analysis/layer2/channel1/digit3.png) | ![](results/analysis/layer2/channel1/digit4.png) | ![](results/analysis/layer2/channel1/digit5.png) | ![](results/analysis/layer2/channel1/digit6.png) | ![](results/analysis/layer2/channel1/digit7.png) | ![](results/analysis/layer2/channel1/digit8.png) | ![](results/analysis/layer2/channel1/digit9.png) |
| Channel 2  | ![](results/analysis/layer2/channel2/digit0.png) | ![](results/analysis/layer2/channel2/digit1.png) | ![](results/analysis/layer2/channel2/digit2.png) | ![](results/analysis/layer2/channel2/digit3.png) | ![](results/analysis/layer2/channel2/digit4.png) | ![](results/analysis/layer2/channel2/digit5.png) | ![](results/analysis/layer2/channel2/digit6.png) | ![](results/analysis/layer2/channel2/digit7.png) | ![](results/analysis/layer2/channel2/digit8.png) | ![](results/analysis/layer2/channel2/digit9.png) |
| Channel 3  | ![](results/analysis/layer2/channel3/digit0.png) | ![](results/analysis/layer2/channel3/digit1.png) | ![](results/analysis/layer2/channel3/digit2.png) | ![](results/analysis/layer2/channel3/digit3.png) | ![](results/analysis/layer2/channel3/digit4.png) | ![](results/analysis/layer2/channel3/digit5.png) | ![](results/analysis/layer2/channel3/digit6.png) | ![](results/analysis/layer2/channel3/digit7.png) | ![](results/analysis/layer2/channel3/digit8.png) | ![](results/analysis/layer2/channel3/digit9.png) |
| Channel 4  | ![](results/analysis/layer2/channel4/digit0.png) | ![](results/analysis/layer2/channel4/digit1.png) | ![](results/analysis/layer2/channel4/digit2.png) | ![](results/analysis/layer2/channel4/digit3.png) | ![](results/analysis/layer2/channel4/digit4.png) | ![](results/analysis/layer2/channel4/digit5.png) | ![](results/analysis/layer2/channel4/digit6.png) | ![](results/analysis/layer2/channel4/digit7.png) | ![](results/analysis/layer2/channel4/digit8.png) | ![](results/analysis/layer2/channel4/digit9.png) |
| Channel 5  | ![](results/analysis/layer2/channel5/digit0.png) | ![](results/analysis/layer2/channel5/digit1.png) | ![](results/analysis/layer2/channel5/digit2.png) | ![](results/analysis/layer2/channel5/digit3.png) | ![](results/analysis/layer2/channel5/digit4.png) | ![](results/analysis/layer2/channel5/digit5.png) | ![](results/analysis/layer2/channel5/digit6.png) | ![](results/analysis/layer2/channel5/digit7.png) | ![](results/analysis/layer2/channel5/digit8.png) | ![](results/analysis/layer2/channel5/digit9.png) |
| Channel 6  | ![](results/analysis/layer2/channel6/digit0.png) | ![](results/analysis/layer2/channel6/digit1.png) | ![](results/analysis/layer2/channel6/digit2.png) | ![](results/analysis/layer2/channel6/digit3.png) | ![](results/analysis/layer2/channel6/digit4.png) | ![](results/analysis/layer2/channel6/digit5.png) | ![](results/analysis/layer2/channel6/digit6.png) | ![](results/analysis/layer2/channel6/digit7.png) | ![](results/analysis/layer2/channel6/digit8.png) | ![](results/analysis/layer2/channel6/digit9.png) |
| Channel 7  | ![](results/analysis/layer2/channel7/digit0.png) | ![](results/analysis/layer2/channel7/digit1.png) | ![](results/analysis/layer2/channel7/digit2.png) | ![](results/analysis/layer2/channel7/digit3.png) | ![](results/analysis/layer2/channel7/digit4.png) | ![](results/analysis/layer2/channel7/digit5.png) | ![](results/analysis/layer2/channel7/digit6.png) | ![](results/analysis/layer2/channel7/digit7.png) | ![](results/analysis/layer2/channel7/digit8.png) | ![](results/analysis/layer2/channel7/digit9.png) |
| Channel 8  | ![](results/analysis/layer2/channel8/digit0.png) | ![](results/analysis/layer2/channel8/digit1.png) | ![](results/analysis/layer2/channel8/digit2.png) | ![](results/analysis/layer2/channel8/digit3.png) | ![](results/analysis/layer2/channel8/digit4.png) | ![](results/analysis/layer2/channel8/digit5.png) | ![](results/analysis/layer2/channel8/digit6.png) | ![](results/analysis/layer2/channel8/digit7.png) | ![](results/analysis/layer2/channel8/digit8.png) | ![](results/analysis/layer2/channel8/digit9.png) |
| Channel 9  | ![](results/analysis/layer2/channel9/digit0.png) | ![](results/analysis/layer2/channel9/digit1.png) | ![](results/analysis/layer2/channel9/digit2.png) | ![](results/analysis/layer2/channel9/digit3.png) | ![](results/analysis/layer2/channel9/digit4.png) | ![](results/analysis/layer2/channel9/digit5.png) | ![](results/analysis/layer2/channel9/digit6.png) | ![](results/analysis/layer2/channel9/digit7.png) | ![](results/analysis/layer2/channel9/digit8.png) | ![](results/analysis/layer2/channel9/digit9.png) |
| Channel 10 | ![](results/analysis/layer2/channel10/digit0.png) | ![](results/analysis/layer2/channel10/digit1.png) | ![](results/analysis/layer2/channel10/digit2.png) | ![](results/analysis/layer2/channel10/digit3.png) | ![](results/analysis/layer2/channel10/digit4.png) | ![](results/analysis/layer2/channel10/digit5.png) | ![](results/analysis/layer2/channel10/digit6.png) | ![](results/analysis/layer2/channel10/digit7.png) | ![](results/analysis/layer2/channel10/digit8.png) | ![](results/analysis/layer2/channel10/digit9.png) |
| Channel 11 | ![](results/analysis/layer2/channel11/digit0.png) | ![](results/analysis/layer2/channel11/digit1.png) | ![](results/analysis/layer2/channel11/digit2.png) | ![](results/analysis/layer2/channel11/digit3.png) | ![](results/analysis/layer2/channel11/digit4.png) | ![](results/analysis/layer2/channel11/digit5.png) | ![](results/analysis/layer2/channel11/digit6.png) | ![](results/analysis/layer2/channel11/digit7.png) | ![](results/analysis/layer2/channel11/digit8.png) | ![](results/analysis/layer2/channel11/digit9.png) |
| Channel 12 | ![](results/analysis/layer2/channel12/digit0.png) | ![](results/analysis/layer2/channel12/digit1.png) | ![](results/analysis/layer2/channel12/digit2.png) | ![](results/analysis/layer2/channel12/digit3.png) | ![](results/analysis/layer2/channel12/digit4.png) | ![](results/analysis/layer2/channel12/digit5.png) | ![](results/analysis/layer2/channel12/digit6.png) | ![](results/analysis/layer2/channel12/digit7.png) | ![](results/analysis/layer2/channel12/digit8.png) | ![](results/analysis/layer2/channel12/digit9.png) |
| Channel 13 | ![](results/analysis/layer2/channel13/digit0.png) | ![](results/analysis/layer2/channel13/digit1.png) | ![](results/analysis/layer2/channel13/digit2.png) | ![](results/analysis/layer2/channel13/digit3.png) | ![](results/analysis/layer2/channel13/digit4.png) | ![](results/analysis/layer2/channel13/digit5.png) | ![](results/analysis/layer2/channel13/digit6.png) | ![](results/analysis/layer2/channel13/digit7.png) | ![](results/analysis/layer2/channel13/digit8.png) | ![](results/analysis/layer2/channel13/digit9.png) |
| Channel 14 | ![](results/analysis/layer2/channel14/digit0.png) | ![](results/analysis/layer2/channel14/digit1.png) | ![](results/analysis/layer2/channel14/digit2.png) | ![](results/analysis/layer2/channel14/digit3.png) | ![](results/analysis/layer2/channel14/digit4.png) | ![](results/analysis/layer2/channel14/digit5.png) | ![](results/analysis/layer2/channel14/digit6.png) | ![](results/analysis/layer2/channel14/digit7.png) | ![](results/analysis/layer2/channel14/digit8.png) | ![](results/analysis/layer2/channel14/digit9.png) |
| Channel 15 | ![](results/analysis/layer2/channel15/digit0.png) | ![](results/analysis/layer2/channel15/digit1.png) | ![](results/analysis/layer2/channel15/digit2.png) | ![](results/analysis/layer2/channel15/digit3.png) | ![](results/analysis/layer2/channel15/digit4.png) | ![](results/analysis/layer2/channel15/digit5.png) | ![](results/analysis/layer2/channel15/digit6.png) | ![](results/analysis/layer2/channel15/digit7.png) | ![](results/analysis/layer2/channel15/digit8.png) | ![](results/analysis/layer2/channel15/digit9.png) |

### Layer 4 (Max Pool)

A large reduction in information density while maintaining a useful feature map. The pixels of these channels go on to feed the neural network.

| Digit     | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-----------|---|---|---|---|---|---|---|---|---|---|
| Channel 0  | ![](results/analysis/layer3/channel0/digit0.png) | ![](results/analysis/layer3/channel0/digit1.png) | ![](results/analysis/layer3/channel0/digit2.png) | ![](results/analysis/layer3/channel0/digit3.png) | ![](results/analysis/layer3/channel0/digit4.png) | ![](results/analysis/layer3/channel0/digit5.png) | ![](results/analysis/layer3/channel0/digit6.png) | ![](results/analysis/layer3/channel0/digit7.png) | ![](results/analysis/layer3/channel0/digit8.png) | ![](results/analysis/layer3/channel0/digit9.png) |
| Channel 1  | ![](results/analysis/layer3/channel1/digit0.png) | ![](results/analysis/layer3/channel1/digit1.png) | ![](results/analysis/layer3/channel1/digit2.png) | ![](results/analysis/layer3/channel1/digit3.png) | ![](results/analysis/layer3/channel1/digit4.png) | ![](results/analysis/layer3/channel1/digit5.png) | ![](results/analysis/layer3/channel1/digit6.png) | ![](results/analysis/layer3/channel1/digit7.png) | ![](results/analysis/layer3/channel1/digit8.png) | ![](results/analysis/layer3/channel1/digit9.png) |
| Channel 2  | ![](results/analysis/layer3/channel2/digit0.png) | ![](results/analysis/layer3/channel2/digit1.png) | ![](results/analysis/layer3/channel2/digit2.png) | ![](results/analysis/layer3/channel2/digit3.png) | ![](results/analysis/layer3/channel2/digit4.png) | ![](results/analysis/layer3/channel2/digit5.png) | ![](results/analysis/layer3/channel2/digit6.png) | ![](results/analysis/layer3/channel2/digit7.png) | ![](results/analysis/layer3/channel2/digit8.png) | ![](results/analysis/layer3/channel2/digit9.png) |
| Channel 3  | ![](results/analysis/layer3/channel3/digit0.png) | ![](results/analysis/layer3/channel3/digit1.png) | ![](results/analysis/layer3/channel3/digit2.png) | ![](results/analysis/layer3/channel3/digit3.png) | ![](results/analysis/layer3/channel3/digit4.png) | ![](results/analysis/layer3/channel3/digit5.png) | ![](results/analysis/layer3/channel3/digit6.png) | ![](results/analysis/layer3/channel3/digit7.png) | ![](results/analysis/layer3/channel3/digit8.png) | ![](results/analysis/layer3/channel3/digit9.png) |
| Channel 4  | ![](results/analysis/layer3/channel4/digit0.png) | ![](results/analysis/layer3/channel4/digit1.png) | ![](results/analysis/layer3/channel4/digit2.png) | ![](results/analysis/layer3/channel4/digit3.png) | ![](results/analysis/layer3/channel4/digit4.png) | ![](results/analysis/layer3/channel4/digit5.png) | ![](results/analysis/layer3/channel4/digit6.png) | ![](results/analysis/layer3/channel4/digit7.png) | ![](results/analysis/layer3/channel4/digit8.png) | ![](results/analysis/layer3/channel4/digit9.png) |
| Channel 5  | ![](results/analysis/layer3/channel5/digit0.png) | ![](results/analysis/layer3/channel5/digit1.png) | ![](results/analysis/layer3/channel5/digit2.png) | ![](results/analysis/layer3/channel5/digit3.png) | ![](results/analysis/layer3/channel5/digit4.png) | ![](results/analysis/layer3/channel5/digit5.png) | ![](results/analysis/layer3/channel5/digit6.png) | ![](results/analysis/layer3/channel5/digit7.png) | ![](results/analysis/layer3/channel5/digit8.png) | ![](results/analysis/layer3/channel5/digit9.png) |
| Channel 6  | ![](results/analysis/layer3/channel6/digit0.png) | ![](results/analysis/layer3/channel6/digit1.png) | ![](results/analysis/layer3/channel6/digit2.png) | ![](results/analysis/layer3/channel6/digit3.png) | ![](results/analysis/layer3/channel6/digit4.png) | ![](results/analysis/layer3/channel6/digit5.png) | ![](results/analysis/layer3/channel6/digit6.png) | ![](results/analysis/layer3/channel6/digit7.png) | ![](results/analysis/layer3/channel6/digit8.png) | ![](results/analysis/layer3/channel6/digit9.png) |
| Channel 7  | ![](results/analysis/layer3/channel7/digit0.png) | ![](results/analysis/layer3/channel7/digit1.png) | ![](results/analysis/layer3/channel7/digit2.png) | ![](results/analysis/layer3/channel7/digit3.png) | ![](results/analysis/layer3/channel7/digit4.png) | ![](results/analysis/layer3/channel7/digit5.png) | ![](results/analysis/layer3/channel7/digit6.png) | ![](results/analysis/layer3/channel7/digit7.png) | ![](results/analysis/layer3/channel7/digit8.png) | ![](results/analysis/layer3/channel7/digit9.png) |
| Channel 8  | ![](results/analysis/layer3/channel8/digit0.png) | ![](results/analysis/layer3/channel8/digit1.png) | ![](results/analysis/layer3/channel8/digit2.png) | ![](results/analysis/layer3/channel8/digit3.png) | ![](results/analysis/layer3/channel8/digit4.png) | ![](results/analysis/layer3/channel8/digit5.png) | ![](results/analysis/layer3/channel8/digit6.png) | ![](results/analysis/layer3/channel8/digit7.png) | ![](results/analysis/layer3/channel8/digit8.png) | ![](results/analysis/layer3/channel8/digit9.png) |
| Channel 9  | ![](results/analysis/layer3/channel9/digit0.png) | ![](results/analysis/layer3/channel9/digit1.png) | ![](results/analysis/layer3/channel9/digit2.png) | ![](results/analysis/layer3/channel9/digit3.png) | ![](results/analysis/layer3/channel9/digit4.png) | ![](results/analysis/layer3/channel9/digit5.png) | ![](results/analysis/layer3/channel9/digit6.png) | ![](results/analysis/layer3/channel9/digit7.png) | ![](results/analysis/layer3/channel9/digit8.png) | ![](results/analysis/layer3/channel9/digit9.png) |
| Channel 10 | ![](results/analysis/layer3/channel10/digit0.png) | ![](results/analysis/layer3/channel10/digit1.png) | ![](results/analysis/layer3/channel10/digit2.png) | ![](results/analysis/layer3/channel10/digit3.png) | ![](results/analysis/layer3/channel10/digit4.png) | ![](results/analysis/layer3/channel10/digit5.png) | ![](results/analysis/layer3/channel10/digit6.png) | ![](results/analysis/layer3/channel10/digit7.png) | ![](results/analysis/layer3/channel10/digit8.png) | ![](results/analysis/layer3/channel10/digit9.png) |
| Channel 11 | ![](results/analysis/layer3/channel11/digit0.png) | ![](results/analysis/layer3/channel11/digit1.png) | ![](results/analysis/layer3/channel11/digit2.png) | ![](results/analysis/layer3/channel11/digit3.png) | ![](results/analysis/layer3/channel11/digit4.png) | ![](results/analysis/layer3/channel11/digit5.png) | ![](results/analysis/layer3/channel11/digit6.png) | ![](results/analysis/layer3/channel11/digit7.png) | ![](results/analysis/layer3/channel11/digit8.png) | ![](results/analysis/layer3/channel11/digit9.png) |
| Channel 12 | ![](results/analysis/layer3/channel12/digit0.png) | ![](results/analysis/layer3/channel12/digit1.png) | ![](results/analysis/layer3/channel12/digit2.png) | ![](results/analysis/layer3/channel12/digit3.png) | ![](results/analysis/layer3/channel12/digit4.png) | ![](results/analysis/layer3/channel12/digit5.png) | ![](results/analysis/layer3/channel12/digit6.png) | ![](results/analysis/layer3/channel12/digit7.png) | ![](results/analysis/layer3/channel12/digit8.png) | ![](results/analysis/layer3/channel12/digit9.png) |
| Channel 13 | ![](results/analysis/layer3/channel13/digit0.png) | ![](results/analysis/layer3/channel13/digit1.png) | ![](results/analysis/layer3/channel13/digit2.png) | ![](results/analysis/layer3/channel13/digit3.png) | ![](results/analysis/layer3/channel13/digit4.png) | ![](results/analysis/layer3/channel13/digit5.png) | ![](results/analysis/layer3/channel13/digit6.png) | ![](results/analysis/layer3/channel13/digit7.png) | ![](results/analysis/layer3/channel13/digit8.png) | ![](results/analysis/layer3/channel13/digit9.png) |
| Channel 14 | ![](results/analysis/layer3/channel14/digit0.png) | ![](results/analysis/layer3/channel14/digit1.png) | ![](results/analysis/layer3/channel14/digit2.png) | ![](results/analysis/layer3/channel14/digit3.png) | ![](results/analysis/layer3/channel14/digit4.png) | ![](results/analysis/layer3/channel14/digit5.png) | ![](results/analysis/layer3/channel14/digit6.png) | ![](results/analysis/layer3/channel14/digit7.png) | ![](results/analysis/layer3/channel14/digit8.png) | ![](results/analysis/layer3/channel14/digit9.png) |
| Channel 15 | ![](results/analysis/layer3/channel15/digit0.png) | ![](results/analysis/layer3/channel15/digit1.png) | ![](results/analysis/layer3/channel15/digit2.png) | ![](results/analysis/layer3/channel15/digit3.png) | ![](results/analysis/layer3/channel15/digit4.png) | ![](results/analysis/layer3/channel15/digit5.png) | ![](results/analysis/layer3/channel15/digit6.png) | ![](results/analysis/layer3/channel15/digit7.png) | ![](results/analysis/layer3/channel15/digit8.png) | ![](results/analysis/layer3/channel15/digit9.png) |
