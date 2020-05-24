import numpy as np

predictedOutput = 0
np.random.seed(1)

trainInput = np.array([[0, 0, 1],
                       [1, 1, 1],
                       [1, 0, 1],
                       [0, 1, 1]])

trainOutput = np.array([[0, 1, 1, 0]]).T

synapticWeightsHidden = 2 * np.random.rand(len(trainInput[0]), 4) - 1
synapticWeightsOutput = 2 * np.random.rand(4, 1) - 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return x * (x > 0)


def reluDerivative(x):
    return 1. * (x > 0)


for i in range(2000):
    # Feedforward
    hiddenLayerWeights = np.dot(trainInput, synapticWeightsHidden)
    predictedHiddenOutput = sigmoid(hiddenLayerWeights)

    outputLayerWeights = np.dot(predictedHiddenOutput, synapticWeightsOutput)
    predictedOutput = sigmoid(outputLayerWeights)

    # Back propagation of hidden-output layer
    errorOutput = predictedOutput - trainOutput
    derivativeHidden = reluDerivative(outputLayerWeights)
    pho = predictedHiddenOutput
    adjustmentHidden = np.dot(pho.T, errorOutput * derivativeHidden)
    # Back propagation of input-hidden layer
    inputLayerOut = trainInput
    erdh = errorOutput * derivativeHidden
    swo = synapticWeightsOutput
    hiddenLayerError = np.dot(erdh, swo.T)
    derivativeOutput = reluDerivative(hiddenLayerWeights)
    adjustmentOutput = np.dot(inputLayerOut.T, derivativeOutput * hiddenLayerError)

    synapticWeightsOutput -= adjustmentHidden
    synapticWeightsHidden -= adjustmentOutput

print("Weights:", synapticWeightsOutput)
print("Output:", predictedOutput)

newInput = np.array([1, 1, 0]), np.array([0, 1, 1])
for newInput in newInput:
    hidLayer = sigmoid(np.dot(newInput, synapticWeightsHidden))
    output = sigmoid(np.dot(hidLayer, synapticWeightsOutput))
    print("New situation {}, with same weights: ".format(newInput), output)
