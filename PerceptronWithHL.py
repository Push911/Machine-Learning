import numpy as np

out = 0
learningRate = 0.5
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
    # inputLayerHid = trainInput
    matrixHidden = np.dot(trainInput, synapticWeightsHidden)
    outHidden = sigmoid(matrixHidden)

    matrixOutput = np.dot(outHidden, synapticWeightsOutput)
    out = sigmoid(matrixOutput)

    # Back propagation of hidden layer
    errorOutput = out - trainOutput
    derivativeHidden = sigmoidDerivative(matrixOutput)
    oh = outHidden
    adjustmentHidden = np.dot(oh.T, errorOutput * derivativeHidden)
    # Back propagation of output layer
    inputLayerOut = trainInput
    erdh = errorOutput * derivativeHidden
    swo = synapticWeightsOutput
    matr = np.dot(erdh, swo.T)
    derivativeOutput = sigmoidDerivative(matrixHidden)
    adjustmentOutput = np.dot(inputLayerOut.T, derivativeOutput * matr)

    synapticWeightsOutput -= adjustmentHidden
    synapticWeightsHidden -= adjustmentOutput
    # # Feedforward
    # inputLayer = trainInput
    # hiddenOutputs = sigmoid(np.dot(inputLayer, synapticWeightsHidden))
    # outputs = sigmoid(np.dot(hiddenOutputs, synapticWeightsOutput))
    # # Back propagation of hidden layer
    # errorHidden = trainOutput - outputs
    # adjustmentHidden = np.dot(hiddenOutputs.T, errorHidden * sigmoidDerivative(outputs))
    # # Back propagation of output layer
    # error = trainOutput - outputs
    # adjustment = np.dot(inputLayer.T, error * sigmoidDerivative(outputs))
    # synapticWeightsOutput += adjustment

print("Weights:", synapticWeightsOutput)
print("Output:", out)

# Test new situation
# nums = []
# inp = int(input("Please enter size of array: "))
# for _ in range(inp):
#     num = int(input("Please enter number: "))
#     nums.append(num)

# newInput = np.array(nums)
# newInput = np.array([1, 1, 0])
# output = sigmoid(np.dot(newInput, synapticWeightsOutput))
#
# print("New situation with same weights:", output)
