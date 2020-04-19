import numpy as np

outputs = 0

np.random.seed(1)

trainInput = np.array([[0, 0, 1],
                       [1, 1, 1],
                       [1, 0, 1],
                       [0, 1, 1]])

trainOutput = np.array([[0, 1, 1, 0]]).T

synapticWeights = 2 * np.random.random((3, 1)) - 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for i in range(20000):
    inputLayer = trainInput
    outputs = sigmoid(np.dot(inputLayer, synapticWeights))
    err = trainOutput - outputs

    adjustment = np.dot(inputLayer.T, err * (outputs * (1 - outputs)))
    synapticWeights += adjustment


print("Weights:", synapticWeights)
print("Output:", outputs)

# Test new situation
newInput = np.array([1, 1, 0])
output = sigmoid(np.dot(newInput, synapticWeights))

print("New situation with same weights:", output)
