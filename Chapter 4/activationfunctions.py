import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#Using class made in prev lesson
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#Creating hidden layer and using the activation function on their ouputs
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense1.forward(X)
activation1.forward(dense1.output)
print(activation1.output[:5])
