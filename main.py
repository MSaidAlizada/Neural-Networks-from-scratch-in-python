import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(samples=100, classes=3)



#Class for dense layer (fully connected layer)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#Rectified linear activation function for hidden layers
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#Softmax activation function for output layer
class Activation_Softmax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        sum_exp_inputs = np.sum(exp_inputs, axis=1, keepdims=True)
        self.output = exp_inputs / sum_exp_inputs

#Creating hidden layer and using the activation function on their ouputs
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense1.forward(X)
activation1.forward(dense1.output)

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])
