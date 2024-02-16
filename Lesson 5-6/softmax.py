import numpy as np

#Softmax activation function in python
layer_outputs = [4.8, 1.21, 2.385]

exp_vals = np.exp(layer_outputs) #exponentiating all values
sum_exp = np.sum(exp_vals) #getting the sum of exp values
output = exp_vals / sum_exp #making the outputs into a distribution of probabilities
sum_output = np.sum(output)
print(output)
print(sum_output) #All probs added
print('-------------')
#Softmax activation function for batches and implementing max substraction to prevent overflox error
layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])

exp_values = np.exp(layer_outputs - np.max(layer_outputs, axis=1, keepdims=True))
sum_exp = np.sum(exp_values, axis=1, keepdims=True)
output = exp_values / sum_exp
print(output)