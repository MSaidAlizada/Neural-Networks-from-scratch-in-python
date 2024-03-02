import numpy as np

#Neuron code
#Without the use of numpy
inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, 0.48, 0.5, 1]
weights3 = [0.2, 0.28, -0.15, 1]
bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0]*weights1[0]+inputs[1]*weights1[1]+inputs[2]*weights1[2]+inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0]+inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0]+inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3] + bias3
         ]

#Dot product (using numpy)
#This section HAS 2 Hidden layers
#Each neuron in the layers has a unique bias but each connection has a unique weight
input1 = [1, 2, 3, 2.5]
input2 = [2, 5, -1, 2]
input3 = [-1.5, 2.7, 3.3, -0.8]
inputs = [input1, input2, input3]
weights = [[0.2, 0.8, -0.5, 1],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]
biases1 = [bias1, bias2, bias3]
layer1_output = np.dot(inputs, np.array(weights).T) + biases1
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]
layer2_output = np.dot(layer1_output, np.array(weights2).T)+biases2
print(layer2_output)