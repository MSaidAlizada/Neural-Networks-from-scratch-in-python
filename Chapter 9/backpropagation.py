x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

z = xw0 + xw1 + xw2 + b #Z is the output calculated of the neuron

y = max(z,0) #This is the final output after being run through the activation function ReLU

#Our equation right now is y=ReLU(sum(mul(x0,w0),mul(x1,w1),mul(x2,w2),b))
#Now to get the impact of each weight we are going to use the chain to rule to split into to multiple derivatives that we can calculate
#The first is the dReLU()/dSum() which is also in our code dReLU()/dz

dReLU_dz = (1. if z > 0 else 0.)
#This is because the ReLU uses the max function which the partial derivative is 1(x > y)

#Next we get the partial derivative of the sum with each parameter in it
#The partial derivative of the sum function is always 1 
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1

#The sum cancels out leaving just the partial derivative of the ReLU with respect to the weighted input/bias
dReLU_dxw0 = dReLU_dz * dsum_dxw0
dReLU_dxw1 = dReLU_dz * dsum_dxw1
dReLU_dxw2 = dReLU_dz * dsum_dxw2
dReLU_db = dReLU_dz * dsum_db

#Continuing backward we get the partial derivative of mul function with respect to the input
#The partial derivative of the mul function is the other thing ur multiplying so in this case the weight
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

#the mul cancels leaving:
dReLU_dx0 = dReLU_dxw0 * dmul_dx0
dReLU_dw0 = dReLU_dxw0 * dmul_dw0
dReLU_dx1 = dReLU_dxw1 * dmul_dx1
dReLU_dw1 = dReLU_dxw1 * dmul_dw1
dReLU_dx2 = dReLU_dxw2 * dmul_dx2
dReLU_dw2 = dReLU_dxw2 * dmul_dw2

dx = [dReLU_dx0, dReLU_dx1, dReLU_dx2] # gradients on inputs
dw = [dReLU_dw0, dReLU_dw1, dReLU_dw2] # gradients on weights
db = dReLU_db # gradient on bias...just 1 bias here.