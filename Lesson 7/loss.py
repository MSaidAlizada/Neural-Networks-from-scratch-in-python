import math 

#cross-entropy loss
softmax_output = [0.7, 0.1, 0.2]
target_output = [1,0,0]
loss = -(math.log(softmax_output[0]*target_output[0]))
print(loss)

#EXPLAINING THE CODE IN THE MAIN.PY
import numpy as np
#this class is being created so when we have multiple loss functions we want them all to have this method so they all inherit it from this
class Loss: 
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses) #getting a mean of the losses for the sample
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)#gets the number of samples
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #this is clipping the values to prevent it from being zero so that you dont get the error from logging zero
        #IF STATEMENT ARE BEING USED TO CHECK IF THE TRUE PROBABILITY IS HOT ENCODED OR IF IT IS A SCALAR
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true] #this grabs all the confidences for the correct prediction from each sample
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum((y_pred_clipped*y_true), axis=1) #Sums all of the confidences by the true probability for each sample and will remove the false ones as you are multiplying by zero
        negative_log_likelihoods = -np.log(correct_confidences) #the loss function :)
        return negative_log_likelihoods