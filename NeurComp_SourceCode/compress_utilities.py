""" Created: 26.10.2022  \\  Updated: 12.04.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import math
import tensorflow as tf

#==============================================================================
# Define a custom subclass to encapsulate tf.keras.metrics.Metric logic & state

# Inherits from 'tf.keras.metrics.Metric'

class MeanSquaredErrorMetric(tf.keras.metrics.Metric):
    
    # Initialise internal state variables using the 'self.add_weight()' method
    def __init__(self,name='mse_metric',**kwargs):
        
        super().__init__(name=name, **kwargs)
        self.error_sum = self.add_weight(name='error_sum',initializer='zeros')
        self.n_batches = self.add_weight(name='n_batches',initializer='zeros')
        return None

    # Define a method to update the state variables after each train minibatch 
    def update_state(self,true,pred,weights):
                
        mse = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(weights,tf.math.square(tf.math.subtract(pred,true)))),tf.reduce_sum(weights))
        self.error_sum.assign_add(mse)
        self.n_batches.assign_add(1.0)
        return None
    
    # Define a method to reset the state variables after each terminated epoch
    def reset_state(self):
        
        self.error_sum.assign(0.0)
        self.n_batches.assign(0.0)
        return None
    
    # Define a method to evaluate and return the mean squared error metric
    def result(self):
        
        mse = self.error_sum/self.n_batches
        return mse

#==============================================================================
# Define a function to perform training on batches of data within the main loop

def TrainStep(model,optimiser,metric,volume_batch,values_batch,weights_batch):
            
    # Open 'GradientTape' to record the operations run in each forward pass
    with tf.GradientTape() as tape:
        
        # Compute a forward pass on the current mini-batch
        values_predicted = model(volume_batch,training=True)
        
        # Compute the mean-squared error for the current mini-batch
        mse = MeanSquaredError(values_batch,values_predicted,weights_batch)
    ##
   
    # Determine the weight and bias gradients with respect to error
    gradients = tape.gradient(mse,model.trainable_variables)
    
    # Update the weight and bias values to minimise the total error
    optimiser.apply_gradients(zip(gradients,model.trainable_variables))
            
    # Update the training metric
    metric.update_state(values_batch,values_predicted,weights_batch)
        
    return None

#==============================================================================
# Define a function that computes the mean squared loss on predictions 

def MeanSquaredError(true,pred,weights):
        
    # Compute the weighted mean squared error between signals
    mse = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(weights,tf.math.square(tf.math.subtract(pred,true)))),tf.reduce_sum(weights))                             
    
    return mse

#==============================================================================
# Define a function to calculate the current learning rate based on epoch/decay

def GetLearningRate(initial_lr,half_life,epoch):
    
    # Decay learning rate following an exponentially decaying curve
    current_lr = initial_lr / (2**(epoch//half_life))

    return current_lr

#==============================================================================
# Define a function that computes the peak signal-to-noise ratio (PSNR) 

def SignalToNoise(true,pred,weights):
    
    # Compute the mean squared error between signals
    mse = MeanSquaredError(true,pred,weights)

    # Compute the range of the true signal
    rng = abs(true.max()-true.min())

    # Compute the peak signal-to-noise ratio
    psnr = -20.0*(math.log10(math.sqrt(mse)/rng))
    
    return psnr

#==============================================================================