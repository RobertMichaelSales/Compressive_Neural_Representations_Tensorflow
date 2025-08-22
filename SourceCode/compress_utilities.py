""" Created: 26.10.2022  \\  Updated: 12.04.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import math, sys
import numpy as np
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
    def update_state(self,true,pred,scales):
                
        mse = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(scales,tf.math.square(tf.math.subtract(pred,true)))),tf.reduce_sum(scales))
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
# Define a console logger class to simultaneously log and print stdout messages

class Logger():
    
    # Initialise internal states and open log file
    def __init__(self, logfile):
        
        self.stdout = sys.stdout
        self.txtlog = open(logfile,'w')

    # Define a function to write to both stdout and txt
    def write(self, text):
        
        self.stdout.write(text)
        self.txtlog.write(text)
        self.txtlog.flush()
        
    # Define a function to flush both stdout and txt streams
    def flush(self):
        
        self.stdout.flush()
        self.txtlog.flush()

    # Define a function to close both stdout and txt streams
    def close(self):
        
        self.stdout.close()
        self.txtlog.close()
        

#==============================================================================
# Define a function to perform training on batches of data within the main loop

def TrainStep(model,optimiser,metric,coords_batch,values_batch,scales_batch):
            
    # Open 'GradientTape' to record the operations run in each forward pass
    with tf.GradientTape() as tape:
        
        # Compute a forward pass on the current mini-batch
        values_predicted = model(coords_batch,training=True)
        
        # Compute the mean-squared error for the current mini-batch
        mse = MeanSquaredError(values_batch,values_predicted,scales_batch)
    ##
   
    # Determine the weight and bias gradients with respect to error
    gradients = tape.gradient(mse,model.trainable_variables)
    
    # Update the weight and bias values to minimise the total error
    optimiser.apply_gradients(zip(gradients,model.trainable_variables))
            
    # Update the training metric
    metric.update_state(values_batch,values_predicted,scales_batch)
        
    return None

#==============================================================================
# Define a function that computes the mean squared loss on predictions 

def MeanSquaredError(true,pred,scales):
        
    # Compute the weighted mean squared error between signals
    mse = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(scales,tf.math.square(tf.math.subtract(pred,true)))),tf.math.reduce_sum(scales))                             
    
    return mse

#==============================================================================
# Define a function to calculate the current learning rate based on epoch/decay

def GetLearningRate(initial_lr,half_life,epoch):
    
    # Decay learning rate following an exponentially decaying curve
    current_lr = initial_lr / (2**(epoch//half_life))

    return current_lr

#==============================================================================
# Define a function that computes the peak signal-to-noise ratio (PSNR) 

def SignalToNoise(true,pred,scales):
    
    # Compute the mean squared error between signals
    mse = MeanSquaredError(true,pred,scales)

    # Compute the range of the true signal
    rng = abs(true.max()-true.min())

    # Compute the peak signal-to-noise ratio
    psnr = -20.0*(np.log10(np.sqrt(mse)/rng))
    
    return psnr

#==============================================================================

def QuantiseParameters(original_weights,bits_per_neuron):

    quantised_parameters = []    
    
    # Find the min/max values of weights
    min_w = min([x.min() for x in original_weights if x.ndim == 2])
    max_w = min([x.max() for x in original_weights if x.ndim == 2])
    
    # Find the min/max values of biases
    min_b = min([x.min() for x in original_weights if x.ndim == 1])
    max_b = min([x.max() for x in original_weights if x.ndim == 1])

    # Iterate through all parameter matrices
    for index,original_parameter in enumerate(original_weights):
        
        # Weights
        if original_parameter.ndim == 2:
        
            # Normalise each matrix to [0,1]
            normalised_parameter = (original_parameter - min_w) / (max_w - min_w)
            
            # Round to nearest tenable value
            quantised_parameter  = (np.round(normalised_parameter*(2**bits_per_neuron))) / (2**bits_per_neuron)
            
            # Rescale each to original range
            rescaled_parameter   = (quantised_parameter * (max_w - min_w)) + min_w
            
            # Append to the new_weights list
            quantised_parameters.append(rescaled_parameter.astype(np.float32))
            
        ##
        
        # Biases
        if original_parameter.ndim == 1:
            
            # Normalise each matrix to [0,1]
            normalised_parameter = (original_parameter - min_b) / (max_b - min_b)
            
            # Round to nearest tenable value
            quantised_parameter  = (np.round(normalised_parameter*(2**bits_per_neuron))) / (2**bits_per_neuron)
            
            # Rescale each to original range
            rescaled_parameter   = (quantised_parameter * (max_b - min_b)) + min_b
            
            # Append to the new_weights list
            quantised_parameters.append(rescaled_parameter.astype(np.float32))
            
        ##
    
    ##
        
    return quantised_parameters

#==============================================================================


