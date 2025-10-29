""" Created: 26.10.2022  \\  Updated: 12.04.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import math, sys
import numpy as np
import tensorflow as tf

#==============================================================================
# Define MSE metric: Inherits from 'tf.keras.metrics.Metric'

class MSEMetric(tf.keras.metrics.Metric):
    
    # Initialise internal state variables using the 'self.add_weight()' method
    def __init__(self,name='mse_metric',**kwargs):
        
        super().__init__(name=name, **kwargs)
        self.error_sum = self.add_weight(name='error_sum',initializer='zeros')
        self.n_batches = self.add_weight(name='n_batches',initializer='zeros')
        return None
    ##

    # Define a method to update the state variables after each train minibatch 
    def update_state(self,true,pred,scales):
                
        mse = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(scales,tf.math.square(tf.math.subtract(pred,true)))),tf.reduce_sum(scales))
        self.error_sum.assign_add(mse)
        self.n_batches.assign_add(1.0)
        return None
    ##
    
    # Define a method to reset the state variables after each terminated epoch
    def reset_state(self):
        
        self.error_sum.assign(0.0)
        self.n_batches.assign(0.0)
        return None
    ##
    
    # Define a method to evaluate and return the mean squared error metric
    def result(self):
        
        mse = self.error_sum/self.n_batches
        return mse
    ##
##

#==============================================================================
# Define MAE error metric: Inherits from 'tf.keras.metrics.Metric'

class MAEMetric(tf.keras.metrics.Metric):
    
    # Initialise internal state variables using the 'self.add_weight()' method
    def __init__(self,name='mae_metric',**kwargs):
        
        super().__init__(name=name, **kwargs)
        self.error_sum = self.add_weight(name='error_sum',initializer='zeros')
        self.n_batches = self.add_weight(name='n_batches',initializer='zeros')
    ##

    # Define a method to update the state variables after each train minibatch 
    def update_state(self,true,pred):
        
        mae = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(pred,true)))
        self.error_sum.assign_add(mae)
        self.n_batches.assign_add(1.0)
    ##
    
    # Define a method to reset the state variables after each terminated epoch
    def reset_state(self):
        
        self.error_sum.assign(0.0)
        self.n_batches.assign(0.0)
    ##
    
    # Define a method to evaluate and return the mean squared error metric
    def result(self):
        
        mae = self.error_sum/self.n_batches
        return mae
    ##
##

    
#==============================================================================
# Define a console logger class to simultaneously log and print stdout messages

class Logger():
    
    # Initialise internal states and open log file
    def __init__(self, logfile):
        
        self.stdout = sys.stdout
        self.txtlog = open(logfile,'w')
    ##

    # Define a function to write to both stdout and txt
    def write(self, text):
        
        self.stdout.write(text)
        self.txtlog.write(text)
        self.txtlog.flush()
    ##
        
    # Define a function to flush both stdout and txt streams
    def flush(self):
        
        self.stdout.flush()
        self.txtlog.flush()
    ##

    # Define a function to close both stdout and txt streams
    def close(self):
        
        self.stdout.close()
        self.txtlog.close()
    ##
##

#==============================================================================
# Define a function to perform training on batches of data within the main loop

def TrainStep(model,optimiser,metric,coords_batch,values_batch,scales_batch):
            
    # Open 'GradientTape' to record the operations run in each forward pass
    with tf.GradientTape() as tape:
        
        # Compute a forward pass on the current mini-batch
        values_predicted_batch = model(coords_batch,training=True)
        
        # Compute the mean-squared error for the current mini-batch
        mse_batch = MeanSquaredError(values_batch,values_predicted_batch,scales_batch)
        
    ##
   
    # Determine the weight and bias gradients with respect to error
    gradients_batch = tape.gradient(mse_batch,model.trainable_variables)
    
    # Update the weight and bias values to minimise the total error
    optimiser.apply_gradients(zip(gradients_batch,model.trainable_variables))
            
    # Update the training metric
    metric.update_state(values_batch,values_predicted_batch,scales_batch)
        
    return mse_batch

##

#==============================================================================
# Define a function that computes the mean squared loss on predictions 

def MeanSquaredError(true,pred,scales):
        
    # Compute the weighted mean squared error between signals
    mse = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(scales,tf.math.square(tf.math.subtract(pred,true)))),tf.math.reduce_sum(scales))                             
    
    return mse

##

#==============================================================================
# Define a function that computes the mean squared loss on predictions 

def MeanAbsoluteError(true,pred):
        
    # Compute the weighted mean squared error between signals
    mae = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(pred,true)))                           
    
    return mae
##

#==============================================================================
# Define a function to calculate the current learning rate based on epoch/decay

def GetLearningRate(initial_lr,half_life,epoch):
    
    # Decay learning rate following an exponentially decaying curve
    current_lr = initial_lr / (2**(epoch//half_life))

    return current_lr

##

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
##

#==============================================================================

def QuantiseParameters(original_parameters,bits_per_neuron):

    # Get original parameters (ws) and extract max and min value range
    original_ws = original_parameters[0::2]
    original_ws_min = np.percentile(np.concatenate([w.flatten() for w in original_ws]),00.05)
    original_ws_max = np.percentile(np.concatenate([w.flatten() for w in original_ws]),99.95)
    
    # Get original parameters (bs) and extract max and min value range
    original_bs = original_parameters[1::2]
    original_bs_min = np.percentile(np.concatenate([w.flatten() for w in original_bs]),00.01)
    original_bs_max = np.percentile(np.concatenate([w.flatten() for w in original_bs]),99.99)
    
    # Create empty list for storing new quantised parameters
    quantised_parameters = []    
   
    # Iterate through all parameter matrices
    for original_w, original_b in zip(original_ws, original_bs):
        
        ## Weights
        
        # Clip each matrix to [original_ws_min,original_ws_max]
        clipped_w = np.clip(original_w, original_ws_min, original_ws_max)
        
        # Normalise each matrix to [0,1]
        normalised_w = (clipped_w - original_ws_min) / (original_ws_max - original_ws_min)
        
        # Round to nearest tenable value
        quantised_w  = (np.round(normalised_w*(2**bits_per_neuron - 1))) / (2**bits_per_neuron - 1)
        
        # Rescale each to original range
        rescaled_w = (quantised_w * (original_ws_max - original_ws_min)) + original_ws_min
        
        # Append to the quantised parameters list
        quantised_parameters.append(rescaled_w.astype(np.float32))
            
        ##
        
        ## Biases

        # Clip each matrix to [original_bs_min,original_bs_max]
        clipped_b = np.clip(original_b, original_bs_min, original_bs_max)
        
        # Normalise each matrix to [0,1]
        normalised_b = (clipped_b - original_bs_min) / (original_bs_max - original_bs_min)
        
        # Round to nearest tenable value
        quantised_b  = (np.round(normalised_b*(2**bits_per_neuron - 1))) / (2**bits_per_neuron - 1)
        
        # Rescale each to original range
        rescaled_b = (quantised_b * (original_bs_max - original_bs_min)) + original_bs_min
        
        # Append to the quantised parameters list
        quantised_parameters.append(rescaled_b.astype(np.float32))
            
        ##
    
    ##
        
    return quantised_parameters

##

#==============================================================================


