""" Created: 26.10.2022  \\  Updated: 05.01.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import math
import tensorflow as tf

#==============================================================================
# Define a function to perform training on batches of data within the main loop

@tf.function
def TrainStep(model,optimiser,metric,volume_batch,values_batch):
    
    # Open 'GradientTape' to record the operations run in each forward pass
    with tf.GradientTape() as tape:
        
        # Compute a forward pass on the current mini-batch
        values_predicted = model(volume_batch,training=True)
        
        # Compute the mean-squared error for the current mini-batch
        mse = MeanSquaredError(values_batch,values_predicted)
   
    # Determine the weight and bias gradients with respect to error
    gradients = tape.gradient(mse,model.trainable_variables)
    
    # Update the weight and bias values to minimise the total error
    optimiser.apply_gradients(zip(gradients,model.trainable_variables))
            
    # Update the training metric
    metric.update_state(values_batch,values_predicted)
    
    return None

#==============================================================================
# Define a function that computes the mean squared loss on predictions 

@tf.function
def MeanSquaredError(true,pred):
    
    # Compute the mean squared error between signals
    mse = tf.math.reduce_mean(tf.math.square(tf.math.subtract(pred,true)))
    
    return mse

#==============================================================================
# Define a function that computes the peak signal-to-noise ratio (PSNR) 

def SignalToNoise(true,pred):
    
    # Compute the mean squared error between signals
    mse = tf.math.reduce_mean(tf.math.square(tf.math.subtract(pred,true)))
    
    # Compute the range of the true signal
    rng = abs(tf.math.reduce_max(true)-tf.math.reduce_min(true))

    # Compute the peak signal-to-noise ratio
    psnr = -20.0*(math.log10(math.sqrt(mse)/rng))
    
    return psnr

#==============================================================================
# Define a function to calculate the current learning rate based on epoch/decay

def GetLearningRate(initial_lr,half_life,epoch):
    
    # Decay learning rate following an exponentially decaying curve
    current_lr = initial_lr / (2**(epoch//half_life))

    return current_lr

#==============================================================================