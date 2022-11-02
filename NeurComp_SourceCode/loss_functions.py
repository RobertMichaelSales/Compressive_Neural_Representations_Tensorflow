""" Created: 26.10.2022  \\  Updated: 02.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import tensorflow as tf

#==============================================================================
# Define a function that computes the mean squared loss on predictions 

def LossPred(true,pred):
    
    loss = tf.math.reduce_mean(tf.math.square(tf.math.subtract(pred,true)))
    
    return loss

#==============================================================================
# Define a function that computes the mean squared loss on prediction gradients 
        
def LossGrad(true,pred):
    
    loss = tf.math.reduce_mean(tf.math.square(tf.math.subtract(pred,true)))
    
    return loss

#==============================================================================
# Define a function that computes the peak signal-to-noise ratio (PSNR) 
        
def LossPSNR(true,pred):
    
    # Compute the mean squared error between signals
    mse = tf.math.reduce_mean(tf.math.square(tf.math.subtract(pred,true)))
    
    # Compute the range of the true signal
    rng = abs(tf.math.reduce_max(true)-tf.math.reduce_min(true))

    # Compute the peak signal-to-noise ratio
    psnr = -20.0*(math.log10(math.sqrt(mse)/rng))
    
    return psnr

#==============================================================================