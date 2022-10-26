""" Created: 26.10.2022  \\  Updated: 26.10.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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