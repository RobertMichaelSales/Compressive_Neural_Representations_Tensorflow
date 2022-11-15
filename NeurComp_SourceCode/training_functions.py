""" Created: 26.10.2022  \\  Updated: 10.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import tensorflow as tf

#==============================================================================
# Define a function (as a static graph) to perform training steps on batches of
# training data within the main training loop

@tf.function
def TrainingStep(model,optimiser,metric,volume_batch,values_batch,indices_batch):
    
    # Open a 'GradientTape' to record the operations run during the forward
    # pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        
        # Compute a forward pass for the current mini-batch
        values_predicted = model(volume_batch,training=True)
        
        # Compute the losses for the current mini-batch
        loss_on_predicted = LossPred(values_batch,values_predicted)
        # loss_on_gradients = LossGrad(values_batch,values_predicted)
        total_loss = loss_on_predicted

    # Use the gradient tape to automatically retrieve the gradients for the
    # trainable variables (weights/biases) with respect to loss
    gradients = tape.gradient(total_loss,model.trainable_variables)
    
    # Run a single step of gradient descent by updating the variable values
    # in order to minimise the total loss per mini-batch
    optimiser.apply_gradients(zip(gradients,model.trainable_variables))
            
    # Update the training metric for the current mini-batch results
    metric.update_state(values_batch,values_predicted)
    
    return None


#==============================================================================
# Define a function (as a static graph) to perform training steps on batches of
# training data within the main training loop

@tf.function
def TrainStep(model,optimiser,metric,volume_batch,values_batch,indices_batch):
    
    # Open a 'GradientTape' to record the operations run during the forward
    # pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        
        # Compute a forward pass for the current mini-batch
        values_predicted = model(volume_batch,training=True)
        
        # Compute the losses for the current mini-batch
        loss_on_predicted = LossPred(values_batch,values_predicted)
        # loss_on_gradients = LossGrad(values_batch,values_predicted)
        total_loss = loss_on_predicted

    # Use the gradient tape to automatically retrieve the gradients for the
    # trainable variables (weights/biases) with respect to loss
    gradients = tape.gradient(total_loss,model.trainable_variables)
    
    # Run a single step of gradient descent by updating the variable values
    # in order to minimise the total loss per mini-batch
    optimiser.apply_gradients(zip(gradients,model.trainable_variables))
            
    # Update the training metric for the current mini-batch results
    metric.update_state(values_batch,values_predicted)
    
    return None

#==============================================================================
# Define a function to calculate the current learning rate based on epoch/decay

def LRScheduler(network_config,epoch):
    
    initial_learning_rate = network_config.initial_learning_rate
    decay_rate = network_config.decay_rate
    
    current_learning_rate = initial_learning_rate / (2**(epoch//decay_rate))

    return current_learning_rate

#==============================================================================
# Define a function that computes the mean squared loss on predictions 

@tf.function
def LossPred(true,pred):
    
    loss = tf.math.reduce_mean(tf.math.square(tf.math.subtract(pred,true)))
    
    return loss

#==============================================================================
# Define a function that computes the mean squared loss on prediction gradients 

@tf.function
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