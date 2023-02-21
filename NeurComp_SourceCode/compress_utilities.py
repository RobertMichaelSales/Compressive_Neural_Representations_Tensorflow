""" Created: 26.10.2022  \\  Updated: 21.02.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import math
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from keras.initializers import glorot_uniform

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
        
    ##
   
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

def LearningRateStudy(model,optimiser,dataset,lr_bounds,plot):
    

    lr_lspace = 10.0 ** np.linspace(lr_bounds[0],lr_bounds[1],25)
    
    lr_metric = tf.keras.metrics.MeanSquaredError()
    
    lr_errors = []
    
    ModelClone = tf.keras.models.clone_model(model=model)
    
    initial_weights = [glorot_uniform()(w.shape) for w in ModelClone.get_weights()]
    
    
    for learning_rate in np.append([0],lr_lspace):
        
        print("{:30}{}".format("Current Learning Rate:",learning_rate))
        
        ModelClone.set_weights(initial_weights)
        
        tf.keras.backend.clear_session()
        
        lr_metric.reset_state()
        
        optimiser.lr.assign(learning_rate)  
        

        for batch, (volume_batch,values_batch) in enumerate(dataset):
                        
            TrainStep(model=ModelClone,optimiser=optimiser,metric=lr_metric,volume_batch=volume_batch,values_batch=values_batch)
        ##
        
        lr_errors.append(lr_metric.result().numpy())
        
        print("{:30}{}".format("Mean-Squared Error:",lr_errors[-1]))
        
    ##
    
    lr_deltas = lr_errors[1:] - lr_errors[0]
    
    if plot:
        
        plt.style.use("Auxiliary_Scripts/plot.mplstyle")  
        
        params_plot = {'text.latex.preamble': [r'\usepackage{amsmath}',r'\usepackage{amssymb}'],
                       'axes.grid': True}
        
        matplotlib.rcParams.update(params_plot)
        
        fig, ax = plt.subplots(1,1,constrained_layout=True)
        
        ax.plot(lr_lspace,lr_deltas)
        
        ax.set_xscale('log')
                
        ax.set_title(r"Learning Rate Study: Initial Learning Rate Vs. Initial Loss Recovery")
        ax.set_xlabel(r"Initial Learning Rate")
        ax.set_ylabel(r"Initial Loss Recovery ($\Delta MSE$)")
        
        plt.show()
        
    else: pass
    
    return lr_lspace,lr_deltas  

#==============================================================================