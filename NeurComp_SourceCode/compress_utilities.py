""" Created: 26.10.2022  \\  Updated: 24.03.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import math
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
from keras.initializers import glorot_uniform

#==============================================================================

class MeanSquaredErrorMetric(tf.keras.metrics.Metric):
    
    def __init__(self,name='mse_metric',**kwargs):
        
        super().__init__(name=name, **kwargs)
        self.error_sum = self.add_weight(name='error_sum',initializer='zeros')
        self.batch_num = self.add_weight(name='batch_num',initializer='zeros')
        return None

    def update_state(self,true,pred,weights):
        
        mse = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(weights,tf.math.square(tf.math.subtract(pred,true)))),tf.reduce_sum(weights))
        self.error_sum.assign_add(mse)
        self.batch_num.assign_add(1.0)
        return None
    
    def reset_state(self):
        
        self.error_sum.assign(0.0)
        self.batch_num.assign(0.0)
        return None
    
    def result(self):
        
        mse = self.error_sum/self.batch_num
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

@tf.function
def MeanSquaredError(true,pred,weights):
        
    # Compute the weighted mean squared error between signals
    mse = tf.math.divide(tf.math.reduce_mean(tf.math.multiply(weights,tf.math.square(tf.math.subtract(pred,true)))),tf.reduce_sum(weights))                             
    
    return mse

#==============================================================================
# Define a function to calculate the current learning rate based on epoch/decay

def GetLearningRate(initial_lr,half_life,epoch):
    
    # Decay learning rate following an exponentially decaying curve
    current_lr = initial_lr / (2**(epoch//half_life))

    return current_lr

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
# Define a function to calculate the standard deviation of points in 1-D

def CalculateStandardDeviation(points):

    # Compute standard deviation ignoring the dimensionality
    standard_deviation = np.std(points,axis=None)
    
    return standard_deviation

#==============================================================================
# Define a function to calculate the pointcloud density of points in N-D    

def CalculatePointCloudDensity(points):
    
    # Extract the number of dimensions associated with the point cloud
    dimensions = points.shape[-1]
    
    # Instantiate a KDTree object on the pointcloud data
    kd_tree = KDTree(points,leaf_size=20)
    
    # Query the KDTree for the k-nearest neighbours data
    distances,neighbours = kd_tree.query(points,k=2**dimensions)
    
    # Compute the radius of the furthest k-nearest point
    max_distances = np.amax(distances,axis=-1)
    
    # Compute the density using circle, sphere or 4-ball volume
    if dimensions == 2:
        
        # / (pi * r^2)
        density = np.average(1/(np.pi*(max_distances**2)))

    elif dimensions == 3:
        
        # / (4/3 * pi * r^3) 
        density = np.average(3/(4*np.pi*(max_distances**3)))
        
    elif dimensions == 4:
        
        # / (1/2 * pi^2 * r^4)
        density = np.average(2/((np.pi**2)*(max_distances**4)))
        
    else: return None
    
    return density

#==============================================================================
# Define a function to perform/plot an initial learning rate optimisation study

def LRStudy(model,volume,values,weights,bf_guess,lr_bounds,plot):
    
    # Import the 'MakeDataset' function from 'data_management.py'
    from data_management import MakeDataset
    TrainStepTFF = tf.function(TrainStep)
    
    # Guess a usable batch size and create a temporary TF dataset 
    batch_size = math.floor(bf_guess*values.size)
    dataset = MakeDataset(volume=volume,values=values,weights=weights,batch_size=batch_size)
    
    # Create a temportary local training optimiser
    optimiser = tf.keras.optimizers.Adam()
    
    # Create a exponentially (regularly) increasing array of learning rates
    lr_lspace = 10.0 ** np.linspace(lr_bounds[0],lr_bounds[1],25)
    
    # Set a performance metric
    lr_metric = MeanSquaredErrorMetric()
    
    # Create an empty list to store the first-pass training errors
    lr_errors = []
    
    # Clone the original SquashNet model for this study
    ModelClone = tf.keras.models.clone_model(model=model)
    
    # Initialise the weights of the model using a Glorot initialisation scheme
    initial_weights = [glorot_uniform()(w.shape) for w in ModelClone.get_weights()]
    
    # Iterate through the learning rates (including zero) for computing deltas
    for learning_rate in np.append([0],lr_lspace):
        
        print("\n{:30}{:.3e}".format("Current Learning Rate:",learning_rate))
        
        # Restore the initial weights for each subsequent pass
        ModelClone.set_weights(initial_weights)
        
        # Reset/release all states generated by Keras
        tf.keras.backend.clear_session()
        
        # Reset the performance metric
        lr_metric.reset_state()
        
        # Update the learning rate
        optimiser.lr.assign(learning_rate)  
        
        # Iterate through each batch
        for batch, (volume_batch,values_batch,weights_batch) in enumerate(dataset):
                        
            TrainStepTFF(model=ModelClone,optimiser=optimiser,metric=lr_metric,volume_batch=volume_batch,values_batch=values_batch,weights_batch=weights_batch)
            
            if batch > len(dataset)/8: break
        ##
        
        # Fetch and store the performance metric
        lr_errors.append(lr_metric.result().numpy())
        
        print("{:30}{:.3f}".format("Mean-Squared Error:",lr_errors[-1]))
        
    ##
    
    # Compute the deltas relative to zero learning rate
    lr_deltas = lr_errors[1:] - lr_errors[0]
    
    # Convert 'NaN' values to equal the zero-rate error
    lr_deltas = np.nan_to_num(lr_deltas,nan=np.nanmax(lr_deltas))
    
    # Check if plot flag is raised
    if plot:
                
        # Plot the results using MatplotLib
        plt.style.use("Auxiliary_Scripts/plot.mplstyle")  
        params_plot = {'text.latex.preamble': [r'\usepackage{amsmath}',r'\usepackage{amssymb}'],'axes.grid': True}
        matplotlib.rcParams.update(params_plot)
                
        lr_argmin = lr_deltas.argmin()
        
        # Plot 'lr_deltas' versus 'lr_space'
        fig, ax = plt.subplots(1,1,constrained_layout=True)
        ax.plot(lr_lspace[:lr_argmin+1],lr_deltas[:lr_argmin+1],color="b",marker="o",fillstyle="full",markerfacecolor="w",zorder=2)
        ax.plot(lr_lspace[lr_argmin+0:],lr_deltas[lr_argmin+0:],color="b",marker="o",fillstyle="full",markerfacecolor="r",zorder=1)
        ax.set_xscale('log') 
        ax.set_title(r"Learning Rate Study: Initial Learning Rate Vs. Initial Loss Recovery")
        ax.set_xlabel(r"Initial Learning Rate")
        ax.set_ylabel(r"Initial Loss Recovery ($\Delta MSE$)")
        plt.show()
        
        # Plot the gradient of 'lr_deltas' versus 'lr_space'
        fig, ax = plt.subplots(1,1,constrained_layout=True)
        ax.plot(lr_lspace[:lr_argmin+1],np.gradient(lr_deltas)[:lr_argmin+1],color="b",marker="o",fillstyle="full",markerfacecolor="w",zorder=2)
        ax.plot(lr_lspace[lr_argmin+0:],np.gradient(lr_deltas)[lr_argmin+0:],color="b",marker="o",fillstyle="full",markerfacecolor="r",zorder=1)
        ax.set_xscale('log')
        ax.set_title(r"Learning Rate Study: Initial Learning Rate Vs. Initial Loss Recovery")
        ax.set_xlabel(r"Initial Learning Rate")
        ax.set_ylabel(r"Gradient of Initial Loss Recovery ($\delta(\Delta MSE)$)")
        plt.show()

    else: pass

    print("\n{:30}{:.3e}".format("Maximum Stable Learning Rate:",lr_lspace[lr_argmin]))
    
    return lr_lspace[lr_argmin]


#==============================================================================
# Define a function to perform/plot a batch fraction optimisation study

def BFStudy(model,volume,values,weights,lr_guess,bf_bounds,plot):
    
    # Import the 'MakeDataset' function from 'data_management.py'
    from data_management import MakeDataset
    TrainStepTFF = tf.function(TrainStep)
    
    # Create a temportary optimiser and specify the learning rate
    optimiser = tf.keras.optimizers.Adam()
    optimiser.lr.assign(lr_guess)
    
    # Create a exponentially (regularly) increasing array of learning rates
    bf_lspace = np.linspace(bf_bounds[0],bf_bounds[1],25)
    
    # Set a performance metric
    bf_metric = MeanSquaredErrorMetric()
    
    # Create an empty list to store the first-pass training errors and time
    bf_errors = []
    bf_actual = []
    bf_times  = []
    
    # Clone the original SquashNet model for this study
    ModelClone = tf.keras.models.clone_model(model=model)
    
    # Initialise the weights of the model using a Glorot initialisation scheme
    initial_weights = [glorot_uniform()(w.shape) for w in ModelClone.get_weights()]
    
    # Iterate through the learning rates (including zero) for computing deltas
    for batch_fraction in bf_lspace:
        
        # Calculate the nearest batch size and the actual batch fraction
        batch_size = math.floor(batch_fraction*values.size)
        bf_actual.append(batch_size/values.size)
        
        print("\n{:30}{:.3e}".format("Current Batch Fraction:",batch_fraction),end="")
        
        # Restore the initial weights for each subsequent pass
        ModelClone.set_weights(initial_weights)
        
        # Reset/release all states generated by Keras
        tf.keras.backend.clear_session()
        
        # Reset the performance metric
        bf_metric.reset_state()
        
        # Guess a learning rate and create a temporary TF dataset 
        dataset = MakeDataset(volume=volume,values=values,weights=weights,batch_size=batch_size)
        
        # Start timing
        init_time_tick = time.time()
        
        # Iterate through each batch
        for batch, (volume_batch,values_batch,weights_batch) in enumerate(dataset):
            
            if batch ==1: 
                init_time_tock = time.time()
                time_time_tick = time.time()
            else: pass
                      
            TrainStepTFF(model=ModelClone,optimiser=optimiser,metric=bf_metric,volume_batch=volume_batch,values_batch=values_batch,weights_batch=weights_batch)
            
            if batch > len(dataset)/8: break
        ##
        
        # Stop timing and append the time
        time_time_tock = time.time()
        bf_times.append(((init_time_tock-init_time_tick)+8*(time_time_tock-time_time_tick)))
        
        # Fetch and store the performance metric
        bf_errors.append(bf_metric.result().numpy())
        
        print("{:30}{:.3f}".format("Mean-Squared Error:",bf_errors[-1]))
        print("{:30}{:.3f}".format("Elapsed Time/Batch:",bf_times[-1]) )
        
        # import gc
        # dataset_length = len(dataset)
        # tf.keras.backend.clear_session()
        # gc.collect()
        # del dataset
        
    ##
    
    bf_actual = np.array(bf_actual)
    bf_errors = np.array(bf_errors)
    bf_times = np.array(bf_times) / len(dataset)
    # bf_times = np.array(bf_times) / dataset_length

    
    # Check if plot flag is raised
    if plot:
                
        # Plot the results using MatplotLib
        plt.style.use("Auxiliary_Scripts/plot.mplstyle")  
        params_plot = {'text.latex.preamble': [r'\usepackage{amsmath}',r'\usepackage{amssymb}'],'axes.grid': True}
        matplotlib.rcParams.update(params_plot)
                        
        # Plot 'bf_errors' and 'bf_times' versus 'bf_actual'
        fig, ax1 = plt.subplots(1,1,constrained_layout=True)
        ax2 = ax1.twinx()
        ax1.plot(bf_actual,bf_errors,color="b",marker="o",fillstyle="full",markerfacecolor="w",zorder=1)
        ax2.plot(bf_actual,bf_times ,color="r",marker="o",fillstyle="full",markerfacecolor="w",zorder=2)
        ax1.set_xscale('linear') 
        ax1.set_title(r"Batch Fraction Study: B'Frac Vs. Init' Loss Recovery Vs. Time")
        ax1.set_xlabel(r"Batch Fraction")
        ax1.set_ylabel(r"Initial Loss Recovery ($\Delta MSE$)",color="b")
        ax2.set_ylabel(r"Per-Batch Training TIme [s]",color="r")
        plt.show()
      
    else: pass
    
    return bf_actual,bf_errors,bf_times

#==============================================================================