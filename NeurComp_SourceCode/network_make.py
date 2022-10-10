""" Created: 18.07.2022  \\  Updated: 18.07.2022  \\   Author: Robert Sales """

#=# IMPORT LIBRARIES #========================================================#

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from tensorflow import keras 
from tensorflow.keras import layers

#=# DEFINE FUNCTIONS #========================================================#

def SineLayerBlock(inputs,units,name):

    name_dense = name + "_dense"
    
    x = tf.keras.layers.Dense(units=units,
                              activation=None,
                              use_bias=True,
                              kernel_initializer="glorot_uniform",
                              bias_initializer="zeros",
                              name=name_dense)(inputs)
    
    name_sine = name + "_sine"
    
    x = tf.math.sin(x,name=name_sine)
    
    return x
    

def ResidualBlock(inputs,units,name,avg_1=False,avg_2=False):

    w1 = 0.5 if avg_1 else 1.0  # Weight 1
    w2 = 0.5 if avg_2 else 1.0  # Weight 2
    
    name_sine_1 = name + "_sineblock_a"
    name_sine_2 = name + "_sineblock_b"
    
    sine_1 = SineLayerBlock(inputs*w1,units,name=name_sine_1)
    sine_2 = SineLayerBlock(sine_1   ,units,name=name_sine_2)
    
    name_add = name + "_add"
    name_mul = name + "_mul"
    
    x = tf.math.multiply((tf.math.add(inputs,sine_2,name=name_add)),w2,name=name_mul)
    
    return x


def BuildNeurComp(hyperparameters):
    
    print("Building Network: '{}'\n".format(hyperparameters.save_name))

    # Iterate through each layer in the network
    for layer in np.arange(hyperparameters.total_num_layers):
    
        # Obtain the input dimensions for that particular layer
        units = hyperparameters.layer_dimensions[layer]
          
        # Add the input layers
        if (layer == 0):                  
          
            name = "l0_input"
            
            input = tf.keras.layers.Input(shape=(units,),name=name)
            
            name = "l0_sineblock"
            
            x = SineLayerBlock(input,hyperparameters.layer_dimensions[layer+1],name=name)
          
        # Add the output layer
        elif (layer == hyperparameters.total_num_layers - 1):
          
            name = "l{}_output".format(layer)
            
            output =  tf.keras.layers.Dense(units=units,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer="glorot_uniform",
                                            bias_initializer="zeros",
                                            name=name)(x)
          
        # Add residual block layers
        else:
          
            name = "l{}_res".format(layer)
            
            avg_1 = (layer > 1)
            avg_2 = (layer == (hyperparameters.total_num_layers - 3))
            
            x = ResidualBlock(x,units,name,avg_1=avg_1,avg_2=avg_2)
    
    # Declare the network model
    NeurComp = tf.keras.Model(inputs=input,outputs=output)
    
    return NeurComp


def ComputeTotalParameters(hyperparameters,neurons_per_layer):
 
    # Determine the number of inter-layer operations
    num_of_operations = hyperparameters.total_num_layers     
    
    # [input -> dense] + [dense / residual -> residual] + [residual -> output]                        
      
    # Set the total number of network parameters to zero
    num_of_parameters = 0                                                         
      
    # Iterate through each of the temporary layers
    for layer in range(0,num_of_operations):
      
        if (layer==0):                             # [input -> dense]
    
            # Determine the input and output dimensions of each layer
            dim_input  = hyperparameters.input_dimension              
            dim_output = neurons_per_layer
              
            # Add parameters from the weight matrix and bias vector
            num_of_parameters += (dim_input * dim_output) + dim_output
      
        elif (layer==num_of_operations-1):     # [residual -> output]
      
            # Determine the input and output dimensions of each layer
            dim_input  = neurons_per_layer
            dim_output = hyperparameters.yield_dimension
              
            # Add parameters from the weight matrix and bias vector
            num_of_parameters += (dim_input * dim_output) + dim_output 
      
        else:                         # [dense / residual -> residual]
      
            # Add parameters from the weight matrix and bias vector
            num_of_parameters += (neurons_per_layer * neurons_per_layer) + neurons_per_layer
            num_of_parameters += (neurons_per_layer * neurons_per_layer) + neurons_per_layer        
              
    return num_of_parameters


def ComputeNeuronsPerLayer(hyperparameters):
  
    # Set the minimum neurons per layer
    neurons_per_layer = hyperparameters.min_neurons_per_layer
      
    # Keep adding neurons until the network size exceeds the target size
    while (ComputeTotalParameters(hyperparameters,neurons_per_layer) < hyperparameters.target_size):
        neurons_per_layer = neurons_per_layer + 1
      
    # Return the first neuron count to exceed the compression target
    neurons_per_layer = neurons_per_layer - 1
    
    return neurons_per_layer


#=# DEFINE CLASSES #==========================================================#


#=============================================================================#