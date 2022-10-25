""" Created: 18.07.2022  \\  Updated: 25.10.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from tensorflow import keras 
from tensorflow.keras import layers

#==============================================================================
# Define a 'Sine Layer Block' using TensorFlow's functional API and Keras

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
    
#==============================================================================
# Define a 'Residual Block' using TensorFlow's functional API and Keras

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

#==============================================================================
# Define a function that constructs the 'SIREN' network from a specific network
# configuration using the TensorFlow functional API and Keras (called NeurComp)

def BuildNeurComp(network_config):
    
    print("Building Network: '{}'\n".format(network_config.network_save_name))

    # Iterate through each layer in the 'SIREN' network
    for layer in np.arange(network_config.total_layers):
    
        # Obtain the input dimensions for that particular layer
        units = network_config.layer_dimensions[layer]
          
        # Add the input layer
        if (layer == 0):                  
          
            name = "l0_input"
            input_layer = tf.keras.layers.Input(shape=(units,),name=name)
            
            name = "l0_sineblock"
            x = SineLayerBlock(input_layer,network_config.layer_dimensions[layer+1],name=name)
          
        # Add the output layer
        elif (layer == network_config.total_layers - 1):
          
            name = "l{}_output".format(layer)
            
            output_layer =  tf.keras.layers.Dense(units=units,name=name)(x)
          
        # Add residual block layers
        else:
          
            name = "l{}_res".format(layer)
            
            avg_1 = (layer > 1)
            avg_2 = (layer == (network_config.total_layers - 3))
            
            x = ResidualBlock(x,units,name,avg_1=avg_1,avg_2=avg_2)
    
    # Declare the network model
    NeurComp = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    
    return NeurComp

#=============================================================================