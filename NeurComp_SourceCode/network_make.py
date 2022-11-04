""" Created: 18.07.2022  \\  Updated: 26.10.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

#==============================================================================
# Define a 'Sine Layer Block' using TensorFlow's functional API and Keras

def SineLayer(inputs,units,scale,name):
    
    x = tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros",name=name+"_dense")(inputs)
    
    x = tf.math.sin(tf.math.multiply(x,scale))
    
    return x
    
#==============================================================================
# Define a 'Residual Block' using TensorFlow's functional API and Keras

def SineBlock(inputs,units,scale,name,avg_1=False,avg_2=False):

    weight_1 = tf.constant(0.5) if avg_1 else tf.constant(1.0)  # Weight 1
    weight_2 = tf.constant(0.5) if avg_2 else tf.constant(1.0)  # Weight 2
        
    sine_1 = tf.math.sin(tf.math.multiply(tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros",name=name+"_dense_a")(inputs*weight_1),scale))
    sine_2 = tf.math.sin(tf.math.multiply(tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros",name=name+"_dense_b")(sine_1)         ,scale))
    
    x = tf.math.add(tf.math.multiply(inputs,weight_2),sine_2)

    return x

#==============================================================================
# Define a function that constructs the 'SIREN' network from a specific network
# configuration using the TensorFlow functional API and Keras (called NeurComp)

def BuildNeurComp(network_config):
    
    print("Constructing Network Model: {}".format(network_config.network_name))
    
    scale = tf.constant(1.0)

    # Iterate through each layer in the 'SIREN' network
    for layer in np.arange(network_config.total_layers):
          
        # Add the input layer and the first sine layer
        if (layer == 0):                  
          
            name = "l0_input"
            input_layer = tf.keras.layers.Input(shape=(network_config.layer_dimensions[layer],),name=name)
            
            name = "l0_sinelayer"
            x = SineLayer(inputs=input_layer,units=network_config.layer_dimensions[layer+1],scale=scale,name=name)
          
        # Add the final dense output layer
        elif (layer == network_config.total_layers - 1):
          
            name = "l{}_output".format(layer)
            
            output_layer =  tf.keras.layers.Dense(units=network_config.layer_dimensions[layer],name=name)(x)
          
        # Add intermediate residual block layers
        else:
          
            name = "l{}_sineblock".format(layer)
            
            avg_1 = (layer > 1)
            avg_2 = (layer == (network_config.total_layers - 2))                    # this used to be 3
            
            x = SineBlock(inputs=x,units=network_config.layer_dimensions[layer],scale=scale,name=name,avg_1=avg_1,avg_2=avg_2)
    
    # Declare the network model
    NeurComp = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    
    return NeurComp

#=============================================================================