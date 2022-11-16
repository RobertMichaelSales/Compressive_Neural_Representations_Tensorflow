""" Created: 18.07.2022  \\  Updated: 10.11.2022  \\   Author: Robert Sales """

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
# layer arrangement using the TensorFlow functional API and Keras

def BuildNeurComp(layer_dimensions):
    
    # Set a scale to multiply the forward signal
    scale = tf.constant(1.0)
    
    # Determine the total number of layer blocks
    total_layers = len(layer_dimensions)

    # Iterate through each layer in the 'SIREN' network
    for layer in np.arange(total_layers):
          
        # Add the input layer and the first sine layer
        if (layer == 0):                  
          
            name = "l0_input"
            input_layer = tf.keras.layers.Input(shape=(layer_dimensions[layer],),name=name)
            
            name = "l0_sinelayer"
            x = SineLayer(inputs=input_layer,units=layer_dimensions[layer+1],scale=scale,name=name)
          
        # Add the final dense output layer
        elif (layer == total_layers - 1):
          
            name = "l{}_output".format(layer)
            
            output_layer =  tf.keras.layers.Dense(units=layer_dimensions[layer],name=name)(x)
          
        # Add intermediate residual block layers
        else:
          
            name = "l{}_sineblock".format(layer)
            
            avg_1 = (layer > 1)
            avg_2 = (layer == (total_layers - 2))
            
            x = SineBlock(inputs=x,units=layer_dimensions[layer],scale=scale,name=name,avg_1=avg_1,avg_2=avg_2)
    
    # Declare the network model
    NeurComp = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    
    return NeurComp

#=============================================================================