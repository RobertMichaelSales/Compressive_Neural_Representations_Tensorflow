""" Created: 01.05.2022  \\  Updated: 02.06.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import math
import numpy as np
import tensorflow as tf

#==============================================================================
# Define a 'Sine Layer' class

class SineLayer(tf.keras.layers.Layer):
    
    def __init__(self,units,name):
        super(SineLayer,self).__init__(name=name)
        omega = 1.0
        
        # Create learnable scale parameter(s) ('omega' from NeurComp)
        self.scale_1 = tf.Variable(initial_value=omega,dtype="float32",trainable=False)
        
        # Create dense layer(s) using the TensorFlow Keras layer API
        self.dense_1 = tf.keras.layers.Dense(units=units)
        
    def call(self,inputs):
        
        # Define the forward propagation path(s) of the input signal
        sine = tf.math.sin(self.scale_1*self.dense_1(inputs))
        
        return sine
    
#==============================================================================
# Define a 'Sine Block' class

class SineBlock(tf.keras.layers.Layer):
    
    def __init__(self,units,name):
        super(SineBlock,self).__init__(name=name)
        omega = 1.0
        
        # Create learnable scale parameter(s) ('omega' from NeurComp)
        self.scale_1 = tf.Variable(initial_value=omega,dtype="float32",trainable=False)
        self.scale_2 = tf.Variable(initial_value=omega,dtype="float32",trainable=False)
        
        # Create dense layer(s) using the TensorFlow Keras layer API
        self.dense_1 = tf.keras.layers.Dense(units=units)
        self.dense_2 = tf.keras.layers.Dense(units=units)
        
    def call(self,inputs):
               
        # Define the forward propagation path(s) of the input signal
        sine_1 = tf.math.sin(self.scale_1*self.dense_1(inputs))
        sine_2 = tf.math.sin(self.scale_2*self.dense_2(sine_1))
        
        return tf.math.add(inputs,sine_2)

#==============================================================================
#  Define a positional encoding of inputs (from the NeRF paper, Page 7, Eq.(4))

def PositionalEncoding(inputs,frequencies):
    
    # Define the positional encoding frequency bands
    frequency_bands = 2.0**tf.linspace(0.0,frequencies-1,frequencies)
    
    # Define the positional encoding periodic functions
    periodic_functions = [tf.math.sin,tf.math.cos]
    
    # Create an empty list to fill with encoding functions
    encoding_functions = []
    
    # Iterate through each of the frequency bands
    for fb in frequency_bands:
        
        # Iterate through each of the periodic functions
        for pf in periodic_functions:
            
            # Append encoding lambda functions with arguments
            encoding_functions.append(lambda x, pf=pf, fb=fb: pf(x*math.pi*fb))
        ##  
    ##
    
    # Evaluate the encoding function on each input and concatenate
    encoding = tf.concat([ef(inputs) for ef in encoding_functions],axis=-1)
    
    return encoding
 
#==============================================================================
# Define a function that creates a SIREN network using the Keras functional API

def ConstructNetwork(layer_dimensions,frequencies):
 
    # Set python, numpy and tensorflow random seeds for the same initialisation
    import random; tf.random.set_seed(123);np.random.seed(123);random.seed(123)
 
    # Compute the total number of network layers
    total_layers = len(layer_dimensions)

    # Iterate through each of the network layers
    for layer in np.arange(total_layers):
          
        # Input + Sine Layers
        if (layer == 0):                  
          
            input_layer = tf.keras.layers.Input(shape=(layer_dimensions[layer],),name="l{}_input".format(layer))
            
            if (frequencies > 0):
                x = PositionalEncoding(inputs=input_layer,frequencies=frequencies)               
                x = SineLayer(units=layer_dimensions[layer+1],name="l{}_sinelayer".format(layer))(x)
            else:
                x = SineLayer(units=layer_dimensions[layer+1],name="l{}_sinelayer".format(layer))(input_layer)
            ##
          
        # Output
        elif (layer == (total_layers - 1)):
          
            output_layer =  tf.keras.layers.Dense(units=layer_dimensions[layer],name="l{}_output".format(layer))(x)
          
        # Sine Blocks
        else:
            
            x = SineBlock(units=layer_dimensions[layer],name="l{}_sineblock".format(layer))(x)
            
        ##
    
    # Create the network model
    NeurComp = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    
    return NeurComp

#==============================================================================
