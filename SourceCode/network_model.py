""" Created: 18.07.2022  \\  Updated: 29.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
import tensorflow as tf

#==============================================================================
# Define a 'Sine Layer' 

def SineLayer(inputs,units,activation,kernel_initializer,name):
    
    # Mathematically: x1 = sin(W1*x0 + b1)
            
    x = activation(tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer=kernel_initializer,name=name+"_dense")(inputs)) # (s*inputs) 
    
    return x

#==============================================================================
# Define a 'Sine Block'

def SineBlock(inputs,units,activation,kernel_initializer,identity_mapping,name):
    
    # Mathematically: x1 = (1/2) * (x0 + sin(w12*sin(w11*x0 + b11) + b12))
            
    sine_1 = activation(tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer=kernel_initializer,name=name+"_dense_a")(inputs)) # (s1*inputs)
    sine_2 = activation(tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer=kernel_initializer,name=name+"_dense_b")(sine_1)) # (s2*sine_1)
    
    if identity_mapping:
        x = tf.multiply(tf.math.add(inputs,sine_2),0.5)
    else:
        x = sine_2
    ##
    
    return x

#==============================================================================
# Define the positional encoding layer (from the Neural Radiance Fields paper)

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
            encoding_functions.append(lambda x, pf=pf, fb=fb: pf(x*np.pi*fb))
       
        ##
        
    ##
    
    # Evaluate the encoding function on each input and concatenate
    x = tf.concat([ef(inputs) for ef in encoding_functions],axis=-1)
    
    return x

#==============================================================================
# Define a function that constructs the 'SIREN' network 

def ConstructNetwork(layer_dimensions,frequencies,activation,kernel_initializer,identity_mapping):
    
    # Set python, numpy and tensorflow random seeds for the same initialisation
    import random; tf.random.set_seed(123);np.random.seed(123);random.seed(123)
 
    # Set activation
    if activation == "tanh": activation = tf.math.tanh
    if activation == "sine": activation = tf.math.sin
    if activation == "relu": activation = tf.nn.relu
    if activation == "gelu": activation = tf.nn.gelu
 
    # Compute the number of total network layers
    total_layers = len(layer_dimensions)

    # Iterate through network layers
    for layer in range(total_layers):
          
        # Add the input layer and the first sine layer
        if (layer == 0):                  
          
            input_layer = tf.keras.layers.Input(shape=(layer_dimensions[layer],),name="l{}_input".format(layer))
            
            # Add positional encoding if 'frequencies' > 0
            if (frequencies > 0):
                inter_layer = PositionalEncoding(inputs=input_layer,frequencies=frequencies)               
                inter_layer = SineLayer(inputs=inter_layer,units=layer_dimensions[layer+1],activation=activation,kernel_initializer=kernel_initializer,name="l{}_sinelayer".format(layer))
            else:
                inter_layer = SineLayer(inputs=input_layer,units=layer_dimensions[layer+1],activation=activation,kernel_initializer=kernel_initializer,name="l{}_sinelayer".format(layer))
          
        # Add the final output layer
        elif (layer == (total_layers - 1)):
          
            output_layer =  tf.keras.layers.Dense(units=layer_dimensions[layer],activation=activation,kernel_initializer=kernel_initializer,name="l{}_output".format(layer))(inter_layer)
          
        # Add the intermediate sine blocks
        else:
            
            inter_layer = SineBlock(inputs=inter_layer,units=layer_dimensions[layer],activation=activation,kernel_initializer=kernel_initializer,identity_mapping=identity_mapping,name="l{}_sineblock".format(layer))
            
        ##
    
    ##
    
    # Create the network model
    ISONet = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    
    # Copy attributes to network properties
    ISONet.network_type = "ISO Dataset Network"
    ISONet.layer_dimensions = layer_dimensions
    ISONet.frequencies = frequencies    
    ISONet.identity_mapping = identity_mapping
    
    return ISONet

#==============================================================================