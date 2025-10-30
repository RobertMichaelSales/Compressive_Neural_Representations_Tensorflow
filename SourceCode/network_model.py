""" Created: 18.07.2022  \\  Updated: 29.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
import tensorflow as tf

#==============================================================================
# Define a 'Layer' 

def Layer(inputs,units,activation_function,kernel_initializer,omega_0,name):
        
    if kernel_initializer == "siren":
        w_std = 1 / inputs.shape[-1]
        kernel_initializer = tf.keras.initializers.RandomUniform(-w_std,w_std)
    else:
        kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    ##
    
    x = activation_function(tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer=kernel_initializer,name=name+"_dense")(omega_0 * inputs))
    
    return x

#==============================================================================
# Define a 'Block'

def Block(inputs,units,activation_function,kernel_initializer,identity_mapping,name):
    
    if kernel_initializer == "siren":
        w_std = np.sqrt(6 / units)
        kernel_initializer_1 = tf.keras.initializers.RandomUniform(-w_std,w_std)
        kernel_initializer_2 = tf.keras.initializers.RandomUniform(-w_std,w_std)
    else:
        kernel_initializer_1 = tf.keras.initializers.get(kernel_initializer)
        kernel_initializer_2 = tf.keras.initializers.get(kernel_initializer)
    ##
            
    sine_1 = activation_function(tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer=kernel_initializer_1,name=name+"_dense_a")(inputs))

    sine_2 = activation_function(tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer=kernel_initializer_2,name=name+"_dense_b")(sine_1))
    
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

def ConstructNetwork(layer_dimensions,frequencies,activation,kernel_initializer,identity_mapping,omega_0,**kwargs):
    
    # Set python, numpy and tensorflow random seeds for the same initialisation
    import random; tf.random.set_seed(123);np.random.seed(123);random.seed(123)
 
    # Set activation
    if activation == "tanh": activation_function = tf.math.tanh
    if activation == "sine": activation_function = tf.math.sin
    if activation == "relu": activation_function = tf.nn.relu
    
    # Set omega for sine only
    if activation != "sine": omega_0 = 1.0
     
    # Compute the number of total network layers
    total_layers = len(layer_dimensions)

    # Iterate through network layers
    for layer in range(total_layers):
          
        # Add the input and projection layers
        if (layer == 0):                  
          
            inputs = tf.keras.layers.Input(shape=(layer_dimensions[layer],),name="l{}_input".format(layer))
            x = inputs
            
            # Add positional encoding if 'frequencies' > 0
            if (frequencies > 0):
                
                x = PositionalEncoding(inputs=inputs,frequencies=frequencies) 
                
                x = Layer(inputs=x,units=layer_dimensions[layer+1],activation_function=activation_function,kernel_initializer=kernel_initializer,omega_0=omega_0,name="l{}_sinelayer".format(layer))
                
            else:
                
                x = Layer(inputs=x,units=layer_dimensions[layer+1],activation_function=activation_function,kernel_initializer=kernel_initializer,omega_0=omega_0,name="l{}_sinelayer".format(layer))
          
        # Add the final output layer
        elif (layer == (total_layers - 1)):
          
            outputs =  tf.keras.layers.Dense(units=layer_dimensions[layer],activation=None,kernel_initializer=tf.keras.initializers.GlorotUniform(),name="l{}_output".format(layer))(x)
          
        # Add the intermediate sine blocks
        else:
          
            x = Block(inputs=x,units=layer_dimensions[layer],activation_function=activation_function,kernel_initializer=kernel_initializer,identity_mapping=identity_mapping,name="l{}_sineblock".format(layer))
            
        ##
    
    ##
    
    # Create the network model
    ISONet = tf.keras.Model(inputs=inputs,outputs=outputs)
    
    # Copy attributes to network properties
    ISONet.network_type = "ISO Network"
    ISONet.layer_dimensions = layer_dimensions
    ISONet.frequencies = frequencies    
    ISONet.activation = activation
    ISONet.kernel_initializer = kernel_initializer
    ISONet.identity_mapping = identity_mapping
    ISONet.omega_0 = omega_0
    
    return ISONet

#==============================================================================