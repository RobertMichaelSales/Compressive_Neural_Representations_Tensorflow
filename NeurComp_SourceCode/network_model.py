""" Created: 18.07.2022  \\  Updated: 02.06.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import math
import numpy as np
import tensorflow as tf

#==============================================================================
# Define a 'Sine Layer' 

def SineLayer(inputs,units,name):
    
    # Mathematically: x1 = sin(W1*x0 + b1)
    
    # s = tf.Variable(initial_value=1.0,dtype=tf.float32,trainable=True,name=name+"_scale")
        
    x = tf.keras.layers.Dense(units=units,name=name+"_dense")(inputs) # (s*inputs)
    x = tf.math.sin(x)
    
    return x

#==============================================================================
# Define a 'Sine Block'

def SineBlock(inputs,units,name):
    
    # Mathematically: x1 = (1/2) * (x0 + sin(w12*sin(w11*x0 + b11) + b12))
    
    # s1 = tf.Variable(initial_value=1.0,dtype=tf.float32,trainable=True,name=name+"_scale_a")
    # s2 = tf.Variable(initial_value=1.0,dtype=tf.float32,trainable=True,name=name+"_scale_b")
            
    sine_1 = tf.math.sin(tf.keras.layers.Dense(units=units,name=name+"_dense_a")(inputs)) # (s1*inputs)
    sine_2 = tf.math.sin(tf.keras.layers.Dense(units=units,name=name+"_dense_b")(sine_1)) # (s2*sine_1)
    
    x = tf.math.add(inputs,sine_2)
    
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

def ConstructNetwork(layer_dimensions,frequencies):
    
    # Set python, numpy and tensorflow random seeds for the same initialisation
    import random; tf.random.set_seed(123);np.random.seed(123);random.seed(123)
 
    # Record the number of total network layers
    total_layers = len(layer_dimensions)

    for layer in range(total_layers):
          
        # Add the input layer and the first sine layer
        if (layer == 0):                  
          
            input_layer = tf.keras.layers.Input(shape=(layer_dimensions[layer],),name="l{}_input".format(layer))
            
            # Add positional encoding if 'frequencies' > 0
            if (frequencies > 0):
                x = PositionalEncoding(inputs=input_layer,frequencies=frequencies)               
                x = SineLayer(inputs=x          ,units=layer_dimensions[layer+1],name="l{}_sinelayer".format(layer))
            else:
                x = SineLayer(inputs=input_layer,units=layer_dimensions[layer+1],name="l{}_sinelayer".format(layer))
          
        # Add the final output layer
        elif (layer == (total_layers - 1)):
          
            output_layer =  tf.keras.layers.Dense(units=layer_dimensions[layer],name="l{}_output".format(layer))(x)
          
        # Add the intermediate sine blocks
        else:
            
            x = SineBlock(inputs=x,units=layer_dimensions[layer],name="l{}_sineblock".format(layer))
    
    # Create the network model
    NeurComp = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    
    return NeurComp

#==============================================================================