""" Created: 18.07.2022  \\  Updated: 09.02.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import math
import numpy as np
import tensorflow as tf

#==============================================================================
# Define a 'Sine Layer' 

def SineLayer(inputs,units,name):
    
    # x1 = sin(W1*x0 + b1)
    
    x = tf.keras.layers.Dense(units=units,name=name+"_dense")(inputs)
    x = tf.math.sin(x)
    
    return x

#==============================================================================
# Define a 'Sine Block'

def SineBlock(inputs,units,name):
    
    # x1 = (1/2) * (x0 + sin(w12*sin(w11*x0 + b11) + b12))
        
    sine_1 = tf.math.sin(tf.keras.layers.Dense(units=units,name=name+"_dense_a")(inputs))
    sine_2 = tf.math.sin(tf.keras.layers.Dense(units=units,name=name+"_dense_b")(sine_1))
    
    x = tf.math.add(inputs,sine_2)
    
    return x


#==============================================================================
# Define the positional encoding layer

def PositionalEncoding(inputs,frequencies):
    
    frequency_bands = 2.0**tf.linspace(0.0,frequencies-1,frequencies)
    
    periodic_functions = [tf.math.sin,tf.math.cos]
    
    encoding_functions = []
    
        
    for fb in frequency_bands:
        
        for pf in periodic_functions:
            
            encoding_functions.append(lambda x, pf=pf, fb=fb: pf(x*math.pi*fb))
       
        ##
        
    ##
    
    x = tf.concat([ef(inputs) for ef in encoding_functions],axis=-1)
    
    return x

#==============================================================================
# Define a function that constructs the 'SIREN' network 

def ConstructNetwork(layer_dimensions,frequencies):
 
    total_layers = len(layer_dimensions)

    for layer in np.arange(total_layers):
          
        # Add the input layer and the first sine layer
        if (layer == 0):                  
          
            input_layer = tf.keras.layers.Input(shape=(layer_dimensions[layer],),name="l{}_input".format(layer))
            
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