""" Created: 18.07.2022  \\  Updated: 10.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

'''
#==============================================================================
# Define a 'Sine Layer' 

def OLDSineLayer(inputs,units,scale,name):
    
    # x1 = sin(W1*x0 + b1)
    
    x = tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros",name=name+"_dense")(inputs)
    x = tf.math.sin(tf.math.multiply(x,scale))
    
    return x
    
#==============================================================================
# Define a 'Sine Block'

def OLDSineBlock(inputs,units,scale,name,avg_1,avg_2):
    
    # x1 = (1/2) * (x0 + sin(w12*sin(w11*x0 + b11) + b12))
            
    weight_1 = tf.constant(0.5) if avg_1 else tf.constant(1.0)  # Weight 1
    weight_2 = tf.constant(0.5) if avg_2 else tf.constant(1.0)  # Weight 2
        
    sine_1 = tf.math.sin(tf.math.multiply(tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros",name=name+"_dense_a")(inputs*weight_1),scale))
    sine_2 = tf.math.sin(tf.math.multiply(tf.keras.layers.Dense(units=units,activation=None,use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros",name=name+"_dense_b")(sine_1)         ,scale))
    
    x = tf.math.add(tf.math.multiply(inputs,weight_2),sine_2)
    
    return x

#==============================================================================
# Define a function that constructs the 'SIREN' network 

def OLDConstructNetwork(layer_dimensions):
    
    total_layers = len(layer_dimensions)
    
    scale = tf.constant(1.0)

    for layer in np.arange(total_layers):
          
        # Add the input layer and the first sine layer
        if (layer == 0):                  
          
            input_layer = tf.keras.layers.Input(shape=(layer_dimensions[layer],),name="l{}_input".format(layer))
            x = SineLayer(inputs=input_layer,units=layer_dimensions[layer+1],scale=scale,name="l{}_sinelayer".format(layer))
          
        # Add the final output layer
        elif (layer == (total_layers - 1)):
          
            output_layer =  tf.keras.layers.Dense(units=layer_dimensions[layer],name="l{}_output".format(layer))(x)
          
        # Add the intermediate sine blocks
        else:
            
            avg_1 = False # (layer > 1)
            avg_2 = False #(layer == (total_layers - 2))           # this used to be 3
        
            x = SineBlock(inputs=x,units=layer_dimensions[layer],scale=scale,avg_1=avg_1,avg_2=avg_2,name="l{}_sineblock".format(layer))
    
    # Create the network model
    NeurComp = tf.keras.Model(inputs=input_layer,outputs=output_layer)
    
    return NeurComp

#==============================================================================
'''


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
# Define a function that constructs the 'SIREN' network 

def ConstructNetwork(layer_dimensions):
    
    total_layers = len(layer_dimensions)

    for layer in np.arange(total_layers):
          
        # Add the input layer and the first sine layer
        if (layer == 0):                  
          
            input_layer = tf.keras.layers.Input(shape=(layer_dimensions[layer],),name="l{}_input".format(layer))
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