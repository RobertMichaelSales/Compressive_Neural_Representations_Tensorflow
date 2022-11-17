""" Created: 15.11.2022  \\  Updated: 15.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

#==============================================================================
# Define a function to decode the weights/biases of each layer

def DecodeWeights(network,filepath):
    
    # Extract a sorted list of the names of each layer in the network
    layer_names = network.get_weight_paths().keys()
    layer_names = sorted(list(layer_names),key=SortLayerNames)
    
    # Determine the number of bytes per value
    bytes_per_weight = len(np.array([1.0]).astype(np.float32).tobytes())
    
    # Open the weights file in 'read as binary' mode
    file = open(filepath,"rb")
    
    # Create an empty dictionary of the form {layer_name,weights}
    weights_dict = {}
    
    # Iterate through each of the network layers
    for layer_name in layer_names:
        
        # Extract the un-initialised layer from the network
        layer = network.get_weight_paths()[layer_name]
              
        # Extract the layer shape and size
        layer_shape,layer_size = layer.numpy().shape, layer.numpy().size
        
        # Read the current layer's weights
        weights_bytestring = file.read(layer_size*bytes_per_weight)    
        
        # Convert the weights bytestring into a 1-d array
        weights = np.frombuffer(weights_bytestring,dtype=np.float32)
        
        # Resize the 1-d array according to layer_shape
        weights = np.reshape(weights,layer_shape,order="C")
        
        # Add the weights to the dictionary
        weights_dict[layer_name] = weights

    # Flush the buffer and close the file 
    file.flush()
    file.close()    
    
    return weights_dict

#==============================================================================
# Define a function to assign the weights/biases of each layer

def AssignWeights(network,weights_dict):
        
    # Iterate through each of the network layers
    for layer_name in weights_dict.keys():
        
        # Extract the un-initialised layer from the network
        layer = network.get_weight_paths()[layer_name]
                     
        # Assign the weights to the un-initialised network
        layer.assign(weights_dict[layer_name])
    
    return network
    
#==============================================================================
# Define a function to decode the network layer dimensions (architecture) from

def DecodeArchitecture(filepath):

    # Open the architecture file in 'read as binary' mode
    file = open(filepath,"rb")
    
    # Read the architecture file (everything in one pass)
    layer_dimensions_bytestring = file.read()
    
    # Convert the architecture bytestring into a Python list
    layer_dimensions = np.frombuffer(layer_dimensions_bytestring,dtype=np.uint16).tolist()
    
    # Flush the buffer and close the file 
    file.flush()
    file.close()
    
    return layer_dimensions

#==============================================================================
# Define a function to sort the layer names alpha-numerically so that the saved
# weights are always in the correct order

def SortLayerNames(layer_name):
    
    layer_index = int(layer_name.split("_")[0][1:])

    if "_a" in layer_name: 
        layer_index = layer_index
    
    if "_b" in layer_name: 
        layer_index = layer_index + 0.50
        
    if ".kernel" in layer_name: 
        layer_index = layer_index
        
    if ".bias" in layer_name: 
        layer_index = layer_index + 0.25   
    
    return layer_index

#==============================================================================
    
