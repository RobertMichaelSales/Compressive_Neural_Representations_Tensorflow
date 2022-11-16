""" Created: 15.11.2022  \\  Updated: 15.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

#==============================================================================
# Define a function to encode the weights/biases of each layer as a binary file
# containing strings of bytes

# Note: The np.method '.tobytes()' returns the same bytestring as 'struct.pack'

def EncodeWeights(network,filepath):
    
    # Extract a sorted list of the names of each layer in the network
    layer_names = network.get_weight_paths().keys()
    layer_names = sorted(list(layer_names),key=SortLayerNames)
    
    # Open the weights file in 'write as binary' mode
    file = open(filepath,"wb")
    
    # Iterate through each of the network layers, in order 
    for layer_name in layer_names: 
        
        # Extract the layer weights and biases
        weights = network.get_weight_paths()[layer_name].numpy()
        
        # Flatten the layer weights and biases
        weights = np.ravel(weights,order="C").astype(np.float32)
    
        # Serialise weights into a string of bytes
        weights_bytestring = weights.tobytes(order="C")
             
        # Write 'weight_bytestring' to file
        file.write(weights_bytestring)
            
    # Flush the buffer and close the file 
    file.flush()
    file.close()
    
    return None
    
#==============================================================================
# Define a function to encode the network layer dimensions (or architecture) as
# a binary file containing strings of bytes

def EncodeArchitecture(config,filepath):

    # Extract the list of network layer dimensions
    layer_dimensions = np.array(config.layer_dimensions).astype(np.uint16)
    
    # Open the architecture file in 'write as binary' mode
    file = open(filepath,"wb")
    
    # Serialise layer dimensions into a string of bytes
    layer_dimensions_bytestring = layer_dimensions.tobytes(order="C")
    
    # Write 'layer_dimensions_bytestring' to file
    file.write(layer_dimensions_bytestring)
    
    # Flush the buffer and close the file 
    file.flush()
    file.close()

    return None

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
    