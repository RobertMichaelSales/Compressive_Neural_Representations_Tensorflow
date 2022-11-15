""" Created: 15.11.2022  \\  Updated: 15.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

#==============================================================================
# Define a function to encode the weights/biases of each layer as a binary file
# containing strings of bytes.

# Note: The np.method '.tobytes()' returns the same bytestring as 'struct.pack'

def EncodeWeights(network,filepath="weights.bin"):
    
    # Extract a sorted list of the names of each layer in the network
    layer_names = sorted(list(network.get_weight_paths().keys()),key=SortLayerNames,reverse=False)
    
    # Open the weights file in 'write as binary' mode
    file = open(filepath,"wb")
    
    # Iterate through each of the network layers, in order 
    for layer_name in layer_names: 
        
        # Extract the layer weights and biases
        weights = network.get_weight_paths()[layer_name].numpy()
        
        # Flatten the layer weights and biases
        weights = np.ravel(weights,order="C").astype(np.float32)
    
        # Serialise weights into a string of bytes
        weight_bytestring = weights.tobytes(order="C")
             
        # Write 'weight_bytestring' to file
        file.write(weight_bytestring)
            
    # Flush the buffer and close the file 
    file.flush()
    file.close()

def EncodeArchitecture(network,filepath):

    # Extract the network configuration 
    
    # with open(filepaths.network_architecture_path,"w") as file: json.dump(neur_comp.get_config(),file,indent=4)

    # THE NETWORK ARCHITECTURE ONLY DEPENDS ON:
        
    #     network_config.network_name
        
    #     network_config.total_layers
        
    #     network_config.layer_dimensions
        
    # BECAUSE network_make.py ONLY REQUIRES THOSE THREE THINGS

    return None


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

    
    