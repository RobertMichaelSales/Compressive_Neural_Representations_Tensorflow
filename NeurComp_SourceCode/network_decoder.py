""" Created: 15.11.2022  \\  Updated: 05.01.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np

#==============================================================================
# Define a function to decode the network layer dimensions (architecture) from

def DecodeArchitecture(architecture_path):

    # Open the architecture file in 'read as binary' mode
    with open(architecture_path,"rb") as file:
    
        # Read the architecture file (everything in one pass)
        layer_dimensions_as_bytestring = file.read()
        
        # Convert the architecture bytestring into a Python list
        layer_dimensions = np.frombuffer(layer_dimensions_as_bytestring,dtype=np.uint16).tolist()
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()
    ##
    
    return layer_dimensions

#==============================================================================

# Define a function to decode the weights/biases of each layer

def DecodeParameters(network,parameters_path):
    
    # Extract a sorted list of the names of each layer in the network
    layer_names = network.get_weight_paths().keys()
    layer_names = sorted(list(layer_names),key=SortLayerNames)
    
    # Determine the number of bytes per value
    bytes_per_float = len(np.array([1.0]).astype(np.float32).tobytes())
    
    # Open the weights file in 'read as binary' mode
    with open(parameters_path,"rb") as file:
    
        # Create an empty dictionary of the form {layer_name,weights}
        parameters = {}
        
        # Iterate through each of the network layers
        for layer_name in layer_names:
            
            # Extract the un-initialised layer from the network
            layer = network.get_weight_paths()[layer_name].numpy()
            
            # Read the current layer weights bytestring
            weights_as_bytestring = file.read(layer.size*bytes_per_float)    
            
            # Convert the bytestring into a 1-d array
            weights = np.frombuffer(weights_as_bytestring,dtype=np.float32)
            
            # Resize the 1-d array according to layer.shape
            weights = np.reshape(weights,layer.shape,order="C")
            
            # Add the weights to the dictionary
            parameters[layer_name] = weights
        ##
        
        # Read the values bounds bytestring
        bounds_as_bytestring = file.read(bytes_per_float+bytes_per_float)
        
        # Convert the bytestring into a 1-d array
        values_bounds = np.frombuffer(bounds_as_bytestring,dtype=np.float32)
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()   
    ##
    
    return parameters,values_bounds

#==============================================================================
# Define a function to assign the weights/biases of each layer

def AssignParameters(network,parameters):
        
    # Iterate through each of the network layers
    for layer_name in parameters.keys():
        
        # Extract the un-initialised layer from the network
        layer = network.get_weight_paths()[layer_name]
                     
        # Assign the weights to the un-initialised network
        layer.assign(parameters[layer_name])
    ##
    
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
    
