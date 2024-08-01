""" Created: 15.11.2022  \\  Updated: 29.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np

#==============================================================================
# Reads and decodes the network layer dimensions (architecture) from binary

def DecodeArchitecture(architecture_path):
    
    # Determine the number of bytes per value
    bytes_per_value = len(np.array([1]).astype('uint16').tobytes())

    # Open the architecture file in 'read as binary' mode
    with open(architecture_path,"rb") as file:
        
        # Read the total number of layer dimensions as bytestring
        total_num_layers_as_bytestring = file.read(1*bytes_per_value)
        total_num_layers = int(np.frombuffer(total_num_layers_as_bytestring,dtype='uint16'))
        
        # Read the list of network layer dimensions as bytestring
        layer_dimensions_as_bytestring = file.read(total_num_layers*bytes_per_value)
        layer_dimensions = list(np.frombuffer(layer_dimensions_as_bytestring,dtype=np.uint16))
    
        # Read the number of positional encoding frequencies as bytestring
        frequencies_as_bytestring = file.read(1*bytes_per_value)
        frequencies = int(np.frombuffer(frequencies_as_bytestring,dtype='uint16'))
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()
    ##
    
    return layer_dimensions,frequencies

##

#==============================================================================

# Reads and decodes the weights/biases of each layer of a network from binary

def DecodeParameters(network,parameters_path):
    
    # Extract a sorted list of the names of each layer in the network
    layer_names = sorted(list(network.get_weight_paths().keys()),key=SortLayerNames)
    
    # Determine the number of bytes per value
    bytes_per_value = len(np.array([1.0]).astype('float32').tobytes())
    
    # Determine input and output dimensions
    i_dimensions = network.layer_dimensions[0 ]
    o_dimensions = network.layer_dimensions[-1]
    
    # Open the weights file in 'read as binary' mode
    with open(parameters_path,"rb") as file:
    
        # Create an empty dictionary of the form {layer_name,weights}
        parameters = {}
        
        # Iterate through each of the network layers
        for layer_name in layer_names:
            
            # Extract the un-initialised layer from the network
            layer = network.get_weight_paths()[layer_name].numpy()
            
            # Read the current layer weights bytestring
            weights_as_bytestring = file.read(layer.size*bytes_per_value)    
            
            # Convert the bytestring into a 1-d array
            weights = np.frombuffer(weights_as_bytestring,dtype='float32')
            
            # Resize the 1-d array according to layer.shape
            weights = np.reshape(weights,layer.shape,order="C")
            
            # Add the weights to the dictionary
            parameters[layer_name] = weights
        ##
        
        # Read the original values bounds bytestring
        original_values_bounds_as_bytestring = file.read(o_dimensions*bytes_per_value*2)
        
        # Convert the bytestring into a numpy array
        original_values_bounds = np.frombuffer(original_values_bounds_as_bytestring,dtype='float32').reshape(o_dimensions,2)
        
        #============================= TEMPORARY? =============================            

        # Read the original volume centre bytestring
        original_volume_centre_as_bytestring = file.read(i_dimensions*bytes_per_value*1)

        # Convert the bytestring into a numpy array
        original_volume_centre = np.frombuffer(original_volume_centre_as_bytestring,dtype='float32')
        
        # Read the original volume radius bytestring
        original_volume_radius_as_bytestring = file.read(bytes_per_value)

        # Convert the bytestring into a numpy array
        original_volume_radius = np.frombuffer(original_volume_radius_as_bytestring,dtype='float32')
        
        #============================= TEMPORARY? =============================
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()   
    ##
    
    return parameters,original_values_bounds,original_volume_centre,original_volume_radius

##

#==============================================================================
# Assign the weights/biases of each layer to the weights/biases of a network

def AssignParameters(network,parameters):
        
    # Iterate through each of the network layers
    for layer_name in parameters.keys():
        
        # Extract the un-initialised layer from the network
        layer = network.get_weight_paths()[layer_name]
                     
        # Assign the weights to the un-initialised network
        layer.assign(parameters[layer_name])
    ##
    
    return None

##
    
#==============================================================================

# Sort layer names so that saved weights/biases are always in the correct order 

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

##

#==============================================================================