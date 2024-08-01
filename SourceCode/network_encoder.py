""" Created: 15.11.2022  \\  Updated: 29.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
import json

#==============================================================================
# Encodes the network layer dimensions as a binary file
# Note: The np.method '.tobytes()' returns the same bytestring as 'struct.pack()'

def EncodeArchitecture(network,architecture_path):
    
    # Extract the total number of layer dimensions to bytestrings
    total_num_layers = np.array(len(network.layer_dimensions)).astype('uint16')    
    total_num_layers_as_bytestring = total_num_layers.tobytes()
    
    # Extract the list of network layer dimensions to bytestrings
    layer_dimensions = np.array(network.layer_dimensions).astype('uint16')
    layer_dimensions_as_bytestring = layer_dimensions.tobytes()
    
    # Extract the number of positional encoding frequencies to bytestrings
    frequencies = np.array(network.frequencies).astype('uint16')
    frequencies_as_bytestring = frequencies.tobytes()
    
    # Open the architecture file in 'write as binary' mode
    with open(architecture_path,"wb") as file:
        
        # Write each bytestring to file
        file.write(total_num_layers_as_bytestring)
        file.write(layer_dimensions_as_bytestring)
        file.write(frequencies_as_bytestring)
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()
    ##

    return None

#==============================================================================
# Encodes the weights and biases of each layer as a binary file
# Note: The np.method '.tobytes()' returns the same bytestring as 'struct.pack()'

def EncodeParameters(network,parameters_path):
    
    # Extract a sorted list of the names of each layer in the network
    layer_names = sorted(list(network.get_weight_paths().keys()),key=SortLayerNames)
    
    # Open the parameters file in 'write as binary' mode
    with open(parameters_path,"wb") as file:
    
        # Iterate through each of the network layers, in order 
        for layer_name in layer_names: 
            
            # Extract the layer weights and biases
            weights = network.get_weight_paths()[layer_name].numpy()
            
            # Flatten the layer weights and biases
            weights = np.ravel(weights,order="C").astype('float32')
        
            # Serialise weights into a string of bytes
            weights_as_bytestring = weights.tobytes(order="C")
                 
            # Write 'weight_as_bytestring' to file
            file.write(weights_as_bytestring)
        ##
            
        # Convert original values bounds to a numpy array
        original_values_bounds = np.array(network.original_values_bounds).astype('float32')
        
        # Serialise original values bounds into a string of bytes
        original_values_bounds_as_bytestring = original_values_bounds.tobytes(order="C")
        
        # Write 'original_bounds_as_bytestring' to file
        file.write(original_values_bounds_as_bytestring)

        #============================= TEMPORARY? =============================            

        # Convert original volume centre to a numpy array
        original_volume_centre = np.array(network.original_volume_centre).astype('float32')
        
        # Serialise original volume centre into a string of bytes
        original_volume_centre_as_bytestring = original_volume_centre.tobytes(order="C")
        
        # Write 'original_bounds_as_bytestring' to file
        file.write(original_volume_centre_as_bytestring)
        
        # Convert original volume radius to a numpy array
        original_volume_radius = np.array(network.original_volume_radius).astype('float32')
        
        # Serialise original volume radius into a string of bytes
        original_volume_radius_as_bytestring = original_volume_radius.tobytes(order="C")
        
        # Write 'original_bounds_as_bytestring' to file
        file.write(original_volume_radius_as_bytestring)
        
        #============================= TEMPORARY? =============================            
                
        # Flush the buffer and close the file 
        file.flush()
        file.close()
    ##
    
    return None

##

#==============================================================================

def SaveNetworkJSON(network,network_data_path):
    
    # Create empty dictionary for storing network data
    network_data = {"architecture": {}, "parameters": {}}
    
    # Add architecture information
    network_data["architecture"]["network_type"] = str(network.network_type)
    network_data["architecture"]["layer_dimensions"] = list(network.layer_dimensions)
    network_data["architecture"]["frequencies"] = int(network.frequencies)
    network_data["architecture"]["original_values_bounds"] = list(np.array(network.original_values_bounds).tolist())
    
    # Extract a sorted list of the names of each layer in the network
    layer_names = sorted(list(network.get_weight_paths().keys()),key=SortLayerNames)
    
    # Iterate through each of the network layers, in order 
    for layer_name in layer_names: 
        
        # Extract the layer weights and biases
        weights = network.get_weight_paths()[layer_name].numpy()
        
        # Flatten the layer weights and biases
        weights = np.ravel(weights,order="C").astype('float32')
        
        # Add parameters information
        network_data["parameters"][layer_name] = weights.tolist()
    
    ##
    
    # Write to JSON file
    with open(network_data_path,"w") as file: json.dump(network_data,file,indent=4,sort_keys=False)
    
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

