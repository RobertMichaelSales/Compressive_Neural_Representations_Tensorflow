""" Created: 15.11.2022  \\  Updated: 26.09.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import json
import numpy as np

#==============================================================================

def SaveNetworkJSON(network,network_data_filepath):
    
    # Create empty dictionary for storing network data
    network_data = {"architecture": {}, "parameters": {}, "metadata": {}}
    
    # Add architecture information
    network_data["architecture"]["network_type"] = str(network.network_type)
    network_data["architecture"]["layer_dimensions"] = list(network.layer_dimensions)
    network_data["architecture"]["frequencies"] = int(network.frequencies)
    network_data["architecture"]["activation"] = str(network.activation)
    network_data["architecture"]["identity_mapping"] = bool(network.identity_mapping)
    network_data["architecture"]["omega_0"] = float(network.omega_0)
    
    # Add normalisation parameters
    network_data["metadata"]["original_coords_centre"] = network.original_coords_centre.astype(np.float32).tolist()
    network_data["metadata"]["original_coords_radius"] = float(network.original_coords_radius.astype(np.float32))
    network_data["metadata"]["original_values_bounds"] = network.original_values_bounds.astype(np.float32).tolist()
    
    # Iterate through each of the network layers, in order 
    for layer_name in sorted(list(network.get_weight_paths().keys()),key=SortLayerNames): 
        
        # Extract the layer
        layer = network.get_weight_paths()[layer_name]
        
        # Extract type
        layer_type = layer_name.split(".")[1]
        
        # Extract data
        layer_data = layer.numpy().flatten(order="C").astype(np.float32).tolist()
        
        # Extract dims
        if (layer_type == "kernel"):
            layer_dims = list(layer.shape)
        else:
            layer_dims = list(layer.shape + (1,))
        ##
        
        # Add parameters information
        network_data["parameters"][layer_name] = {"data": layer_data, "dims": layer_dims}
    
    ##
    
    # Write to JSON file
    with open(network_data_filepath,"w") as file: json.dump(network_data,file,indent=4,sort_keys=False)
    
    return None
##
    
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
##

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
            weights = np.ravel(weights,order="C").astype(np.float32)
        
            # Serialise weights into a string of bytes
            weights_as_bytestring = weights.tobytes(order="C")
                 
            # Write 'weight_as_bytestring' to file
            file.write(weights_as_bytestring)
            
        ##
         
        # Convert original centre to a numpy array
        original_coords_centre = np.array(network.original_coords_centre).astype(np.float32)
        
        # Serialise original centre into a string of bytes
        original_coords_centre_as_bytestring = original_coords_centre.tobytes(order="C")
        
        # Write 'original_coords_centre_as_bytestring' to file
        file.write(original_coords_centre_as_bytestring)
        
        # Convert original radius to a numpy array
        original_coords_radius = np.array(network.original_coords_radius).astype(np.float32)
        
        # Serialise original radius into a string of bytes
        original_coords_radius_as_bytestring = original_coords_radius.tobytes(order="C")
        
        # Write 'original_coords_radius_as_bytestring' to file
        file.write(original_coords_radius_as_bytestring)
        
        # Convert original radius to a numpy array
        original_values_bounds = np.array(network.original_values_bounds).astype(np.float32)
        
        # Serialise original radius into a string of bytes
        original_values_bounds_as_bytestring = original_values_bounds.tobytes(order="C")
        
        # Write 'original_coords_radius_as_bytestring' to file
        file.write(original_values_bounds_as_bytestring)
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()
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