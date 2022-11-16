""" Created: 18.07.2022  \\  Updated: 10.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#==============================================================================
# Define a class for generating and storing all the key filenames and filepaths

class FileClass():
    
    #==========================================================================
    # Define the initialisation constructor function for 'FileStructureClass'
    
    def __init__(self,base_directory,network_config):
        
        # Copy the base directory and network save name
        base_directory  = base_directory
        network_name    = network_config.network_name
        
        # Set the outputs path and create the output folder
        self.output_directory = os.path.join(base_directory,"outputs",network_name)
        if not os.path.exists(self.output_directory): os.makedirs(self.output_directory)
        
        # Set the architecture filepath 
        self.network_architecture_path  = os.path.join(self.output_directory,"network_architecture")
        
        # Set the network weights filepath
        self.network_weights_path       = os.path.join(self.output_directory,"network_weights")
        
        # Set the configuration filepath
        self.network_configuration_path = os.path.join(self.output_directory,"network_configuration.json")
        
        # Set the network image filepath
        self.network_image_path         = os.path.join(self.output_directory,"network_image.png")
        
        # Set the training data filepath
        self.training_data_path         = os.path.join(self.output_directory,"training_data.json")
        
        # Set the output volume filepath
        self.output_volume_path         = os.path.join(self.output_directory,"output_volume")
 
        return None
    
#==============================================================================    
