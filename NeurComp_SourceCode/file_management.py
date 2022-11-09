""" Created: 18.07.2022  \\  Updated: 08.11.2022  \\   Author: Robert Sales """

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
        self.base_directory = base_directory
        self.network_name = network_config.network_name
        
        print("\nCreating FileClass Object For: '{}'".format(self.network_name))
        
        #======================================================================
        # Make the outputs path and create directory folder
        
        self.output_directory = os.path.join(self.base_directory,"outputs",self.network_name)
        if not os.path.exists(self.output_directory): os.makedirs(self.output_directory)
        
        #======================================================================
        # Set the output file filepaths       
        
        # Make the network architecture, weights and biases .json filepaths
        self.network_architecture_path = os.path.join(self.output_directory,"network_architecture.json")
        self.network_ws_path = os.path.join(self.output_directory,"network_ws.json")
        self.network_bs_path = os.path.join(self.output_directory,"network_bs.json")
        
        # Set the network configuration .json filepath
        self.network_configuration_path = os.path.join(self.output_directory,"network_configuration.json")
        
        # Make the network image PNG filepath
        self.network_image_path = os.path.join(self.output_directory,"network_image.png")
        
        # Make the training data and output volume filepaths
        self.training_data_path = os.path.join(self.output_directory,"training_data.json")
        self.output_volume_path = os.path.join(self.output_directory,"output_volume")
 
        return None
        
    
#==============================================================================    
