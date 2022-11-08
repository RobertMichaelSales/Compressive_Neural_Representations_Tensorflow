""" Created: 18.07.2022  \\  Updated: 08.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os

#==============================================================================
# Define a class for generating and storing all the key filenames and filepaths

class FileClass():
    
    #==========================================================================
    # Define the initialisation constructor function for 'FileStructureClass'
    
    def __init__(self,base_directory,network_config):
        
        # Copy the base directory and network save name
        self.base_dir = base_directory
        self.network_name = network_config.network_name
        
        print("\nCreating FileClass Object For: {}".format(self.network_name))

        return None
        
    #==========================================================================
    # Define a function to create the output directory and output file paths    
    
    def MakeDirectoriesAndFilepaths(self):
        
        # Make the outputs path and create directory folder
        
        self.output_dir = os.path.join(self.base_dir,"outputs",self.network_name)
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        
        print("Creating Output Directory At: \\{}".format(self.network_name))
    
        
        # Make the network configuration .json filepath
        self.network_configuration_path = os.path.join(self.output_dir,"network_configuration.json")
        
        # Make the network weights and biases .json filepath
        self.network_weights_and_biases_path = os.path.join(self.output_dir,"network_weights_and_biases.json")
        
        # Make the network architecture .json filepath
        self.network_architecture_path = os.path.join(self.output_dir,"network_architecture.json")
        
        # Make the training data .json filepath
        self.training_data_path = os.path.join(self.output_dir,"training_data.json")
        
        # Make the output volume .npy filepath
        self.output_volume_path = os.path.join(self.output_dir,"output_volume.npy")
 
        return None
        
    
#==============================================================================    
