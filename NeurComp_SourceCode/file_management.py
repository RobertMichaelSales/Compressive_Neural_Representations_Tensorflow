""" Created: 18.07.2022  \\  Updated: 25.10.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#==============================================================================
# Define a class for generating and storing all the key filenames and filepaths

class FileStructureClass():
    
    #==========================================================================
    # Define the initialisation constructor function for 'FileStructureClass'
    
    def __init__(self,parent_directory,network_save_name):
        
        #======================================================================    
        # Get parent directory and network save name
        self.parent_directory = parent_directory
        self.network_save_name = network_save_name
        
        #======================================================================    
        # Make folders        
    
        # Make TensorBoard folder + path
        self.tensorboard_path = os.path.join(self.parent_directory,
                                             "training",
                                             "logs",
                                             "tensorboard",
                                             network_save_name)  
        
        if not os.path.exists(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)
            print("Creating Folder: '{}'\n".format(self.tensorboard_path))
          
        # Make CSV summary folder + path
        self.csv_summary_path = os.path.join(self.parent_directory,
                                             "training",
                                             "logs",
                                             "csv_summary",
                                             network_save_name)
        
        if not os.path.exists(self.csv_summary_path):
            os.makedirs(self.csv_summary_path)
            print("Creating Folder: '{}'\n".format(self.csv_summary_path))
    
        # Make checkpoints folder + path
        self.checkpoints_path = os.path.join(self.parent_directory,
                                             "training",
                                             "checkpoints",
                                             network_save_name)
        
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
            print("Creating Folder: '{}'\n".format(self.checkpoints_path))
            
        # Make trained model folder + path
        self.trained_model_path = os.path.join(self.parent_directory,
                                               "training",
                                               "trained_models",
                                               network_save_name)
        
        if not os.path.exists(self.trained_model_path):
            os.makedirs(self.trained_model_path)
            print("Creating Folder: '{}'\n".format(self.trained_model_path))

        # Make output volume folder + path
        self.output_volume_path = os.path.join(self.parent_directory,
                                               "outputs",
                                               "volumes",
                                               network_save_name)
        
        if not os.path.exists(self.output_volume_path):
            os.makedirs(self.output_volume_path)
            print("Creating Folder: '{}'\n".format(self.output_volume_path))
            
        #======================================================================    
        # Make filepaths
        
        # Make normal model save filepath
        normal_model_name = network_save_name        
        self.normal_model_path = os.path.join(self.trained_model_path,
                                              normal_model_name)
        
        # Make tflite model save filepath
        tflite_model_name = network_save_name + "_normal" + ".tflite"        
        self.tflite_model_path = os.path.join(self.trained_model_path,
                                              tflite_model_name)
        
        # Make quantised model save filepath
        quantd_model_name = network_save_name + "_quantd" + ".tflite"
        self.quantd_model_path = os.path.join(self.trained_model_path,
                                              quantd_model_name)
        
        # Make hyperparameters save filepath
        configuration_name = network_save_name + "_parameters.txt"
        self.configuration_path = os.path.join(self.csv_summary_path,
                                               configuration_name)

        return None
#=============================================================================#