""" Created: 10.11.2022  \\  Updated: 10.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json

from NeurComp_SourceCode.compress_main import main as compress

# #==============================================================================

# if __name__=="__main__":
    
#     # Set config filepath
#     config_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/config.json"
    
#     # Set base directory
#     base_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles"

#     # Set input filepath
#     input_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/test_vol.npy"
    
#     # Set up experiments 
#     task_id = int(input("Enter task ID: "))
#     target_compression_ratio = float(input("Enter target compression ratio: "))
#     os.system('clear')
        
#     # Define the network config
#     config_dictionary = {
#         "network_name"               : "neurcomp_"+str(task_id),
#         "target_compression_ratio"   : target_compression_ratio,
#         "hidden_layers"              : 8,
#         "min_neurons_per_layer"      : 10,
#         "initial_learning_rate"      : 5e-3,
#         "batch_size"                 : 1024,
#         "num_epochs"                 : 30,
#         "decay_rate"                 : 3                        
#         }
    
#     # Save the network config
#     with open(config_filepath,"w") as config_file: 
#         json.dump(config_dictionary,config_file,indent=4,sort_keys=True)

#     # Run compression algorithm
#     compress(base_directory=base_directory,input_filepath=input_filepath,config_filepath=config_filepath)     
        
# #==============================================================================

#==============================================================================

if __name__=="__main__":
    
    # Set config filepath
    config_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/config.json"
    
    # Set base directory
    base_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles"

    # Set input filepath
    input_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/test_vol.npy"
    
    # Set up experiments 
    for task_id,target_compression_ratio in enumerate([10,20,30]):
    
    # task_id = int(input("Enter task ID: "))
    # target_compression_ratio = float(input("Enter target compression ratio: "))
    # os.system('clear')
        
        # Define the network config
        config_dictionary = {
            "network_name"               : "neurcomp_"+str(task_id),
            "target_compression_ratio"   : target_compression_ratio,
            "hidden_layers"              : 8,
            "min_neurons_per_layer"      : 10,
            "initial_learning_rate"      : 5e-3,
            "batch_size"                 : 1024,
            "num_epochs"                 : 1,
            "decay_rate"                 : 3                        
            }
        
        # Save the network config
        with open(config_filepath,"w") as config_file: 
            json.dump(config_dictionary,config_file,indent=4,sort_keys=True)
    
        # Run compression algorithm
        compress(base_directory=base_directory,input_filepath=input_filepath,config_filepath=config_filepath)     
        
#==============================================================================
