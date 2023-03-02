""" Created: 10.11.2022  \\  Updated: 01.03.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json
import numpy as np

#==============================================================================

if __name__=="__main__":
    
    # Set config filepaths
    network_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/network_config.json"
    runtime_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/runtime_config.json"
    training_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/training_config.json"
    
    # Set experiment campaign name
    campaign_name = "squashnet_test"
    
    # Iterate through all inputs
    for x in [1]:
                       
        # Define the network config
        network_config = {
            "network_name"              : campaign_name,
            "hidden_layers"             : 8,
            "frequencies"               : 0,
            "target_compression_ratio"  : 100.0,
            }
        
        # Save the network config
        with open(network_config_path,"w") as network_config_file: 
            json.dump(network_config,network_config_file,indent=4,sort_keys=True)
        
        # Define the runtime config
        runtime_config = {
            "bf_study_flag"             : False,
            "lr_study_flag"             : False,
            "ensemble_flag"             : False,
            "graph_flag"                : True,
            "stats_flag"                : False,
            "save_network_flag"         : True,
            "save_outputs_flag"         : True,
            "save_results_flag"         : True
            }
        
        # Save the runtime config
        with open(runtime_config_path,"w") as runtime_config_file: 
            json.dump(runtime_config,runtime_config_file,indent=4,sort_keys=True)
        
        # Define the training config
        training_config = {
            "initial_lr"                : 5e-3,
            "batch_size"                : 1024,
            "batch_fraction"            : 0,
            "epochs"                    : 2,
            "half_life"                 : 3,            
            }            
        
        # Save the training config
        with open(training_config_path,"w") as training_config_file: 
            json.dump(training_config,training_config_file,indent=4,sort_keys=True)

        # Run the compression experiment
        runstring = "python NeurComp_SourceCode/compress_main.py " + "'" + json.dumps(network_config) + "' '" + json.dumps(runtime_config) + "' '" + json.dumps(training_config) + "'"
        os.system(runstring)
        
#==============================================================================
   
