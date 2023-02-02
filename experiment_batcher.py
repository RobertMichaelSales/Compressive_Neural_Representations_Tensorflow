""" Created: 10.11.2022  \\  Updated: 31.01.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json
import numpy as np

#==============================================================================

if __name__=="__main__":
    
    # Set config filepath
    config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/config.json"
           
    # Set input filepath
    input_data_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/passage.npy"

    # Set output filepath
    output_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"
    
    # Iterate through all inputs
    for target_compression_ratio in np.power(10,np.linspace(1,3,10)):
                       
        # Define the network config
        config = {
            "network_name"              : "squashnet_" + "compressratio_" + "{:.2f}_".format(target_compression_ratio) + (input_data_path.split("/")[-1].split(".")[0]),
            "target_compression_ratio"  : target_compression_ratio,
            "hidden_layers"             : 8,
            "min_neurons_per_layer"     : 10,
            "initial_lr"                : 5e-3,
            "batch_size"                : 1024,
            "batch_fraction"            : 0.0005,
            "epochs"                    : 30,
            "half_life"                 : 3,
            "input_data_path"           : input_data_path             
            }
        
        # Save the network config
        with open(config_path,"w") as config_file: json.dump(config,config_file,indent=4,sort_keys=True)
         
        # Run the compression experiment
        runstring = "python NeurComp_SourceCode/compress_main.py" + " " + config_path + " " + input_data_path + " " + output_path
        os.system(runstring)
        
#==============================================================================
   
