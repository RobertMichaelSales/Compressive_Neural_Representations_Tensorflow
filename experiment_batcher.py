""" Created: 10.11.2022  \\  Updated: 05.01.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json

#==============================================================================

if __name__=="__main__":
    
    # Set config filepath
    config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/config.json"
    
    # Set initial task id
    task_id_start = 0
       
    # Iterate through all input pairs
    for task_id,target_compression_ratio in enumerate([10,25,50,75,100,150,200,300,400,500,750,1000]):
        
        # Offset the task id 
        task_id = task_id + task_id_start
                       
        # Define the network config
        config = {
            "network_name"              : "squashnet_"+str(task_id),
            "target_compression_ratio"  : target_compression_ratio,
            "hidden_layers"             : 8,
            "min_neurons_per_layer"     : 10,
            "initial_lr"                : 5e-3,
            "batch_size"                : 1024,
            "epochs"                    : 30,
            "half_life"                 : 3                        
            }
        
        # Save the network config
        with open(config_path,"w") as config_file: json.dump(config,config_file,indent=4,sort_keys=True)
         
        # Run the compression experiment
        os.system("python NeurComp_SourceCode/compress_main.py")
        
        
#==============================================================================


