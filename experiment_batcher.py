""" Created: 10.11.2022  \\  Updated: 31.05.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json, sys, glob
import numpy as np

#==============================================================================

if __name__=="__main__": 

    # Set input data config options
    input_dataset_config_paths = sorted(glob.glob("/Data/Compression_Datasets/jhtdb_isotropic1024coarse_pressure/snips/jhtdb_isotropic1024coarse_pressure_snip8_config.json"))
    
    # Set experiment number
    experiment_num = 1
    
    # Set counter and total
    count = 1
    total = 1*3*7*7
    
    # Iterate through all inputs
    for input_dataset_config_path in input_dataset_config_paths:
    
        for compression_ratio in np.array([1e+2],dtype="float32"):
            
            for learning_rate in np.array([1e-3],dtype="float32"):
                
                for batch_fraction in np.array([3e-4],dtype="float32"):     
                    
                    for frequencies in np.array([0],dtype="int32"):
                        
                        for hidden_layers in np.array([14],dtype="int32"):
             
                            # Set experiment campaign name
                            campaign_name = "exp{:03d}_cr{:011.6f}_lr{:11.9f}_bf{:11.9f}_fr{:03d}_hl{:03d}".format(experiment_num,compression_ratio,learning_rate,batch_fraction,frequencies,hidden_layers)    
                                
                            # Print this experiment number
                            print("\n");print("*"*80);print("Experiment {}/{}: '{}'".format(count,total,campaign_name));print("*"*80);print("\n")
                            
                            # Define the dataset config
                            with open(input_dataset_config_path) as input_dataset_config_file: dataset_config = json.load(input_dataset_config_file)
                            
                            # Define the network config
                            network_config = {
                                "network_name"              : campaign_name,
                                "hidden_layers"             : hidden_layers,
                                "frequencies"               : frequencies,
                                "target_compression_ratio"  : compression_ratio,
                                "minimum_neurons_per_layer" : 1,
                                }
                                                   
                            # Define the runtime config
                            runtime_config = {
                                "print_verbose"             : False,
                                "ensemble_flag"             : False,
                                "save_network_flag"         : False,
                                "save_outputs_flag"         : False,
                                "save_results_flag"         : True
                                }
                            
                            # Define the training config
                            training_config = {
                                "initial_lr"                : learning_rate,
                                "batch_size"                : 1024,
                                "batch_fraction"            : batch_fraction,
                                "epochs"                    : 30,
                                "half_life"                 : 2,            
                                }            
                            
                            # Define the output directory
                            o_filepath = "/Data/Compression_Experiments/" + os.path.join(*dataset_config["i_filepath"].split("/")[3:]).replace(".npy","TEST")
                            
                            # Run the compression experiment
                            runstring = "python NeurComp_SourceCode/compress_main.py " + "'" + json.dumps(network_config) + "' '" + json.dumps(dataset_config) + "' '" + json.dumps(runtime_config) + "' '" + json.dumps(training_config) + "' '" + o_filepath + "'"
                            
                            # os.system(runstring)
                            count = count + 1 
                    ##
                ##
            ##
        ##
    ##
##
        
#==============================================================================
   
