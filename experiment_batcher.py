""" Created: 10.11.2022  \\  Updated: 20.04.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json, sys
import numpy as np

#==============================================================================

if __name__=="__main__": 
    
    # Set experiment inputs 
    if (len(sys.argv) == 1):
        input_dataset_config_path = "/Data/Compression_Datasets/jhtdb_isotropic1024coarse_pressure/snips/jhtdb_isotropic1024coarse_pressure_snip8_config.json"
    else:
        input_dataset_config_path = sys.argv[1] 
    ##
    with open(input_dataset_config_path) as input_dataset_config_file: dataset_config = json.load(input_dataset_config_file)
    
    # Set experiment outputs
    config_dir_filepath  = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/"
    network_config_path  = config_dir_filepath + "network_config.json"
    dataset_config_path  = config_dir_filepath + "dataset_config.json"
    runtime_config_path  = config_dir_filepath + "runtime_config.json"
    training_config_path = config_dir_filepath + "training_config.json"
    with open(dataset_config_path,"w") as dataset_config_file: json.dump(dataset_config,dataset_config_file,indent=4,sort_keys=True)
    
    # Set experiment id number 
    experiment_id = 1
    
    # Set counter and total
    counter, total = 1, (3*7*7)
    
    # Iterate through all inputs
    for compression_ratio in np.power(10,np.linspace(np.log10(10),np.log10(1000),3)):
        
        for learning_rate in np.power(10,np.linspace(np.log10(1e-7),np.log10(1e-1),7)):
            
            for batch_fraction in np.power(10,np.linspace(np.log10(1e-4),np.log10(1e-2),7)):           
         
                # Set experiment campaign name
                campaign_name = "exp{:04d}_cr{:011.6f}_lr{:11.9f}_bf{:11.9f}".format(experiment_id,compression_ratio,learning_rate,batch_fraction)    
                    
                # Print this experiment number
                print("\n")
                print("*"*80)
                print("Experiment {}/{}: '{}'".format(counter,total,campaign_name))
                print("*"*80)
                print("\n")
                
                # Define the network config
                network_config = {
                    "network_name"              : campaign_name,
                    "hidden_layers"             : 14,
                    "frequencies"               : 0,
                    "target_compression_ratio"  : compression_ratio,
                    "minimum_neurons_per_layer" : 1,
                    }
                
                # Save the network config
                with open(network_config_path,"w") as network_config_file: 
                    json.dump(network_config,network_config_file,indent=4,sort_keys=True)
                ##
                                       
                # Define the runtime config
                runtime_config = {
                    "print_verbose"             : False,
                    "ensemble_flag"             : False,
                    "save_network_flag"         : False,
                    "save_outputs_flag"         : False,
                    "save_results_flag"         : True
                    }
                
                # Save the runtime config
                with open(runtime_config_path,"w") as runtime_config_file: 
                    json.dump(runtime_config,runtime_config_file,indent=4,sort_keys=True)
                ##
                
                # Define the training config
                training_config = {
                    "initial_lr"                : learning_rate,
                    "batch_size"                : 1024,
                    "batch_fraction"            : batch_fraction,
                    "epochs"                    : 30,
                    "half_life"                 : 2,            
                    }            
                
                # Save the training config
                with open(training_config_path,"w") as training_config_file: 
                    json.dump(training_config,training_config_file,indent=4,sort_keys=True)
                ##
                
                # Define the output directory
                o_filepath = dataset_config["i_filepath"].replace("Datasets","Experiments").replace(".npy","TEST")
                
                # Run the compression experiment
                runstring = "python NeurComp_SourceCode/compress_main.py " + "'" + json.dumps(network_config) + "' '" + json.dumps(dataset_config) + "' '" + json.dumps(runtime_config) + "' '" + json.dumps(training_config) + "' '" + o_filepath + "'"
                
                print(runstring,"\n"*5)
                
                # os.system(runstring)
                counter = counter + 1 
                
                if counter > 2: raise SystemError
            ##
        ##
    ##
##
        
#==============================================================================
   
