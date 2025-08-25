""" Created: 10.11.2022  \\  Updated: 31.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json, glob
import numpy as np

#==============================================================================

if __name__=="__main__": 

    # Set input data config options
    input_dataset_config_paths = sorted(glob.glob("/Data/ISO_Compression_Datasets/owais_aorta/aorta_average_speed_config_clamped.json"))
        
    # Filter configs
    input_dataset_config_paths = [x for x in input_dataset_config_paths  if "_config" in x]
    
    # Network config options
    target_compression_ratio_   = np.array([10])
    hidden_layers_              = np.array([10])
    frequencies_                = np.array([0])
    activation_                 = np.array(["sine"])
    identity_mapping_           = np.array([True])
    kernel_initializer_         = np.array(["glorot_uniform"])
    bits_per_weight_            = np.array([32])
        
    # Training config options
    batch_size_                 = np.array([128])
    epochs_                     = np.array([40])
    half_life_                  = np.array([3])
    initial_lr_                 = np.array([0.001])
    normalise_data_             = np.array([True])
    weighted_error_             = np.array([True])
    
    # Set experiment number and description
    campaign_number = 0
    campaign_detail = "ABLATION_HIDDEN_LAYERS"        
    count = 0
    total = len(input_dataset_config_paths)*len(target_compression_ratio_)*len(hidden_layers_)*len(frequencies_)*len(activation_)*len(identity_mapping_)*len(kernel_initializer_)*len(bits_per_weight_)*len(batch_size_)*len(epochs_)*len(half_life_)*len(initial_lr_)*len(normalise_data_)*len(weighted_error_)
        
    # Iterate through all inputs
    for input_dataset_config_path in input_dataset_config_paths:
    
        for target_compression_ratio in target_compression_ratio_:
            
            for hidden_layers in hidden_layers_:
            
                for frequencies in frequencies_:
                    
                    for activation in activation_:     
                        
                        for identity_mapping in identity_mapping_:
                            
                            for kernel_initializer in kernel_initializer_:
                                
                                for bits_per_weight in bits_per_weight_:
                                    
                                    for batch_size in batch_size_:
                                        
                                        for epochs in epochs_:
                                            
                                            for half_life in half_life_:
                                                
                                                for initial_lr in initial_lr_:
                                                    
                                                    for normalise_data in normalise_data_:
                                                        
                                                        for weighted_error in weighted_error_:
                                                                                            
                                                            # Set experiment number
                                                            experiment_name = "CAMPAIGN_{:03d}_EXPERIMENT_{:03d}_[{:}]".format(campaign_number,count,campaign_detail)  
                                                            print("\n"); print("*"*80); print(experiment_name); print("*"*80); print("\n")
                                                            
                                                            # Define the dataset config
                                                            with open(input_dataset_config_path) as input_dataset_config_file: dataset_config = json.load(input_dataset_config_file)
                                                            
                                                            # Define the network config
                                                            network_config = {
                                                                "network_name"              : str(experiment_name),
                                                                "description"               : str(campaign_detail),
                                                                "target_compression_ratio"  : float(target_compression_ratio),
                                                                "hidden_layers"             : int(hidden_layers),
                                                                "frequencies"               : int(frequencies),
                                                                "activation"                : str(activation),
                                                                "identity_mapping"          : bool(identity_mapping),
                                                                "kernel_initializer"        : str(kernel_initializer),
                                                                "bits_per_weight"           : int(bits_per_weight),
                                                                }
                                                                                                                
                                                            # Define the runtime config
                                                            runtime_config = {
                                                                "print_verbose"             : bool(False),
                                                                "cache_dataset"             : bool(False),
                                                                "save_network_flag"         : bool(True),
                                                                "save_outputs_flag"         : bool(True),
                                                                "save_results_flag"         : bool(True),
                                                                }
                                                            
                                                            # Define the training config
                                                            training_config = {
                                                                "batch_size"                : int(batch_size),
                                                                "epochs"                    : int(epochs),
                                                                "half_life"                 : int(half_life),            
                                                                "initial_lr"                : float(initial_lr),
                                                                "normalise_data"            : bool(normalise_data),
                                                                "weighted_error"            : bool(weighted_error),
                                                                }            
                                                            
                                                            # Define the output directory
                                                            o_filepath = "/Data/ISO_Compression_Experiments/" + os.path.join(*dataset_config["i_filepath"].split("/")[3:]).replace(".npy","")
                                                            
                                                            # Run the compression experiment
                                                            runstring = "python SourceCode/compress_main.py " + "'" + json.dumps(network_config) + "' '" + json.dumps(dataset_config) + "' '" + json.dumps(runtime_config) + "' '" + json.dumps(training_config) + "' '" + o_filepath + "'"
                                                            os.system(runstring)
                                                            
                                                            # Render outputs
                                                            runstring = "pvpython ParaView/render_outputs.py " + os.path.join(o_filepath,experiment_name)
                                                            os.system(runstring)
                                                            
                                                            # Assess outputs
                                                            runstring = "python ParaView/assess_outputs.py " + os.path.join(o_filepath,experiment_name)
                                                            os.system(runstring)
                                                            
                                                            # Iterate counter
                                                            count = count + 1 
                                                            
                                                        ##  
                                                    ##
                                                ##
                                            ##
                                        ##
                                    ##
                                ##
                            ##
                        ##
                    ##
                ##
            ##
        ##
    ##
##
        
#==============================================================================
   
