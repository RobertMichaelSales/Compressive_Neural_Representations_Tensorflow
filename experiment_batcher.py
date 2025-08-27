""" Created: 10.11.2022  \\  Updated: 31.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json, glob
import numpy as np

#==============================================================================

if __name__=="__main__": 

    # Set input data config options
    input_dataset_config_paths = []
    input_dataset_config_paths += sorted(glob.glob("/Data/ISO_Compression_Datasets/jhtdb_isotropic/extracts/m/pressure/isotropic_cropped_config_clamped.json"))
    input_dataset_config_paths += sorted(glob.glob("/Data/ISO_Compression_Datasets/wheeler_dns/extracts/m/mach/block_6_config_clamped.json"))
    input_dataset_config_paths += sorted(glob.glob("/Data/ISO_Compression_Datasets/turbostream_rotor67/extracts/ar_0.900/entropy/domain_1_config_clamped.json"))
    input_dataset_config_paths += sorted(glob.glob("/Data/ISO_Compression_Datasets/owais_aorta/extracts/average_pressure/aorta_config_clamped.json"))
        
    # Filter configs
    input_dataset_config_paths = [x for x in input_dataset_config_paths  if "_config" in x]
    
    # Network config options
    target_compression_ratio_   = np.array([50])
    hidden_layers_              = np.array([4,6,8,10,12,14,16,18,20,22])
    frequencies_                = np.array([0])
    activation_                 = np.array(["sine"])
    identity_mapping_           = np.array([False,True])
    kernel_initializer_         = np.array(["glorot_uniform"])
    bits_per_weight_            = np.array([32])
        
    # Training config options
    batch_size_                 = np.array([1024])
    epochs_                     = np.array([30])
    half_life_                  = np.array([2])
    initial_lr_                 = np.array([0.001])
    normalise_data_             = np.array([True])
    weighted_error_             = np.array([True])

    # Additional batch number
    batch_number                = np.array([2048])
    
    # Set experiment number and description
    campaign_number = 1
    campaign_detail = "ABLATION_HL_IM"        
    save_renders_flag = True
    
    # Compute experiment counter and totals
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
                                                            experiment_name = "C{:03d}-{:}".format(campaign_number,campaign_detail)  
                                                            print("\n"); print("*"*80); print(experiment_name + " ({:03d}/{:03d})".format(count+1,total)); print("*"*80)
                                                            
                                                            # Set network name
                                                            network_name = "HL_{:02d}{}".format(hidden_layers,"_IM" if identity_mapping else "")
                                                            
                                                            # Define the dataset config
                                                            with open(input_dataset_config_path) as input_dataset_config_file: dataset_config = json.load(input_dataset_config_file)

                                                            # Overwrite the batch sizes
                                                            if batch_number: batch_size = int(np.ceil(np.prod([dataset_config["shape"][i] for i in dataset_config["columns"][0]]) / batch_number))
                                                            
                                                            # Define the network config
                                                            network_config = {
                                                                "network_name"              : str(network_name),
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
                                                            o_filepath = os.path.join("/Data/ISO_Compression_Experiments/",experiment_name,os.path.join(*dataset_config["i_filepath"].split("/")[3:]).replace(".npy",""),network_name)
                                                            
                                                            # Run the compression experiment
                                                            runstring = "python SourceCode/compress_main.py " + "'" + json.dumps(network_config) + "' '" + json.dumps(dataset_config) + "' '" + json.dumps(runtime_config) + "' '" + json.dumps(training_config) + "' '" + o_filepath + "'"
                                                            os.system(runstring)
                                                            
                                                            # Render outputs
                                                            runstring = "pvpython ParaView/render_outputs.py " + o_filepath + " " + str(save_renders_flag)
                                                            os.system(runstring)
                                                            
                                                            # Assess outputs
                                                            runstring = "python ParaView/assess_outputs.py " + o_filepath                                                           
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
   
