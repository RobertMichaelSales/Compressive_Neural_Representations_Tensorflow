""" Created: 10.11.2022  \\  Updated: 31.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json, glob
import numpy as np

#==============================================================================

if __name__=="__main__": 

    # Set input data config options
    input_dataset_config_paths = []
    input_dataset_config_paths += sorted(glob.glob("/Data/ISO_Compression_Datasets/jhtdb_isotropic/extracts/c/pressure/isotropic_cropped_config_clamped.json"))
    # input_dataset_config_paths += sorted(glob.glob("/Data/ISO_Compression_Datasets/jhtdb_buoyancy/extracts/m/pressure/buoyancy_cropped_config_clamped.json"))
    # input_dataset_config_paths += sorted(glob.glob("/Data/ISO_Compression_Datasets/wheeler_dns/extracts/*/mach/block_6_config_clamped.json"))
    # input_dataset_config_paths += sorted(glob.glob("/Data/ISO_Compression_Datasets/turbostream_rotor67/extracts/ar_0.900/entropy/domain_1_config_clamped.json"))
    # input_dataset_config_paths += sorted(glob.glob("/Data/ISO_Compression_Datasets/owais_aorta/extracts/average_pressure/aorta_config_clamped.json"))
    # input_dataset_config_paths += sorted(glob.glob("/Data/ISO_Compression_Datasets/nasa_ucrm/extracts/ro/ucrm_config_clamped.json"))
        
    # Filter configs
    input_dataset_config_paths  = [x for x in input_dataset_config_paths  if "_config" in x]
    
    # Network config options
    target_compression_ratio_   = np.array([100])
    hidden_layers_              = np.array([8])
    frequencies_                = np.array([10])
    activation_                 = np.array(["sine"])
    identity_mapping_           = np.array([True])
    kernel_initializer_         = np.array(["siren"])
    omega_0_                    = np.array([10])
    bits_per_weight_            = np.array([32])
        
    # Training config options
    batch_size_                 = np.array(np.power(2.0,np.array([14,12,10,8,6,4,2])))
    epochs_                     = np.array([40])
    half_life_                  = np.array([3])
    # initial_lr_                 = np.array(np.power(10.0,np.array([-1.0,-2.5,-4.0])))
    # initial_lr_                 = np.array(np.power(10.0,np.array([-1.5,-3.0,-4.5])))
    initial_lr_                 = np.array(np.power(10.0,np.array([-2.0,-3.5,-5.0])))
    normalise_data_             = np.array([True])
    weighted_error_             = np.array(["none"])

    # Additional batch number
    batch_number                = np.array([2048])
    
    # Set experiment number and description
    campaign_number = 8
    campaign_detail = "RESULTS_COMPRESS"        
    save_renders_flag = True
    
    # Compute experiment counter and totals
    count = 0
    total = len(input_dataset_config_paths)*len(target_compression_ratio_)*len(hidden_layers_)*len(frequencies_)*len(activation_)*len(identity_mapping_)*len(kernel_initializer_)*len(omega_0_)*len(bits_per_weight_)*len(batch_size_)*len(epochs_)*len(half_life_)*len(initial_lr_)*len(normalise_data_)*len(weighted_error_)
        
    # Iterate through all inputs
    for input_dataset_config_path in input_dataset_config_paths:
    
        for target_compression_ratio in target_compression_ratio_:
            
            for hidden_layers in hidden_layers_:
            
                for frequencies in frequencies_:
                    
                    for activation in activation_:     
                        
                        for identity_mapping in identity_mapping_:
                            
                            for kernel_initializer in kernel_initializer_:
                                
                                for omega_0 in omega_0_:
                                
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
                                                                network_name = "BS_{:05d}_LR_{:05f}".format(int(batch_size),initial_lr)
                                                                print(network_name)
                                                                # break

                                                                # Define the dataset config
                                                                with open(input_dataset_config_path) as input_dataset_config_file: dataset_config = json.load(input_dataset_config_file)
                                                                print(input_dataset_config_path)
    
                                                                # Overwrite the batch sizes
                                                                # if batch_number: batch_size = min(int(np.ceil(np.prod(dataset_config["shape"][:-1]) / batch_number)), batch_size)
                                                                # print(batch_size, initial_lr); break
                                                                
                                                                # Define the network config
                                                                network_config = {
                                                                    "network_name"              : str(network_name),
                                                                    "target_compression_ratio"  : float(target_compression_ratio),
                                                                    "hidden_layers"             : int(hidden_layers),
                                                                    "frequencies"               : int(frequencies),
                                                                    "activation"                : str(activation),
                                                                    "identity_mapping"          : bool(identity_mapping),
                                                                    "kernel_initializer"        : str(kernel_initializer),
                                                                    "omega_0"                   : float(omega_0),
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
                                                                    "weighted_error"            : str(weighted_error),
                                                                    }            
                                                                
                                                                # Define the output directory
                                                                o_filepath = os.path.join("/Data/ISO_Compression_Experiments/",experiment_name,os.path.join(*dataset_config["i_filepath"].split("/")[3:]).replace(".npy",""),network_name)
                                                                
                                                                # Run the compression experiment
                                                                runstring = "python SourceCode/compress_main.py " + "'" + json.dumps(network_config) + "' '" + json.dumps(dataset_config) + "' '" + json.dumps(runtime_config) + "' '" + json.dumps(training_config) + "' '" + o_filepath + "'"
                                                                os.system(runstring)
                                                                
                                                                # Render outputs
                                                                runstring = "pvpython --force-offscreen-rendering ParaView/render_outputs.py " + o_filepath + " " + str(save_renders_flag)
                                                                print("TURN RENDERING BACK ON!") # os.system(runstring)
                                                                
                                                                # Assess outputs
                                                                runstring = "python ParaView/assess_outputs.py " + o_filepath                                                           
                                                                print("TURN RENDERING BACK ON!") # os.system(runstring)
                                                                
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
##
        
#==============================================================================
   
