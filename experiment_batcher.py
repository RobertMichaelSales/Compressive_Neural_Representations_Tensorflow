""" Created: 10.11.2022  \\  Updated: 31.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json, glob
import numpy as np

#==============================================================================

if __name__=="__main__": 

    # Set input data config options
    input_dataset_config_paths = sorted(glob.glob("/Data/ISO_Compression_Datasets/sales_nodule/nodule_data_config.json"))
        
    # Filter configs
    input_dataset_config_paths = [x for x in input_dataset_config_paths  if "_config" in x or "_config" in x]
    
    # Set test option sweep
    compression_ratios = np.array([100])
    bits_per_neurons = np.array([32])
    learning_rates = np.array([1e-3])
    batch_fractions = np.array([0])
    frequenciess = np.array([0])
    hidden_layerss = np.array([10])
    
    # Set experiment number
    experiment_num = 0
    
    # Set counter and total
    count = 1
    total = len(input_dataset_config_paths) * len(compression_ratios) * len(bits_per_neurons) * len(learning_rates) * len(batch_fractions) * len(frequenciess) * len(hidden_layerss)
        
    # Iterate through all inputs
    for input_dataset_config_path in input_dataset_config_paths:
    
        for compression_ratio in compression_ratios:
            
            for bits_per_neuron in bits_per_neurons:
            
                for learning_rate in learning_rates:
                    
                    for batch_fraction in batch_fractions:     
                        
                        for frequencies in frequenciess:
                            
                            for hidden_layers in hidden_layerss:
                 
                                # Set experiment campaign name
                                campaign_name = "exp{:03d}_cr{:011.6f}_lr{:11.9f}_bf{:11.9f}_fr{:03d}_hl{:03d}_bn{:03d}".format(experiment_num,compression_ratio,learning_rate,batch_fraction,frequencies,hidden_layers,bits_per_neuron)    
                                
                                # Print this experiment number
                                print("\n");print("*"*80);print("Experiment {}/{}: '{}'".format(count,total,campaign_name));print("*"*80);print("\n")
                                
                                # Define the dataset config
                                with open(input_dataset_config_path) as input_dataset_config_file: dataset_config = json.load(input_dataset_config_file)
                                
                                # Define the network config
                                network_config = {
                                    "network_name"              : str(campaign_name),
                                    "hidden_layers"             : int(hidden_layers),
                                    "frequencies"               : int(frequencies),
                                    "target_compression_ratio"  : float(compression_ratio),
                                    "bits_per_neuron"           : int(bits_per_neuron),
                                    }
                                                       
                                # Define the runtime config
                                runtime_config = {
                                    "cache_dataset"             : bool(False),
                                    "print_verbose"             : bool(False),
                                    "ensemble_flag"             : bool(False),
                                    "shuffle_dataset"           : bool(True),
                                    "save_network_flag"         : bool(True),
                                    "save_outputs_flag"         : bool(True),
                                    "save_results_flag"         : bool(True),
                                    "save_spectra_flag"         : bool(True),
                                    }
                                
                                # Define the training config
                                training_config = {
                                    "initial_lr"                : float(learning_rate),
                                    "batch_size"                : int(4096),
                                    "batch_fraction"            : float(batch_fraction),
                                    "epochs"                    : int(30),
                                    "half_life"                 : int(2),            
                                    }            
                                
                                # Define the output directory
                                o_filepath = "/Data/ISO_Compression_Experiments/nir_experiments/" + os.path.join(*dataset_config["i_filepath"].split("/")[3:]).replace(".npy","")
                                
                                # Run the compression experiment
                                runstring = "python SourceCode/compress_main.py " + "'" + json.dumps(network_config) + "' '" + json.dumps(dataset_config) + "' '" + json.dumps(runtime_config) + "' '" + json.dumps(training_config) + "' '" + o_filepath + "'"
                                os.system(runstring)
                                
                                # Define the plotting config
                                plotting_config = {
                                    "filepath"                  : os.path.join(o_filepath,campaign_name),
                                    "render_isom"               : True,
                                    "render_orth"               : True,
                                    "render_zoom"               : float(1.0),                           
                                    }                            
                                
                                # Render the results in ParaView
                                runstring = "pvpython ParaView/plot_volumes.py " + "'" + json.dumps(plotting_config) + "'"
                                os.system(runstring)
                                
                                count = count + 1 
                                
                            ##
                        ##
                    ##
                ##
            ##
        ##
    ##
##
        
#==============================================================================
   
