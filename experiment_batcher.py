""" Created: 10.11.2022  \\  Updated: 17.07.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json, sys, glob
import numpy as np

#==============================================================================

if __name__=="__main__": 

    # Set input data config options
    input_dataset_config_paths = sorted(glob.glob("/Data/Compression_Datasets/sales_isotropic1024coarse_pressure/3D/*/*_config.json"))
            
    # Set experiment number
    experiment_num = 1
    
    # Set counter and total
    count = 1
    total = 78*1*1*1*1*1*1
        
    # Iterate through all inputs
    for input_dataset_config_path in input_dataset_config_paths:
    
        for compression_ratio in np.array([25]):
            
            for bits_per_neuron in np.array([32]):
            
                for learning_rate in np.array([1e-3]):
                    
                    for batch_fraction in np.array([0]):     
                        
                        for frequencies in np.array([0]):
                            
                            for hidden_layers in np.array([14]):
                 
                                # Set experiment campaign name
                                campaign_name = "exp{:03d}_cr{:011.6f}_lr{:11.9f}_bf{:11.9f}_fr{:03d}_hl{:03d}_bn{:03d}".format(experiment_num,compression_ratio,learning_rate,batch_fraction,frequencies,hidden_layers,bits_per_neuron)    
                                
                                # Print this experiment number
                                print("\n");print("*"*80);print("Experiment {}/{}: '{}'".format(count,total,campaign_name));print("*"*80);print("\n")
                                
                                # Define the dataset config
                                with open(input_dataset_config_path) as input_dataset_config_file: dataset_config = json.load(input_dataset_config_file)
                                
                                # Define the network config
                                network_config = {
                                    "network_name"              : campaign_name,
                                    "hidden_layers"             : int(hidden_layers),
                                    "frequencies"               : int(frequencies),
                                    "target_compression_ratio"  : float(compression_ratio),
                                    "minimum_neurons_per_layer" : 1,
                                    "bits_per_neuron"           : int(bits_per_neuron),
                                    }
                                                       
                                # Define the runtime config
                                runtime_config = {
                                    "cache_dataset"             : False,
                                    "print_verbose"             : False,
                                    "ensemble_flag"             : False,
                                    "shuffle_dataset"           : True,
                                    "save_network_flag"         : False,
                                    "save_outputs_flag"         : True,
                                    "save_results_flag"         : True,
                                    "save_spectra_flag"         : True,
                                    }
                                
                                # Define the training config
                                training_config = {
                                    "initial_lr"                : float(learning_rate),
                                    "batch_size"                : 4096,
                                    "batch_fraction"            : float(batch_fraction),
                                    "epochs"                    : 30,
                                    "half_life"                 : 2,            
                                    }            
                                
                                # Define the output directory
                                o_filepath = "/Data/Compression_Experiments/nir_experiments/" + os.path.join(*dataset_config["i_filepath"].split("/")[3:]).replace(".npy","")
                                
                                # Run the compression experiment
                                runstring = "python NeurComp_SourceCode/compress_main.py " + "'" + json.dumps(network_config) + "' '" + json.dumps(dataset_config) + "' '" + json.dumps(runtime_config) + "' '" + json.dumps(training_config) + "' '" + o_filepath + "'"
                                os.system(runstring)
                                
                                # Define the plotting config
                                plotting_config = {
                                    "filepath"                  : os.path.join(o_filepath,campaign_name),
                                    "render_isom"               : True,
                                    "render_orth"               : True,
                                    "render_zoom"               : float(1.0),                           
                                    }                            
                                
                                # Render the results in ParaView
                                runstring = "pvpython ParaView_SourceCode/plot_volumes.py " + "'" + json.dumps(plotting_config) + "'"
                                # os.system(runstring)
                                
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
   
