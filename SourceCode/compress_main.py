""" Created: 18.07.2022  \\  Updated: 29.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os, time, json, math, psutil, sys, gc, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gc.enable()

import numpy as np
import tensorflow as tf

# Set python, numpy and tensorflow random seeds for the same initialisation
import random; tf.random.set_seed(123); np.random.seed(123); random.seed(123)

tf.keras.backend.set_floatx('float32')
tf.keras.mixed_precision.set_global_policy('float32')

#==============================================================================
# Import user-defined libraries 

from data_management         import DataClass,MakeDatasetFromTensorSlc,SaveData
from network_encoder         import EncodeArchitecture,EncodeParameters,SaveNetworkJSON
from network_model           import ConstructNetwork
from configuration_classes   import GenericConfigurationClass,NetworkConfigurationClass
from compress_utilities      import TrainStep,GetLearningRate,MeanSquaredErrorMetric,Logger,SignalToNoise,QuantiseParameters

#==============================================================================

def compress(network_config,dataset_config,runtime_config,training_config,o_filepath):
        
    print("-"*80,"\nISONet: IMPLICIT NEURAL COMPRESSION OF DATASET")
    
    print("\nDateTime: {}".format(datetime.datetime.now().strftime("%d %b %Y - %H:%M:%S")))
    
    #==========================================================================
    # Check whether hardware acceleration is enabled
    print("-"*80,"\nCHECKING SYSTEM REQUIREMENTS:")
    
    print("\n{:30}{}".format("TensorFlow Version:",tf.__version__))
    gpus = tf.config.list_physical_devices('GPU')
    
    if (len(gpus) != 0): 
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Enabled") )
        tf.config.experimental.set_memory_growth(gpus[0],True)
    else:        
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Disabled"))
        raise SystemError("GPU device not found. Try restarting your system to resolve this error.")
    ##
        
    #==========================================================================
    # Check whether the input size exceeds available memory
    print("-"*80,"\nCHECKING MEMORY REQUIREMENTS:")
    
    # Get and display available memory
    available_memory = psutil.virtual_memory().available
    print("\n{:30}{:.3f} GigaBytes".format("Available Memory:",(available_memory/1e9)))
    
    # Get and display input file size
    input_file_size = os.path.getsize(dataset_config.i_filepath)
    print("\n{:30}{:06.3f} GigaBytes".format("Input File Size:",(input_file_size/1e9)))
    
    if (input_file_size > available_memory):
        raise MemoryError("Dataset size exceeds available RAM. Execution was halted to prevent crashing.")
    ##

    #==========================================================================
    # Initialise i/o 
    print("-"*80,"\nINITIALISING INPUTS:")
    
    # Remove reference to scales column if "weighted_error" is not specified
    if not training_config.weighted_error: dataset_config.columns[2].clear()

    # Create instance of 'DataClass' object to store the input coords, load and normalise
    print("\n{:30}{}".format("Loading Coords:",dataset_config.i_filepath.split("/")[-1]))
    print("{:30}{}".format("Fields:",dataset_config.columns[0]))
    i_coords = DataClass(data_type="coords",tabular=dataset_config.tabular)
    i_coords.LoadData(input_data_path=dataset_config.i_filepath,columns=dataset_config.columns,normalise=training_config.normalise_data)
    
    # Create instance of 'DataClass' object to store the input values, load and normalise
    print("\n{:30}{}".format("Loading Coords:",dataset_config.i_filepath.split("/")[-1]))
    print("{:30}{}".format("Fields:",dataset_config.columns[1]))
    i_values = DataClass(data_type="values",tabular=dataset_config.tabular)
    i_values.LoadData(input_data_path=dataset_config.i_filepath,columns=dataset_config.columns,normalise=training_config.normalise_data)

    # Create instance of 'DataClass' object to store the input scales, for training loss
    print("\n{:30}{}".format("Loading Weights:",dataset_config.i_filepath.split("/")[-1]))
    print("{:30}{}".format("Fields:",dataset_config.columns[2]))
    i_scales = DataClass(data_type="scales",tabular=dataset_config.tabular)
    i_scales.LoadData(input_data_path=dataset_config.i_filepath,columns=dataset_config.columns,normalise=training_config.normalise_data)

    # Create instance of 'DataClass' object to store the output coords, copies input data
    o_coords = DataClass(data_type="coords",tabular=dataset_config.tabular)
    o_coords.CopyData(DataClassObject=i_coords,exception_keys=[])
    
    # Create instance of 'DataClass' object to store the output values, copies input data
    o_values = DataClass(data_type="values",tabular=dataset_config.tabular)
    o_values.CopyData(DataClassObject=i_values,exception_keys=["flat","data"])  
                
    #==========================================================================
    # Configure dataset
    print("-"*80,"\nCONFIGURING DATASET:")
    
    print("\n{:30}{}{}".format("Batch:","batch_size = ",training_config.batch_size))
    print("\n{:30}{}{}".format("Cache:","cache_data = ",runtime_config.cache_dataset))
    
    # Generate a TF dataset to supply coords and values batches during training 
    dataset = MakeDatasetFromTensorSlc(coords=i_coords,values=i_values,scales=i_scales,batch_size=training_config.batch_size,cache_dataset=runtime_config.cache_dataset)
    
    #==========================================================================
    # Configure network 
    print("-"*80,"\nCONFIGURING NETWORK:")
    
    # Generate the network structure based on the input dimensions
    network_config.GenerateStructure(i_dimensions=i_coords.dimensions,o_dimensions=i_values.dimensions,size=i_values.size)
    
    # Build ISONet from the config information
    ISONet = ConstructNetwork(layer_dimensions=network_config.layer_dimensions,frequencies=network_config.frequencies,activation=network_config.activation,kernel_initializer=network_config.kernel_initializer,identity_mapping=network_config.identity_mapping)
    
    # Add original bounds to network attributes
    ISONet.original_values_bounds = i_values.original_bounds
    ISONet.original_coords_centre = i_coords.original_centre
    ISONet.original_coords_radius = i_coords.original_radius

    # Set a training optimiser
    optimiser = tf.keras.optimizers.Adam()
    
    # Set a performance metric (custom weighted mean-squared error metric)
    metric = MeanSquaredErrorMetric()
        
    # Load the training step function as a tf.function for speed increases
    TrainStepTFF = tf.function(TrainStep)
        
    # Save an image of the network graph (helpful to check)
    tf.keras.utils.plot_model(model=ISONet,to_file=os.path.join(o_filepath,"iso_network_graph.png"))

    #==========================================================================  
    # Training loop
    print("-"*80,"\nCOMPRESSING ISO:")
        
    # Create a dictionary of lists to store training data
    training_data = {"epoch":[],"error":[],"time":[],"learning_rate":[],"weighted_psnr":[],"unscaled_psnr":[]}

    # Start the overall training timer
    training_time_tick = time.time()
    
    # Iterate through each epoch
    for epoch in range(training_config.epochs):
        
        print("\n",end="")
                        
        # Store and print the current epoch number
        training_data["epoch"].append(float(epoch))
        print("{:30}{}/{}".format("Epoch:",(epoch+1),training_config.epochs))
        
        # Determine, update, store and print the learning rate 
        learning_rate = GetLearningRate(initial_lr=training_config.initial_lr,half_life=training_config.half_life,epoch=epoch)
        optimiser.lr.assign(learning_rate)
        training_data["learning_rate"].append(float(learning_rate))   
        print("{:30}{:.3E}".format("Learning rate:",learning_rate))
        
        # Start timing current epoch
        epoch_time_tick = time.time()

        # Iterate through each batch
        for batch, (coords_batch,values_batch,scales_batch) in enumerate(dataset):
            
            # Print the current batch number and run a training step
            if runtime_config.print_verbose: print("\r{:30}{:04}/{:04}".format("Batch number:",(batch+1),dataset.size),end="") 
            TrainStepTFF(model=ISONet,optimiser=optimiser,metric=metric,coords_batch=coords_batch,values_batch=values_batch,scales_batch=scales_batch)

            ## Break training loop when a whole batch was trained on
            if batch >= dataset.size: break
            
        ##
        
        print("\n",end="")
        
        # End the epoch time and store the elapsed time 
        epoch_time_tock = time.time() 
        epoch_time = float(epoch_time_tock-epoch_time_tick)
        training_data["time"].append(epoch_time)
        print("{:30}{:.2f} seconds".format("Epoch time:",epoch_time))
        
        # Fetch, store and reset and the training error
        error = float(metric.result().numpy())
        metric.reset_states()
        training_data["error"].append(error)
        print("{:30}{:.7f}".format("Mean squared error:",error))     
        
        # Early stopping for diverging training results
        if np.isnan(error): 
            print("{:30}{:}".format("Early stopping:","Error has diverged")) 
            runtime_config.save_network_flag = False
            runtime_config.save_outputs_flag = False
            break
        ##
    
        # Make a new shuffled dataset instance and garbage collect memory leaks
        if (epoch != (training_config.epochs-1)):
            del(dataset); gc.collect()
            dataset = MakeDatasetFromTensorSlc(coords=i_coords,values=i_values,scales=i_scales,batch_size=training_config.batch_size,cache_dataset=runtime_config.cache_dataset)
            print("\n{:30}".format("Reshuffling dataset"))
        ##
   
    ##   
 
    # End the overall training timer
    training_time_tock = time.time()
    training_time = float(training_time_tock-training_time_tick)
    print("\n{:30}{:.2f} seconds".format("Training duration:",training_time))  
    
    # Generate the predicted output coords
    o_values.flat = ISONet.predict(o_coords.flat,batch_size=training_config.batch_size,verbose="1")
    o_values.data = np.reshape(o_values.flat,i_values.data.shape,order="F")
    
    # Calculate and report weighted PSNR
    weighted_psnr = SignalToNoise(true=i_values.flat,pred=o_values.flat,scales=i_scales.flat)
    training_data["weighted_psnr"].append(weighted_psnr)
    print("{:30}{:.3f}".format("Output weighted PSNR:",weighted_psnr))

    # Calculate and report unweighted PSNR
    unscaled_psnr = SignalToNoise(true=i_values.flat,pred=o_values.flat,scales=np.ones_like(o_values.flat))
    training_data["unscaled_psnr"].append(unscaled_psnr)
    print("{:30}{:.3f}".format("Output unscaled PSNR:",unscaled_psnr))
    
    #==========================================================================
    # Quantise parameters and recompute output
    
    if (network_config.bits_per_weight != 32): 
        
        print("-"*80,"\nQUANTISING PARAMETERS:")
        print("\n",end="")
        
        # Quantise network parameters according to bits per weight
        print("{:30}{:}".format("Quantisation factor:","bits_per_weight = {}".format(network_config.bits_per_weight)))
        quantised_parameters = QuantiseParameters(ISONet.get_weights(),network_config.bits_per_weight)
        ISONet.set_weights(quantised_parameters)
    
        # Regenerate the predicted output coords
        o_values.flat = ISONet.predict(o_coords.flat,batch_size=training_config.batch_size,verbose="1")
        o_values.data = np.reshape(o_values.flat,i_values.data.shape,order="F")
        
        # Recalculate and report predicted PSNR
        peak_signal_to_noise_ratio = SignalToNoise(true=i_values.flat,pred=o_values.flat,scales=i_scales.flat)
        training_data["psnr"].append(peak_signal_to_noise_ratio)
        print("{:30}{:.3f}".format("Output coords PSNR:",peak_signal_to_noise_ratio))
    ##    
    
    #==========================================================================
    # Save network parameters and architecture
    
    if runtime_config.save_network_flag:
        
        print("-"*80,"\nSAVING NETWORK:")
        print("\n",end="")
                
        # Save both the architecture and parameters to JSON 
        network_data_path = os.path.join(o_filepath,"network_data_iso.json")
        SaveNetworkJSON(network=ISONet,network_data_path=network_data_path)
        print("{:30}{}".format("Saved network data to:",network_data_path.split("/")[-1]))
        
        # Encode the parameters to binary
        parameters_path = os.path.join(o_filepath,"parameters.bin")
        EncodeParameters(network=ISONet,parameters_path=parameters_path)
        print("{:30}{}".format("Encoded parameters to:",parameters_path.split("/")[-1]))
        
        # Encode the architecture to binary
        architecture_path = os.path.join(o_filepath,"architecture.bin")
        EncodeArchitecture(network=ISONet,architecture_path=architecture_path)
        print("{:30}{}".format("Encoded architecture to:",architecture_path.split("/")[-1]))
        
    ##
    
    #==========================================================================
    # Save the compression results and configs
    
    if runtime_config.save_results_flag:
        
        print("-"*80,"\nSAVING RESULTS:")        
        print("\n",end="")

        # Pack all configuration dictionaries together
        combined_config = (network_config | training_config | runtime_config | dataset_config)
        
        # Save the training data
        training_data_path = os.path.join(o_filepath,"training_metadata.json")
        with open(training_data_path,"w") as file: json.dump(training_data,file,indent=4,sort_keys=True)
        print("{:30}{}".format("Saved training data to:",training_data_path.split("/")[-1]))
    
        # Save the configuration
        combined_config_path = os.path.join(o_filepath,"config.json")
        with open(combined_config_path,"w") as file: json.dump(combined_config,file,indent=4)
        print("{:30}{}".format("Saved configuration to:",combined_config_path.split("/")[-1]))
        
    else: pass

    #==========================================================================
    # Save input and predictions (.npy & .vtk)
    
    if runtime_config.save_outputs_flag:
        
        print("-"*80,"\nSAVING OUTPUTS:")
        print("\n",end="")
                
        # Save i_coords and i_values to ".npy" and ".vtk" files
        output_data_path = os.path.join(o_filepath,"i_dataset")
        SaveData(output_data_path=output_data_path,template_vtk_path=dataset_config.vtk_filepath,coords=i_coords,values=i_values,normalise=training_config.normalise_data)
        print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1])) 
        
        # Save o_coords and o_values to ".npy" and ".vtk" files
        output_data_path = os.path.join(o_filepath,"o_dataset")
        SaveData(output_data_path=output_data_path,template_vtk_path=dataset_config.vtk_filepath,coords=o_coords,values=o_values,normalise=training_config.normalise_data)
        print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1]))        
        
    ##

    #==========================================================================
    
    print("-"*80,"\n")
    
    return ISONet

##
       
#==============================================================================
# Define the main function to run when file is invoked from within the terminal

if __name__=="__main__":
    
    if (len(sys.argv) == 1):
        
        #======================================================================
        # This block will run in the event that this script is called in an IDE
    
        network_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/AuxFiles/inputs/configs/network_config.json"
        
        with open(network_config_path) as network_config_file: 
            network_config_dictionary = json.load(network_config_file)
            network_config = NetworkConfigurationClass(network_config_dictionary)
        ##   
        
        dataset_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/AuxFiles/inputs/configs/dataset_config.json"
        
        with open(dataset_config_path) as dataset_config_file: 
            dataset_config_dictionary = json.load(dataset_config_file)
            dataset_config = GenericConfigurationClass(dataset_config_dictionary)
        ##  
            
        runtime_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/AuxFiles/inputs/configs/runtime_config.json"
        
        with open(runtime_config_path) as runtime_config_file: 
            runtime_config_dictionary = json.load(runtime_config_file)
            runtime_config = GenericConfigurationClass(runtime_config_dictionary)
        ##    
            
        training_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/AuxFiles/inputs/configs/training_config.json"
        
        with open(training_config_path) as training_config_file: 
            training_config_dictionary = json.load(training_config_file)
            training_config = GenericConfigurationClass(training_config_dictionary)
        ##
        
        o_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/AuxFiles/outputs"
        
    else: 

        #======================================================================
        # This block will run in the event that this script is run via terminal        

        network_config  = NetworkConfigurationClass(json.loads(sys.argv[1]))
    
        dataset_config  = GenericConfigurationClass(json.loads(sys.argv[2]))
    
        runtime_config  = GenericConfigurationClass(json.loads(sys.argv[3]))
       
        training_config = GenericConfigurationClass(json.loads(sys.argv[4]))
        
        o_filepath      = sys.argv[5]
        
    #==========================================================================
    
    # Construct the output filepath
    o_filepath = os.path.join(o_filepath,network_config.network_name)
    if not os.path.exists(o_filepath): os.makedirs(o_filepath)
        
    # Create checkpoint and stdout logging files in case execution fails
    checkpoint_filename = os.path.join(o_filepath,"checkpoint.txt")
    stdout_log_filename = os.path.join(o_filepath,"stdout_log.txt")
    
    # Check if the checkpoint file already exists
    if not os.path.exists(checkpoint_filename): 
        
        # Start logging all console i/o
        sys.stdout = Logger(stdout_log_filename)   
    
        # Execute compression
        ISONet = compress(network_config=network_config,dataset_config=dataset_config,runtime_config=runtime_config,training_config=training_config,o_filepath=o_filepath)           
        
        # Create a checkpoint file after successful execution
        with open(checkpoint_filename, mode='w'): pass

    else: print("Checkpoint file '{}' already exists: skipping.".format(checkpoint_filename))
        
else: pass

#==============================================================================