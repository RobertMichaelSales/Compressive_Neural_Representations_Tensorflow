""" Created: 18.07.2022  \\  Updated: 19.04.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os, time, json, math, psutil, sys, gc, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gc.enable()

import numpy as np
import tensorflow as tf

#==============================================================================
# Import user-defined libraries 

from data_complexity         import SaveSpectrum
from data_management         import DataClass,MakeDatasetFromTensorSlc,MakeDatasetFromGenerator,SaveData
from network_encoder         import EncodeParameters,EncodeArchitecture
from network_model           import ConstructNetwork
from configuration_classes   import NetworkConfigurationClass,GenericConfigurationClass
from compress_utilities      import TrainStep,SignalToNoise,GetLearningRate,MeanSquaredErrorMetric,Logger,QuantiseParameters

#==============================================================================

def compress(network_config,dataset_config,runtime_config,training_config,o_filepath,index,ensemble_output):
        
    print("-"*80,"\nSQUASHNET Mini: IMPLICIT NEURAL REPRESENTATIONS (by Rob Sales)")
    
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
    
    # Get and display available memory (hardware limit)
    available_memory = psutil.virtual_memory().available
    print("\n{:30}{:.3f} GigaBytes".format("Available Memory:",(available_memory/1e9)))
    
    # Set and display threshold memory (software limit)
    threshold_memory = int(20*1e9)
    print("\n{:30}{:.3f} GigaBytes".format("Threshold Memory:",(threshold_memory/1e9)))
    
    # Get and display input file size
    input_file_size = os.path.getsize(dataset_config.i_filepath)
    print("\n{:30}{:06.3f} GigaBytes".format("Input File Size:",(input_file_size/1e9)))
    
    # Determine whether data exceeds memory and choose how to load the dataset
    dataset_exceeds_memory = (input_file_size > min(available_memory,threshold_memory))
    
    #==========================================================================
    # Initialise i/o 
    print("-"*80,"\nINITIALISING INPUTS:")
    
    # Create 'DataClass' objects to store the input volume, load and normalise
    i_volume = DataClass(data_type="volume",tabular=dataset_config.tabular,exceeds_memory=dataset_exceeds_memory)
    i_volume.LoadData(input_data_path=dataset_config.i_filepath,columns=dataset_config.columns,shape=dataset_config.shape,dtype=dataset_config.dtype,normalise=dataset_config.normalise)
    
    # Create 'DataClass' objects to store the input values, load and normalise
    i_values = DataClass(data_type="values",tabular=dataset_config.tabular,exceeds_memory=dataset_exceeds_memory)
    i_values.LoadData(input_data_path=dataset_config.i_filepath,columns=dataset_config.columns,shape=dataset_config.shape,dtype=dataset_config.dtype,normalise=dataset_config.normalise)

    # Create 'DataClass' objects to store the input weights for training loss
    weights = DataClass(data_type="weights",tabular=dataset_config.tabular,exceeds_memory=dataset_exceeds_memory)
    weights.LoadData(input_data_path=dataset_config.i_filepath,columns=dataset_config.columns,shape=dataset_config.shape,dtype=dataset_config.dtype,normalise=dataset_config.normalise )

    # Create 'DataClass' objects to store the output volume, copy input data
    o_volume = DataClass(data_type="volume",tabular=dataset_config.tabular,exceeds_memory=dataset_exceeds_memory)
    o_volume.CopyData(DataClassObject=i_volume,exception_keys=[])
    
    # Create 'DataClass' objects to store the output values, copy input data
    o_values = DataClass(data_type="values",tabular=dataset_config.tabular,exceeds_memory=dataset_exceeds_memory)
    o_values.CopyData(DataClassObject=i_values,exception_keys=["flat","data"])  
    
    #==========================================================================
    
    # Calculate the standard deviation for the volume and values data
    volume_standard_deviation = np.std(i_volume.flat,axis=0).tolist()
    values_standard_deviation = np.std(i_values.flat,axis=0).tolist()
              
    #==========================================================================
    # Configure network 
    print("-"*80,"\nCONFIGURING NETWORK:")
    
    # Generate the network structure based on the input dimensions
    network_config.GenerateStructure(i_dimensions=i_volume.dimensions,o_dimensions=i_values.dimensions,size=i_values.size)
    
    # Build NeurComp from the config information
    SquashNet = ConstructNetwork(layer_dimensions=network_config.layer_dimensions,frequencies=network_config.frequencies)
                      
    # Set a training optimiser
    optimiser = tf.keras.optimizers.Adam()
    
    # Set a performance metric (custom weighted mean-squared error metric)
    metric = MeanSquaredErrorMetric()
        
    # Load the training step function as a tf.function for speed increases
    TrainStepTFF = tf.function(TrainStep)
        
    # Save an image of the network graph (helpful to check)
    tf.keras.utils.plot_model(model=SquashNet,to_file=os.path.join(o_filepath,"network_graph.png"))
                
    #==========================================================================
    # Configure dataset
    print("-"*80,"\nCONFIGURING DATASET:")
    
    # Choose between batch fraction and batch size & recalculate batch fraction
    if (training_config.batch_fraction*i_values.size >= 1):
        print("\n{:30}{:.6e}".format("Input batch fraction:",training_config.batch_fraction))
        training_config.batch_size = math.floor(training_config.batch_fraction*i_values.size)
        training_config.batch_fraction = (training_config.batch_size/i_values.size) 
    else: pass  

    print("\n{:30}{}{}".format("Batch:","batch_size = ",training_config.batch_size))
    print("\n{:30}{}{}".format("Cache:","cache_dataset = ",runtime_config.cache_dataset))
    print("\n{:30}{}{}".format("Shuffle:","shuffle_dataset = ",runtime_config.shuffle_dataset))
    
    # Generate a TF dataset to supply volume and values batches during training 
    dataset = MakeDatasetFromTensorSlc(volume=i_volume,values=i_values,weights=weights,batch_size=training_config.batch_size,cache_dataset=runtime_config.cache_dataset)
    
    #==========================================================================  
    # Training loop
    print("-"*80,"\nCOMPRESSING DATA:")
        
    # Create a dictionary of lists to store training data
    training_data = {"epoch":[],"error":[],"time":[],"learning_rate":[],"psnr":[],"vol_std":volume_standard_deviation,"val_std":values_standard_deviation}

    # Start the overall training timer
    training_time_tick = time.time()
    
    # Iterate through each epoch
    for epoch in range(training_config.epochs):
        
        print("\n",end="")
                        
        # Store and print the current epoch number
        training_data["epoch"].append(float(epoch))
        print("{:30}{:02}/{:02}".format("Epoch:",epoch,training_config.epochs))
        
        # Determine, update, store and print the learning rate 
        learning_rate = GetLearningRate(initial_lr=training_config.initial_lr,half_life=training_config.half_life,epoch=epoch)
        optimiser.lr.assign(learning_rate)
        training_data["learning_rate"].append(float(learning_rate))   
        print("{:30}{:.3E}".format("Learning rate:",learning_rate))
        
        # Start timing current epoch
        epoch_time_tick = time.time()

        # Iterate through each batch
        for batch, (volume_batch,values_batch,weights_batch) in enumerate(dataset):
            
            # Print the current batch number and run a training step
            if runtime_config.print_verbose: print("\r{:30}{:04}/{:04}".format("Batch number:",(batch+1),dataset.size),end="") 
            TrainStepTFF(model=SquashNet,optimiser=optimiser,metric=metric,volume_batch=volume_batch,values_batch=values_batch,weights_batch=weights_batch)

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
        else: pass
    
        # Make a new dataset instance, garbage collect
        if runtime_config.shuffle_dataset and (epoch != (training_config.epochs-1)):
            del(dataset); gc.collect()
            dataset = MakeDatasetFromTensorSlc(volume=i_volume,values=i_values,weights=weights,batch_size=training_config.batch_size,cache_dataset=runtime_config.cache_dataset)
            print("\n{:30}".format("Reshuffling dataset"))
        else: pass
   
    ##   
 
    # End the overall training timer
    training_time_tock = time.time()
    training_time = float(training_time_tock-training_time_tick)
    print("\n{:30}{:.2f} seconds".format("Training duration:",training_time))    

    #==========================================================================
    # Quantise network parameters
    
    if (network_config.bits_per_neuron <= 32): 
        print("{:30}{:}".format("Quantising Weights:","bits_per_neuron = {}".format(network_config.bits_per_neuron)))
        quantised_weights = QuantiseParameters(SquashNet.get_weights(),network_config.bits_per_neuron)
        SquashNet.set_weights(quantised_weights)
    else: pass
        
    #==========================================================================
    # Finalise outputs    

    # Generate the output volume
    o_values.flat = SquashNet.predict(o_volume.flat,batch_size=training_config.batch_size,verbose="1")
    o_values.data = np.reshape(o_values.flat,(o_volume.data.shape[:-1]+(o_values.dimensions,)),order="C")
    
    # Calculate and report PSNR
    print("{:30}{:.3f}".format("Output volume PSNR:",SignalToNoise(true=i_values.flat,pred=o_values.flat,weights=weights.flat)))
    training_data["psnr"].append(SignalToNoise(true=i_values.flat,pred=o_values.flat,weights=weights.flat))
    
    # Pack the configuration dictionaries into just one
    combined_config_dict = (network_config | training_config | runtime_config | dataset_config)

    #==========================================================================
    # Save network
    
    if runtime_config.save_network_flag:
        print("-"*80,"\nSAVING NETWORK:")
        print("\n",end="")
        
        # Save the parameters
        parameters_path = os.path.join(o_filepath,"parameters.bin")
        EncodeParameters(network=SquashNet,parameters_path=parameters_path,values_bounds=(i_values.max,i_values.min))
        print("{:30}{}".format("Saved parameters to:",parameters_path.split("/")[-1]))
        
        # Save the architecture
        architecture_path = os.path.join(o_filepath,"architecture.bin")
        EncodeArchitecture(layer_dimensions=network_config.layer_dimensions,frequencies=network_config.frequencies,architecture_path=architecture_path)
        print("{:30}{}".format("Saved architecture to:",architecture_path.split("/")[-1]))
    else: pass
    
    #==========================================================================
    # Save outputs
    
    if runtime_config.save_outputs_flag:
        print("-"*80,"\nSAVING OUTPUTS:")
        print("\n",end="")
        
        # Save i_volume and i_values to ".npy" and ".vtk" files
        output_data_path = os.path.join(o_filepath,"i_volume")
        SaveData(output_data_path=output_data_path,volume=i_volume,values=i_values,reverse_normalise=True)
        print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1])) 
        
        # Save o_volume and o_values to ".npy" and ".vtk" files
        output_data_path = os.path.join(o_filepath,"o_volume")
        SaveData(output_data_path=output_data_path,volume=o_volume,values=o_values,reverse_normalise=True)
        print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1]))        
    else: pass    

    #==========================================================================
    # Save spectra
    
    if runtime_config.save_spectra_flag:
        print("-"*80,"\nSAVING SPECTRA:")
        print("\n",end="")
        
        # Save the i_values spectrum to ".npy" and ".vtk" files
        output_data_path = os.path.join(o_filepath,"i_spectrum")
        SaveSpectrum(output_data_path=output_data_path,values=i_values,maximum=i_values.max,minimum=i_values.min)
        print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1])) 
        
        # Save the o_values spectrum to ".npy" and ".vtk" files
        output_data_path = os.path.join(o_filepath,"o_spectrum")
        SaveSpectrum(output_data_path=output_data_path,values=o_values,maximum=i_values.max,minimum=i_values.min)
        print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1])) 
    else: pass    

    #==========================================================================
    # Save results
    
    if runtime_config.save_results_flag:
        print("-"*80,"\nSAVING RESULTS:")        
        print("\n",end="")
        
        # Save the training data
        training_data_path = os.path.join(o_filepath,"training_data.json")
        with open(training_data_path,"w") as file: json.dump(training_data,file,indent=4,sort_keys=True)
        print("{:30}{}".format("Saved training data to:",training_data_path.split("/")[-1]))
    
        # Save the configuration
        combined_config_path = os.path.join(o_filepath,"config.json")
        with open(combined_config_path,"w") as file: json.dump(combined_config_dict,file,indent=4)
        print("{:30}{}".format("Saved configuration to:",combined_config_path.split("/")[-1]))
    else: pass
    
    #==========================================================================
    print("-"*80,"\n")
        
    return SquashNet
       
#==============================================================================
# Define the main function to run when file is invoked from within the terminal

if __name__=="__main__":
    
    if (len(sys.argv) == 1):
        
        #======================================================================
        # This block will run in the event that this script is called in an IDE
    
        network_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/network_config.json"
        
        with open(network_config_path) as network_config_file: 
            network_config_dictionary = json.load(network_config_file)
            network_config = NetworkConfigurationClass(network_config_dictionary)
        ##   
        
        dataset_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/dataset_config.json"
        
        with open(dataset_config_path) as dataset_config_file: 
            dataset_config_dictionary = json.load(dataset_config_file)
            dataset_config = GenericConfigurationClass(dataset_config_dictionary)
        ##  
            
        runtime_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/runtime_config.json"
        
        with open(runtime_config_path) as runtime_config_file: 
            runtime_config_dictionary = json.load(runtime_config_file)
            runtime_config = GenericConfigurationClass(runtime_config_dictionary)
        ##    
            
        training_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/training_config.json"
        
        with open(training_config_path) as training_config_file: 
            training_config_dictionary = json.load(training_config_file)
            training_config = GenericConfigurationClass(training_config_dictionary)
        ##
        
        o_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"
        
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
    
        # Define ensemble output weights
        ensemble_output = None

        # Execute compression
        for index in range(1):
            ensemble_output = compress(network_config=network_config,dataset_config=dataset_config,runtime_config=runtime_config,training_config=training_config,o_filepath=o_filepath,index=index,ensemble_output=ensemble_output)   
        ##
        
        # Create a checkpoint file after successful execution
        with open(checkpoint_filename, mode='w'): pass

    else: print("Checkpoint file '{}' already exists: skipping.".format(checkpoint_filename))
        
else: pass

#==============================================================================
