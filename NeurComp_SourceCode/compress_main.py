""" Created: 18.07.2022  \\  Updated: 15.02.2023  \\   Author: Robert Sales """

# Changes made to lines 135/6 to print progress when training with new dataset.

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time, json, math, psutil
import numpy as np
import tensorflow as tf

#==============================================================================
# Import user-defined libraries 

from data_management         import DataClass,MakeDataset,SaveData #,MakeDatasetFromGenerator
from network_configuration   import ConfigurationClass
from network_encoder         import EncodeParameters,EncodeArchitecture
from network_model           import ConstructNetwork
from compress_utilities      import *

#==============================================================================

def compress(input_data_path,config_path,output_path,export_output):
    
    print("-"*80,"\nSQUASHNET: IMPLICIT NEURAL REPRESENTATIONS (by Rob Sales)")
        
    #==========================================================================
    # Check whether hardware acceleration is enabled
   
    gpus = tf.config.list_physical_devices('GPU')
    
    if (len(gpus) != 0): 
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Enabled"))
        tf.config.experimental.set_memory_growth(gpus[0],True)
    else:        
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Disabled"))
        
    #==========================================================================
    # Check whether the input size exceeds available memory
    
    available_memory = psutil.virtual_memory().available
    print("\n{:30}{:.3f} GigaBytes".format("Available Memory:",(available_memory/1e9)))
    
    threshold_memory = int(4*1e9)
    print("\n{:30}{:.3f} GigaBytes".format("Threshold Memory:",(threshold_memory/1e9)))
    
    input_file_size = os.path.getsize(input_data_path)
    print("\n{:30}{:.3f} GigaBytes".format("Input File Size:",(input_file_size/1e9)))
    
    exceeds_memory = (input_file_size > min(available_memory,threshold_memory))
    
    if exceeds_memory: print("\n{:30}{}".format("Warning:","File Size > RAM - Loading as Memmap File"))
    
    #==========================================================================
    # Initialise i/o 
    
    print("-"*80,"\nINITIALISING DATA I/O:")
    
    # is_tabular = True
    # shape = (100520000,10)
    # dtype = 'float64'
    # normalise = True
    # columns = ([2,3],[7])
    
    is_tabular = False
    shape = (150,150,150,4)
    dtype = 'float32'
    normalise = True
    columns = ([0,1,2],[3])
    
    # Create 'DataClass' objects to store i/o volume and values
    i_volume = DataClass(data_type="volume",is_tabular=is_tabular,exceeds_memory=exceeds_memory)
    i_values = DataClass(data_type="values",is_tabular=is_tabular,exceeds_memory=exceeds_memory)
    o_volume = DataClass(data_type="volume",is_tabular=is_tabular,exceeds_memory=exceeds_memory)
    o_values = DataClass(data_type="values",is_tabular=is_tabular,exceeds_memory=exceeds_memory)
    
    # Load and normalise input data
    i_volume.LoadData(input_data_path=input_data_path,columns=columns,shape=shape,dtype=dtype,normalise=normalise)
    i_values.LoadData(input_data_path=input_data_path,columns=columns,shape=shape,dtype=dtype,normalise=normalise)
       
    # Copy meta-data from the input
    o_volume.CopyData(DataClassObject=i_volume,exception_keys=[])
    o_values.CopyData(DataClassObject=i_values,exception_keys=[])    
    
    #==========================================================================
    # Calculate statistics on the input volume
    
    stats_flag = False
    
    if stats_flag:
        values_standard_deviation = CalculateStandardDeviation(i_values.flat)
        volume_pointcloud_density = CalculatePointCloudDensity(i_volume.flat)
    else: pass
    
    return i_volume,i_values

    #==========================================================================    
    # Configure network 
    
    print("-"*80,"\nCONFIGURING NETWORK:")
    
    # Create a 'ConfigurationClass' object to store net parameters 
    network_cfg = ConfigurationClass(config_path=config_path)
    
    # Generate the network structure based on the input dimensions
    network_cfg.GenerateStructure(i_dimensions=i_volume.dimensions,o_dimensions=i_values.dimensions,size=i_values.size)
    
    # Build NeurComp from the config information
    SquashNet = ConstructNetwork(layer_dimensions=network_cfg.layer_dimensions,frequencies=network_cfg.frequencies)
    
    # Set a training optimiser
    optimiser = tf.keras.optimizers.Adam()
    
    # Set a performance metric
    mse_error_metric = tf.keras.metrics.MeanSquaredError()
    
    #==========================================================================
    # Configure output folder
    print("-"*80,"\nCONFIGURING FOLDERS:")
    
    # Create an output directory for all future saved files
    output_directory = os.path.join(output_path,network_cfg.network_name)
    if not os.path.exists(output_directory):os.makedirs(output_directory)
    print("\n{:30}{}".format("Created output folder:",output_directory.split("/")[-1]))
    
    tf.keras.utils.plot_model(model=SquashNet,to_file=os.path.join(output_directory,"model.png"))                                       # <<<<<<<<<
            
    #==========================================================================
    # Configure dataset
    print("-"*80,"\nCONFIGURING DATASET:")
    
    # Choose between batch fraction and size (for experiments only)
    if (network_cfg.batch_fraction):
        network_cfg.batch_size = math.floor(network_cfg.batch_fraction*network_cfg.size)
        network_cfg.batch_fraction = (network_cfg.batch_size/network_cfg.size)
    else: pass
    
    # Generate a TF dataset to supply volume and values batches during training 
    dataset = MakeDataset(volume=i_volume,values=i_values,batch_size=network_cfg.batch_size,repeat=False)
    # dataset = MakeDatasetFromGenerator(volume=i_volume,values=i_values,batch_size=network_cfg.batch_size,repeat=False)                      # <<<<<<<<<
    
    #==========================================================================
    print("-"*80,"\nLEARNING RATE STUDY")
    
    lr_study_flag = False
    
    if lr_study_flag:
        optimal_initial_lr = LearningRateStudy(model=SquashNet,optimiser=optimiser,dataset=dataset,lr_bounds=(-7.0,-1.0),plot=True)         
    else: pass

    bf_study_flag = False
    
    if bf_study_flag:
        pass
    else: pass
    
    #==========================================================================
    # Compression loop
    print("-"*80,"\nCOMPRESSING DATA:")
    
    # Create a dictionary of lists to store training data
    training_data = {"epoch": [],"error": [],"time": [],"learning_rate": [], "psnr": []}

    # Start the overall training timer
    training_time_tick = time.time()
    
    # Iterate through each epoch
    for epoch in range(network_cfg.epochs):
        
        print("\n",end="")
        
        # Store and print the current epoch number
        training_data["epoch"].append(float(epoch))
        print("{:30}{:02}/{:02}".format("Epoch:",epoch,network_cfg.epochs))
        
        # Determine, update, store and print the learning rate 
        learning_rate = GetLearningRate(initial_lr=network_cfg.initial_lr,half_life=network_cfg.half_life,epoch=epoch)
        
        optimiser.lr.assign(learning_rate)
        training_data["learning_rate"].append(float(learning_rate))   
        print("{:30}{:.3E}".format("Learning Rate:",learning_rate))
        
        # Start timing current epoch
        epoch_time_tick = time.time()
        
        ## Iterate through each batch
        for batch, (volume_batch,values_batch) in enumerate(dataset):
            
            # Print the current batch number 
            print("\r{:30}{:04}/{:04}".format("Batch Number:",(batch+1),len(dataset)),end="") 
            
            # Run a training step 
            TrainStep(model=SquashNet,optimiser=optimiser,metric=mse_error_metric,volume_batch=volume_batch,values_batch=values_batch)
        ##
        
        print("\n",end="")
        
        # End the epoch time and store the elapsed time 
        epoch_time_tock = time.time() 
        epoch_time = float(epoch_time_tock-epoch_time_tick)
        training_data["time"].append(epoch_time)
        print("{:30}{:.2f} seconds".format("Epoch Time:",epoch_time))
        
        # Fetch, store and reset and the training error
        mse_error = float(mse_error_metric.result().numpy())
        mse_error_metric.reset_states()
        training_data["error"].append(mse_error)
        print("{:30}{:.7f}".format("Mean Squared Error:",mse_error))
    ##   
 
    # End the overall training timer
    training_time_tock = time.time()
    training_time = float(training_time_tock-training_time_tick)
    print("\n{:30}{:.2f} seconds".format("Training Duration:",training_time))    
       
    #==========================================================================
    # Save network 
    print("-"*80,"\nSAVING NETWORK:")
    
    print("\n",end="")
    
    # Extract value bounds
    values_bounds = (i_values.max,i_values.min)
    
    # Save the parameters
    parameters_path = os.path.join(output_path,network_cfg.network_name,"parameters.bin")
    EncodeParameters(network=SquashNet,parameters_path=parameters_path,values_bounds=values_bounds)
    print("{:30}{}".format("Saved parameters to:",parameters_path.split("/")[-1]))
    
    # Save the architecture
    architecture_path = os.path.join(output_path,network_cfg.network_name,"architecture.bin")
    EncodeArchitecture(layer_dimensions=network_cfg.layer_dimensions,frequencies=network_cfg.frequencies,architecture_path=architecture_path)
    print("{:30}{}".format("Saved architecture to:",architecture_path.split("/")[-1]))

    if not export_output:
        print("-"*80,"\n")
        return None
    else: pass
    
    #==========================================================================
    # Start predicting data
    print("-"*80,"\nSAVING OUTPUTS:")
    
    print("\n",end="")

    # Generate the output volume and calculate the PSNR
    o_values.flat = SquashNet.predict(o_volume.flat,batch_size=network_cfg.batch_size,verbose="1")
    o_values.data = np.reshape(o_values.flat,(o_volume.data.shape[:-1]+(1,)),order="C")
    print("{:30}{:.3f}".format("Output volume PSNR:",SignalToNoise(true=i_values.data,pred=o_values.data)))
    training_data["psnr"].append(SignalToNoise(true=i_values.data,pred=o_values.data))

    # Save the output volume to ".npy" and ".vtk" files
    output_data_path = os.path.join(output_path,network_cfg.network_name,"output_volume")
    SaveData(output_data_path=output_data_path,volume=o_volume,values=o_values,reverse_normalise=True)
    print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1]))
    
    #==========================================================================
    # Save results
    print("-"*80,"\nSAVING RESULTS:")
    
    print("\n",end="")
    
    # Save the training data
    training_data_path = os.path.join(output_path,network_cfg.network_name,"training_data.json")
    with open(training_data_path,"w") as file: json.dump(training_data,file,indent=4,sort_keys=True)
    print("{:30}{}".format("Saved training data to:",training_data_path.split("/")[-1]))

    # Save the configuration
    configuration_path = os.path.join(output_path,network_cfg.network_name,"configuration.json")
    with open(configuration_path,"w") as file: json.dump(vars(network_cfg),file,indent=4)
    print("{:30}{}".format("Saved configuration to:",configuration_path.split("/")[-1]))
    
    #==========================================================================
    print("-"*80,"\n")
   
#==============================================================================
# Define the script to run when envoked from the terminal

if __name__=="__main__":
    
    import sys
    
    # # Set config filepath
    config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/config.json"
    # config_path = sys.argv[1]
    
    # # Set input filepath
    input_data_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/passage.npy"
    # input_data_path = sys.argv[2]
    
    # # Set output filepath
    output_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"
    # output_path = sys.argv[3]
    
    # Execute compression
    vol,val = compress(input_data_path=input_data_path,config_path=config_path,output_path=output_path,export_output=True)   

else: pass

#==============================================================================
