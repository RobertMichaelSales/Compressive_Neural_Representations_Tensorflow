""" Created: 30.11.2022  \\  Updated: 17.01.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import json
import math
import numpy as np
import tensorflow as tf

#==============================================================================
# Import user-defined libraries 

from data_management_edited  import DataClass,MakeDataset,SaveData
from network_configuration   import ConfigurationClass
from network_encoder         import EncodeParameters,EncodeArchitecture
from network_model           import ConstructNetwork
from compress_utilities      import TrainStep,GetLearningRate,SignalToNoise

#==============================================================================
    
def compress(input_data_path,config_path,output_path,export_output):
    
    print("-"*80,"\nNEURCOMP: IMPLICIT NEURAL REPRESENTATIONS (by Rob Sales)")
        
    #==========================================================================
    # Check whether hardware acceleration is enabled
   
    gpus = tf.config.list_physical_devices('GPU')
    
    if (len(gpus) != 0): 
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Enabled"))
        tf.config.experimental.set_memory_growth(gpus[0],True)
    else:        
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Disabled"))
        return None
        
    #==========================================================================
    # Initialise i/o 
    
    print("-"*80,"\nINITIALISING DATA I/O:")
    
    # Create 'DataClass' objects to store i/o volume and values
    i_volume = DataClass(data_type="volume")
    i_values = DataClass(data_type="values")
    o_volume = DataClass(data_type="volume")
    o_values = DataClass(data_type="values")
    
    # Load and normalise input data
    i_volume.LoadData(input_data_path=input_data_path,dimensions=(3,1),normalise=True)
    i_values.LoadData(input_data_path=input_data_path,dimensions=(3,1),normalise=True)
    
    # Copy meta-data from the input
    o_volume.CopyData(DataClassObject=i_volume)
    o_values.CopyData(DataClassObject=i_values)
    
    #==========================================================================
    # Configure network 
    
    print("-"*80,"\nCONFIGURING NETWORK:")
    
    # Create a 'ConfigurationClass' object to store net parameters 
    network_cfg = ConfigurationClass(config_path=config_path)
    
    # Generate the network structure based on the input dimensions
    network_cfg.GenerateStructure(i_dimensions=i_volume.dimensions,o_dimensions=i_values.dimensions,size=i_values.size)
    
    # Build NeurComp from the config information
    SquashNet = ConstructNetwork(layer_dimensions=network_cfg.layer_dimensions)
    
    # Set a training optimiser
    optimiser = tf.keras.optimizers.Adam(learning_rate=network_cfg.initial_lr)
    
    # Set a performance metric
    mse_error_metric = tf.keras.metrics.MeanSquaredError()
    
    #==========================================================================
    # Configure output folder
    print("-"*80,"\nCONFIGURING FOLDERS:")
    
    # Create an output directory for all future saved files
    output_directory = os.path.join(output_path,network_cfg.network_name)
    if not os.path.exists(output_directory):os.makedirs(output_directory)
    print("\n{:30}{}".format("Created output folder:",output_directory.split("/")[-1]))
            
    #==========================================================================
    # Configure dataset
    print("-"*80,"\nCONFIGURING DATASET:")
    
    # Generate a TF dataset to supply volume and values batches during training 
    dataset = MakeDataset(volume=i_volume,values=i_values,batch_size=network_cfg.batch_size,repeat=False)
        
    #==========================================================================
    # Compression loop
    print("-"*80,"\nCOMPRESSING DATA:")
    
    # Create a dictionary of lists to store training data
    training_data = {"epoch": [],"error": [],"time": [],"learning_rate": []}
    
    # Determine the number of batches per epoch and in total
    batches_per_epoch = int(math.ceil(i_values.size/network_cfg.batch_size))
    total_num_batches = int(network_cfg.epochs*batches_per_epoch)
        
    # Start the overall training timer
    training_time_tick = time.time()
    
    # Enter the inner training loop
    for batch,(volume_batch,values_batch) in enumerate(dataset):
        
        # If batch is at the start of training epoch                 
        if (batch % batches_per_epoch == 0):
            
            print("\n",end="")
            
            # Determine, store and print the current epoch
            epoch = batch // batches_per_epoch
            training_data["epoch"].append(float(epoch))
            print("{:30}{:02}/{:02}".format("Epoch:",epoch,network_cfg.epochs))
            
            # Determine, update, store and print the learning rate 
            learning_rate = GetLearningRate(initial_lr=network_cfg.initial_lr,half_life=network_cfg.half_life,epoch=epoch)
            optimiser.lr.assign(learning_rate)
            training_data["learning_rate"].append(float(learning_rate))   
            print("{:30}{:.3E}".format("Learning Rate:",learning_rate))
                
            # Start the epoch timer
            epoch_time_tick = time.time()
        else: pass
    
        # Print the current batch number
        print("\r{:30}{:04}/{:04}".format("Batch Number:",((batch%batches_per_epoch)+1),batches_per_epoch),end="")        
             
        # Run a training step
        TrainStep(model=SquashNet,optimiser=optimiser,metric=mse_error_metric,volume_batch=volume_batch,values_batch=values_batch)
        
        # If batch is at the end of training epoch
        if ((batch+1) % batches_per_epoch == 0):

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
            
        else: pass
    
        # If batch is the last batch then exit from training
        if ((batch+1) == total_num_batches): 
            break
        else: pass
    
        #======================================================================
             
    # End the overall training timer
    training_time_tock = time.time()
    training_time = float(training_time_tock-training_time_tick)
    print("\n{:30}{:.2f} seconds".format("Training Duration:",training_time))               
    
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
    EncodeArchitecture(layer_dimensions=network_cfg.layer_dimensions,architecture_path=architecture_path)
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

    # Save the output volume to ".npy" and ".vtk" files
    output_data_path = os.path.join(output_path,network_cfg.network_name,"output_volume")
    SaveData(output_data_path=output_data_path,volume=o_volume,values=o_values,reverse_normalise=True)
    print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1]))
    
    #==========================================================================
    print("-"*80,"\n")
    
#==============================================================================
# Define the script to run when envoked from the terminal

if __name__=="__main__":

    # Set config filepath
    config_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/config_test.json"
    
    # Set base directory
    base_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles"
    
    # Set input filepath
    input_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/test_vol.npy"
 
    compress(base_directory=base_directory,input_filepath=input_filepath,config_filepath=config_filepath,verbose=True)   

else:
    
    print("Please invoke 'compress_main.py' correctly from the terminal!")
    
#==============================================================================
