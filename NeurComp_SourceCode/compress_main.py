""" Created: 18.07.2022  \\  Updated: 05.01.2023  \\   Author: Robert Sales """

# Separate volume and values (to be separate objects) in DataClass. Then move SaveData() from within DataClass to training_utils.py. Then rename training_utils.py to compress_utils.py.
# Separate volume and values (to be separate objects) in DataClass. Then move SaveData() from within DataClass to training_utils.py. Then rename training_utils.py to compress_utils.py.
# Separate volume and values (to be separate objects) in DataClass. Then move SaveData() from within DataClass to training_utils.py. Then rename training_utils.py to compress_utils.py.
# Separate volume and values (to be separate objects) in DataClass. Then move SaveData() from within DataClass to training_utils.py. Then rename training_utils.py to compress_utils.py.
# Separate volume and values (to be separate objects) in DataClass. Then move SaveData() from within DataClass to training_utils.py. Then rename training_utils.py to compress_utils.py.

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import json
import numpy as np
import tensorflow as tf

#==============================================================================
# Import user-defined libraries 

from data_management         import DataClass
from network_configuration   import ConfigurationClass
from network_encoder         import EncodeParameters,EncodeArchitecture
from network_model           import ConstructNetwork
from compress_utilities      import TrainStep,GetLearningRate,SignalToNoise,MakeDataset

#==============================================================================
    
def compress(volume_path,config_path,output_path,export_output):
    
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
    
    # Create 'DataClass' objects to store i/o data
    i_data, o_data = DataClass(), DataClass()
    
    # Load and normalise input data
    i_data.LoadData(volume_path=volume_path,i_dimensions=3,o_dimensions=1,normalise=True)
    
    # Copy meta-data from the input
    o_data.CopyMetaData(DataClassObject=i_data)
    
    #==========================================================================
    # Configure network 
    
    print("-"*80,"\nCONFIGURING NETWORK:")
    
    # Create a 'ConfigurationClass' object to store net parameters 
    network_cfg = ConfigurationClass(config_path=config_path)
    
    # Generate the network structure based on the input dimensions
    network_cfg.GenerateStructure(i_dimensions=i_data.i_dimensions,o_dimensions=i_data.o_dimensions,i_size=i_data.i_size)
    
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
    dataset = i_data.MakeDataset(batch_size=network_cfg.batch_size,repeat=False)
        
    #==========================================================================
    # Compression loop
    print("-"*80,"\nCOMPRESSING DATA:")
    
    # Create a dictionary of lists to store training data
    training_data = {"epoch": [],"error": [],"time": [],"learning_rate": []}

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
    values_bounds = (i_data.values_max,i_data.values_min)
    
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
    o_data.flat_values = SquashNet.predict(o_data.flat_volume,batch_size=network_cfg.batch_size,verbose="1")
    o_data.values = np.reshape(o_data.flat_values,(o_data.volume.shape[:-1]+(1,)),order="C")
    print("{:30}{:.3f}".format("Output volume PSNR:",SignalToNoise(true=i_data.values,pred=o_data.values)))

    # Save the output volume to ".npy" and ".vtk" files
    output_volume_path = os.path.join(output_path,network_cfg.network_name,"output_volume")
    o_data.SaveData(output_volume_path=output_volume_path,reverse_normalise=True)
    print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_volume_path.split("/")[-1]))
    
    #==========================================================================
    print("-"*80,"\n")
   
#==============================================================================
# Define the script to run when envoked from the terminal

if __name__=="__main__":
    
    # Set config filepath
    config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/config.json"
       
    # Set input filepath
    volume_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/volume.npy"
    
    # Set output filepath
    output_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"
 
    # Execute compression
    compress(volume_path=volume_path,config_path=config_path,output_path=output_path,export_output=True)   

else: pass
    
#==============================================================================
