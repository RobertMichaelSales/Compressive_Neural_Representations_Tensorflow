""" Created: 18.07.2022  \\  Updated: 10.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import user-defined libraries 

from data_management         import DataClass
from file_management         import FileClass
from network_configuration   import NetworkConfigClass
from network_encoder         import EncodeWeights,EncodeArchitecture
from network_make            import BuildNeurComp
from training_functions      import TrainStep,LRScheduler,LossPSNR

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time,json
import numpy as np
import tensorflow as tf

#==============================================================================
    
def compress(base_directory,input_filepath,config_filepath):
        
    #==========================================================================
    # Enter the main script and check for hardware acceleration
    print("-"*80,"\nNEURCOMP: IMPLICIT NEURAL REPRESENTATIONS (Version 2.0)")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if (len(gpus) == 0):        
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Disabled"))
    else:        
        tf.config.experimental.set_memory_growth(gpus[0],True)
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Enabled"))
        
    #==========================================================================
    # Start initialising I/O data files
    print("-"*80,"\nINITIALISING FILES:")
    
    # Create 'DataClass' objecta to store input/output data
    input_data, output_data = DataClass(), DataClass()
    
    # Load and normalise input data
    input_data.LoadData(filepath=input_filepath,normalise=True)
    
    # Copy meta-data from the input
    output_data.CopyData(DataClassObject=input_data)
    
    #==========================================================================
    # Configure network model and build
    print("-"*80,"\nCONFIGURING BUILD:")
    
    # Create a 'NetworkConfigClass' object to store net parameters 
    network_config = NetworkConfigClass(config_filepath=config_filepath)
    
    # Generate the network structure based on the input dimensions
    network_config.NetworkStructure(input_data=input_data)
    
    # Create a 'FileClass'object to store output directory and filepath data
    print("\n{:30}{}".format("Created filepaths at:",base_directory.split("/")[-1]))
    filepaths = FileClass(base_directory=base_directory,network_config=network_config)
        
    # Build NeurComp from the config information
    print("\n{:30}{}".format("Constructed network:",network_config.network_name))
    neur_comp = BuildNeurComp(layer_dimensions=network_config.layer_dimensions)
    
    # Plot and save an image of the network architecture#
    tf.keras.utils.plot_model(neur_comp,to_file=filepaths.network_image_path,show_shapes=True)
    
    # Set a training optimiser and select a training metric 
    optimiser = tf.keras.optimizers.Adam(learning_rate=network_config.initial_learning_rate)
    training_metric = tf.keras.metrics.MeanSquaredError()
    
    #==========================================================================
    # Configure network model and build
    print("-"*80,"\nCONFIGURING DATASET:")
    
    # Use the '.MakeDataset' method to generate a Tensorflow dataset to use ...
    # for supplying volume and values batches during training and evaluation 
    training_dataset = input_data.MakeDataset(network_config=network_config)
    
    #==========================================================================
    # Start compressing data
    print("-"*80,"\nCOMPRESSING DATA:")
    
    # Start timing training
    train_start_time = time.time()
    
    # Create a dictionary of lists for storing training data
    training_data = {"epoch": [],"loss": [],"time": [],"learning_rate": []}
    
    # Enter the outer training loop: epochs:
    for epoch in range(network_config.num_epochs):
        
        # Store the current epoch number
        training_data["epoch"].append(float(epoch))
          
        # Update and store the current learning rate
        learning_rate = LRScheduler(network_config,epoch)
        optimiser.lr.assign(learning_rate)
        training_data["learning_rate"].append(float(learning_rate))
        
        # Print the current epoch number and learning rate
        print("\n",end="")
        print("{:30}{:02}/{:02}".format("Epoch Number:",(epoch+1),network_config.num_epochs))
        print("{:30}{:.3E}".format("Learning Rate:",training_data["learning_rate"][epoch]))
        
        # Start timing epoch
        epoch_start_time = time.time()
        
        #======================================================================
        # Enter the inner training loop: batches:
        for batch, (volume_batch,values_batch,indices_batch) in enumerate(training_dataset):
            
            # Print the batch progress
            print("\r{:30}{:04}/{:04}".format("Batch Number:",(batch+1),len(training_dataset)),end="")
            
            # Run a training step with the current batch
            TrainStep(neur_comp,optimiser,training_metric,volume_batch,values_batch,indices_batch)
                
        #====================================================================== 
        print("\n",end="")
    
        # Fetch, store, reset and print the training metric
        training_data["loss"].append(float(training_metric.result().numpy()))
        training_metric.reset_states()
        print("{:30}{:.7f}".format("Mean Squared Loss:",training_data["loss"][epoch]))
        
        # Stop timing epoch and print the elapsed time
        epoch_end_time = time.time()
        training_data["time"].append(float(epoch_end_time-epoch_start_time))
        print("{:30}{:.2f} seconds".format("Epoch Time:",training_data["time"][epoch]))
        
    # Stop timing training
    train_end_time = time.time()
    print("\n{:30}{:.2f} seconds".format("Training Duration:",train_end_time-train_start_time))
    
    #==========================================================================
    # Save training data
    print("-"*80,"\nSAVING NETWORK AND RESULTS:")
    
    # Save the configuration
    print("\n{:30}{}".format("Saved configuration to:",filepaths.network_configuration_path.split("/")[-1]))
    with open(filepaths.network_configuration_path,"w") as file: json.dump(vars(network_config),file,indent=4)
    
    # Save the training data
    print("{:30}{}".format("Saved training data to:",filepaths.training_data_path.split("/")[-1]))
    with open(filepaths.training_data_path,"w") as file: json.dump(training_data,file,indent=4,sort_keys=True)
    
    # Save the net architecture
    print("{:30}{}".format("Saved architecture to:",filepaths.network_architecture_path.split("/")[-1]))
    EncodeArchitecture(config=network_config,filepath=filepaths.network_architecture_path)
    
    # Save the trained weights
    print("{:30}{}".format("Saved weights/biases to:",filepaths.network_weights_path.split("/")[-1]))
    EncodeWeights(network=neur_comp,filepath=filepaths.network_weights_path)
    
    #==========================================================================
    # Start predicting data
    print("-"*80,"\nRECONSTRUCTING INPUT:")
    
    # Predict values using the Neurcomp's learned weights and biases
    output_data.flat_values = neur_comp.predict(output_data.flat_volume,batch_size=network_config.batch_size,verbose="1")
    output_data.values = np.reshape(output_data.flat_values,(output_data.volume.shape[:-1]+(1,)),order="C")
    
    # Compute the peak signal-to-noise ratio of the predicted volume
    psnr = LossPSNR(true=input_data.values,pred=output_data.values)
    print("\n{:30}{:.3f}".format("Output PSNR:",psnr))
    
    # Save the predicted values to a '.npy' volume rescaling as appropriate
    print("{:30}{}".format("Saved reconstruction to:",filepaths.output_volume_path.split("/")[-1]))
    output_data.SaveData(output_volume_path=filepaths.output_volume_path,reverse_normalise=True)
    
    #==========================================================================
    print("-"*80,"\n"*2)
    
    return input_data.flat_volume
    
#==============================================================================
# Define the script to run when envoked from the terminal

if __name__=="__main__":

    # Set config filepath
    config_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/config_test.json"
    
    # Set base directory
    base_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles"
    
    # Set input filepath
    input_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/test_vol.npy"
 
    flatvol1 = compress(base_directory=base_directory,input_filepath=input_filepath,config_filepath=config_filepath)   

else:
    
    print("Please invoke 'compress_main.py' correctly from the terminal!")
    
#==============================================================================
