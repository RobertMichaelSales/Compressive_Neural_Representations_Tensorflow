""" Created: 18.07.2022  \\  Updated: 10.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import user-defined libraries 

from data_management         import DataClass
from file_management         import FileClass
from network_configuration   import NetworkConfigClass
from network_encoder         import EncodeWeights,EncodeArchitecture
from network_make            import BuildNeurComp
from training_functions      import TrainStep,LearningRate,LossPSNR

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time,json,math
import numpy as np
import tensorflow as tf

#==============================================================================
    
def compress(base_directory,input_filepath,config_filepath,verbose):
        
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
    
    # Plot and save the graph for the network architecture
    tf.keras.utils.plot_model(neur_comp,to_file=filepaths.network_image_path,show_shapes=True)
    
    # Set a training optimiser and select training metrics
    optimiser = tf.keras.optimizers.Adam(learning_rate=network_config.initial_learning_rate)
    mse_error = tf.keras.metrics.MeanSquaredError()
    
    #==========================================================================
    # Configure network model and build
    print("-"*80,"\nCONFIGURING DATASET:")
    
    # Generate a TF dataset to supply volume and values batches during training 
    training_dataset = input_data.MakeDataset(batch_size=network_config.batch_size,repeat=True)
    
    #==========================================================================
    # Start compressing data
    print("-"*80,"\nCOMPRESSING DATA:")
        
    # Create a dictionary of lists for storing training data
    training_data = {"epoch": [],"error": [],"time": [],"learning_rate": []}
    
    # Determine the number of batches per epoch and in total
    batches_per_epoch = int(math.ceil(input_data.size/network_config.batch_size))
    total_num_batches = int(network_config.epochs*batches_per_epoch)
        
    # Start the training timer
    training_time_tick = time.time()
    
    # Enter the inner training loop
    for batch,(volume_batch,values_batch,indices_batch) in enumerate(training_dataset):
        
        
        # If batch is at the start of training epoch                 
        if (batch % batches_per_epoch == 0):
            
            print("\n",end="")
            
            # Determine and store the current epoch
            epoch = batch // batches_per_epoch
            training_data["epoch"].append(float(epoch))
    
            # Determine and store the learning rate 
            learning_rate = LearningRate(network_config=network_config,epoch=epoch)
            training_data["learning_rate"].append(float(learning_rate))   
            
            # Assign the learning rate to the optimiser
            optimiser.lr.assign(learning_rate)
                        
            # Print the epoch number and learning rate
            print("{:30}{:02}/{:02}".format("Epoch:",epoch,network_config.epochs))
            print("{:30}{:.3E}".format("Learning Rate:",learning_rate))
            
            # Start the epoch timer
            epoch_time_tick = time.time()
        else: pass
    
    
        # Print the batch number
        if verbose: 
            print("\r{:30}{:04}/{:04}".format("Batch Number:",((batch%batches_per_epoch)+1),batches_per_epoch),end="")        
        else: pass                
        
        # Run a training step on the current batch
        TrainStep(model=neur_comp,optimiser=optimiser,metric=mse_error,volume_batch=volume_batch,values_batch=values_batch,indices_batch=indices_batch)

        
        # If batch is at the end of training epoch
        if ((batch+1) % batches_per_epoch == 0):
            
            if verbose: 
                print("\n",end="")
            else: pass
            
            # End the epoch time and store the elapsed time 
            epoch_time_tock = time.time()       
            training_data["time"].append(float(epoch_time_tock-epoch_time_tick))
            
            # Fetch, store and reset and the training error
            training_data["error"].append(float(mse_error.result().numpy()))
            mse_error.reset_states()
            
            # Print the mean squared error and training time
            print("{:30}{:.7f}".format("Mean Squared Error:",training_data["error"][-1]))
            print("{:30}{:.2f} seconds".format("Epoch Time:",training_data["time"][-1]))
            
        else: pass
    
            
        # If batch is the last batch then exit from training
        if ((batch+1) == total_num_batches): 
            break
        else: pass
    
        #======================================================================
             
    # End the training timer
    training_time_tock = time.time()
    print("\n{:30}{:.2f} seconds".format("Training Duration:",training_time_tock-training_time_tick))             
    
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
    
    return None
    
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
