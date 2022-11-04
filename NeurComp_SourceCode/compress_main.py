""" Created: 18.07.2022  \\  Updated: 02.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import user-defined libraries 

from data_management         import DataClass
from training_functions      import TrainStep,LRScheduler,LossPSNR
from network_configuration   import NetworkConfigClass
from network_make            import BuildNeurComp

#==============================================================================
# Import libraries and set flags

import os,time,json
import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if not (len(gpus) == 0): tf.config.experimental.set_memory_growth(gpus[0],True)

#==============================================================================
# Set filepaths

base_directory = os.path.join(os.path.dirname(os.getcwd()),"NeurComp_AuxFiles")

data_file_path = os.path.join(base_directory,"inputs","volumes","test_vol.npy")

#==============================================================================
# Enter the main script
print("-"*80,"\nNEURCOMP: IMPLICIT NEURAL REPRESENTATIONS (Version 2.0)")

#==============================================================================
# Start initialising files and model
print("-"*80,"\nINITIALISING FILES AND MODEL:\n")

# Create a 'DataClass' object to store input data and load input data from file
input_data = DataClass(name="input_data")
input_data.LoadData(filepath=data_file_path,normalise=True)

# Create a 'DataClass' object to store output data and copy important meta-data
output_data = DataClass(name="output_data")
output_data.CopyData(DataClassObject=input_data)

# Create a 'NetworkConfigClass' object to store the network hyperparameters and
# generate the network structure based on the input dimensions
network_config = NetworkConfigClass(name="network_config")
network_config.NetworkStructure(input_data=input_data)

# Use the '.MakeDataset' method to generate a Tensorflow '.data' dataset to use
# for supplying volume and values batches during training and evaluation 
training_dataset = input_data.MakeDataset(network_config=network_config)

# Build NeurComp from the config information stored in 'network_config', select
# an optimiser for training, and select a training metric to track the progress
neur_comp = BuildNeurComp(network_config=network_config)
optimiser = tf.keras.optimizers.Adam(learning_rate=network_config.initial_learning_rate)
training_metric = tf.keras.metrics.MeanSquaredError()

# Plot and save an image of the network architecture
tf.keras.utils.plot_model(neur_comp,to_file="neurcomp.png",show_shapes=True)

#==============================================================================
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
    print("Epoch Number \t\t\t{}/{}".format(epoch+1,network_config.num_epochs))
    print("Learning Rate\t\t\t{:.2E}".format(training_data["learning_rate"][epoch]))
    
    # Start timing epoch
    epoch_start_time = time.time()
    
    #==========================================================================
    # Enter the inner training loop: batches:
    for batch, (volume_batch,values_batch,indices_batch) in enumerate(training_dataset):
        
        # Print the batch progress
        print("\rBatch Number \t\t\t{}/{}".format(batch+1,len(training_dataset)),end="")
        
        # Run a training step with the current batch
        TrainStep(neur_comp,optimiser,training_metric,volume_batch,values_batch,indices_batch)
            
    #==========================================================================  
    print("\n",end="")

    # Fetch, store, reset and print the training metric
    training_data["loss"].append(float(training_metric.result().numpy()))
    training_metric.reset_states()
    print("Mean Squared Loss\t\t{:.5f}".format(training_data["loss"][epoch]))
    
    # Stop timing epoch and print the elapsed time
    epoch_end_time = time.time()
    training_data["time"].append(float(epoch_end_time-epoch_start_time))
    print("Epoch Time\t\t\t\t{:.2f} seconds".format(training_data["time"][epoch]))
    
# Stop timing training
train_end_time = time.time()
print("\nTotal Training Time\t{:.2f} seconds".format(train_end_time-train_start_time))
#==============================================================================
# Save training data
print("-"*80,"\nSAVING RESULTS:")

training_data_filepath = os.path.join(base_directory,"outputs","training_data.json")                            # <------------------ WORK FROM HERE

with open(training_data_filepath,"w") as file: json.dump(training_data,file,indent=4,sort_keys=True)

config_data_filepath = os.path.join(base_directory,"outputs","configuration_data.json")         

network_config.SaveConfigToJson(configuration_filepath=config_data_filepath)

#==============================================================================
# Start predicting data
print("-"*80,"\nPREDICTING")

data_file_path_output = os.path.join(base_directory,"outputs","test_vol_output.json")

# Predict 'flat_values' using the Neurcomp's learned weights and biases
output_data.flat_values = neur_comp.predict(input_data.flat_volume,batch_size=network_config.batch_size,verbose="0")

# Reshape 'flat_values' into the shape of the original input dimensions
output_data.values = np.reshape(output_data.flat_values,input_data.values.shape,order="C")

# Compute the peak signal-to-noise ratio of the predicted volume
psnr = LossPSNR(true=input_data.values,pred=output_data.values)
print("Compression Peak Signal-to-Noise Ratio: {:.2f}".format(psnr))

# Save the predicted values to a '.npy' volume rescaling as appropriate
output_data.SaveData(filepath=data_file_path_output,normalise=True)

#==============================================================================
print("-"*80)
