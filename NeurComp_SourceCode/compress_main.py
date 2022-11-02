""" Created: 18.07.2022  \\  Updated: 02.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import user-defined libraries 

# from callback_functions      import *
# from network_load            import *
# from network_save            import *
# from predict_volumes         import *
# from utility_functions       import *

from data_management         import DataClass
from file_management         import FileStructureClass
from loss_functions          import LossPred,LossGrad,LossPSNR
from network_configuration   import NetworkConfigClass
from network_make            import BuildNeurComp

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)

import numpy as np
import time

#==============================================================================
# Set filepaths

parent_directory    = ("C:\\"
                       "Users\\"
                       "sales\\"
                       "Documents\\"
                       "GitHub\\"
                       "Compressive_Neural_Representations_Tensorflow\\"
                       "NeurComp_AuxFiles")

input_data_filepath = ("C:\\"
                       "Users\\"
                       "sales\\"
                       "Documents\\"
                       "GitHub\\"
                       "Compressive_Neural_Representations_Tensorflow\\"
                       "NeurComp_AuxFiles\\"
                       "inputs\\"
                       "volumes\\"
                       "test_vol.npy")

#==============================================================================
# Enter the main script
print("-"*60,"\nNEURCOMP: IMPLICIT NEURAL REPRESENTATIONS (Version 2.0)\n")

#==============================================================================
# Start initialising files and model
print("-"*60,"\nINITIALISING FILES AND MODEL:\n")

# Create a 'DataClass' object to store input data and load input data from file
input_data = DataClass()
input_data.LoadData(filepath=input_data_filepath,normalise=True)

# Create a 'DataClass' object to store output data and copy important meta-data
output_data = DataClass()
output_data.CopyData(DataClassObject=input_data)

# Create a 'NetworkConfigClass' object to store the network hyperparameters and
# generate the network structure based on the input dimensions
network_config = NetworkConfigClass()
network_config.NetworkStructure(input_data=input_data)

# Use the '.MakeDataset' method to generate a Tensorflow '.data' dataset to use
# for supplying volume and values batches during training 
dataset = input_data.MakeDataset(network_config=network_config)

# Create a 'FileStructureClass' object, and use it's initialisation constructor
# function to generate new folders, filepaths and filenames 
filepaths = FileStructureClass(parent_directory=parent_directory,network_name=network_config.network_name)

# Build NeurComp from the config information stored in 'network_config', select
# an optimiser for training, and select a training metric to track the progress
neur_comp = BuildNeurComp(network_config=network_config)
optimiser = tf.keras.optimizers.Adam(learning_rate=network_config.initial_learning_rate)
training_metric = tf.keras.metrics.MeanSquaredError()

#==============================================================================
# Start compressing data
print("-"*60,"\nCOMPRESSING DATA:")

# Create a dictionary of lists for storing training data
training_data = {"epoch": [],"loss": [],"time": [],"learning_rate": []}

# Enter the outer training loop: epochs:
for epoch in range(network_config.num_epochs):
    
    # Print Epoch information
    print("\nEpoch: {}/{}".format(epoch+1,network_config.num_epochs))
    training_data["epoch"].append(epoch)
      
    # Calculate, set, store and print the current optimiser learning rate
    learning_rate = network_config.initial_learning_rate / (2**(epoch // network_config.decay_rate))
    optimiser.lr.assign(learning_rate)
    training_data["learning_rate"].append(learning_rate)
    print("Learning Rate: {:.2E}".format(training_data["learning_rate"][epoch]))
    
    # Start timing
    epoch_start_time = time.time()
    
    #==========================================================================
    # Enter the inner training loop: batches:
    for batch, (volume_batch,values_batch,indices_batch) in enumerate(dataset):
        
        print("\rBatch: {}/{}".format(batch+1,len(dataset)),end="")
        
        #======================================================================
        # Open a 'GradientTape' to record the operations run during the forward
        # pass, which enables auto-differentiation
        with tf.GradientTape() as tape:
            
            # Compute a forward pass for the current mini-batch
            values_predicted = neur_comp(volume_batch,training=True)
            
            # Compute the losses for the current mini-batch
            loss_on_predicted = LossPred(values_batch,values_predicted)
            loss_on_gradients = LossGrad(values_batch,values_predicted)
            total_loss = loss_on_predicted

        #======================================================================
        # Use the gradient tape to automatically retrieve the gradients for the
        # trainable variables (weights/biases) with respect to loss
        gradients = tape.gradient(total_loss,neur_comp.trainable_weights)
        
        # Run a single step of gradient descent by updating the variable values
        # in order to minimise the total loss per mini-batch
        optimiser.apply_gradients(zip(gradients,neur_comp.trainable_weights))
                
        # Update the training metric for the current mini-batch results
        training_metric.update_state(values_batch,values_predicted)
            
    #==========================================================================  
    # Fetch, store, reset and print the training metric
    training_data["loss"].append(training_metric.result().numpy())
    training_metric.reset_states()
    print("Mean-Squared Error: {:.4f}".format(training_data["loss"][epoch]))
    
    # End timing and print the elapsed time
    epoch_end_time = time.time()
    training_data["time"].append((epoch_end_time-epoch_start_time))
    print("Training Time: {:.1f} secs".format(training_data["time"][epoch]))
    
#==============================================================================
# Save training data



#==============================================================================
# Start predicting data
print("-"*60,"\nPREDICTING")

# Predict 'flat_values' using the Neurcomp's learned weights and biases
output_data.flat_values = neur_comp.predict(input_data.flat_volume,batch_size=network_config.batch_size,verbose="0")

# Reshape 'flat_values' into the shape of the original input dimensions
output_data.values = np.reshape(output_data.flat_values,input_data.values.shape,order="C")

# Compute the peak signal-to-noise ratio of the predicted volume
psnr = LossPSNR(true=input_data.values,pred=output_data.values)
print("Compression PSNR: {:.4f}".format(psnr))

# Save the predicted values to a '.npy' volume rescaling as appropriate
output_data.SaveData(filepath="test_vol_out.npy",normalise=True)

#==============================================================================
print("-"*60)



# print("="*80,"\n")
# print("SAVING MODEL:\n")

# # Save the trained model, save network_config to CSV -------------------------

# SaveTrainedModel(NeurComp,filepaths,overwrite=True,save_format="tf")
# network_config.Savenetwork_config(filepaths)

# #------------------------------------------------------------------------------

# times.append(datetime.now())
# ElapsedTime(times)
# print("="*80,"\n")
# print("QUANTISATION:\n")

# # Make TFLite model, make quantised model, save both --------------------------

# NeurComp_tflite = MakeTFLiteModel(filepaths,overwrite=True)
# NeurComp_quantd = MakeQuantdModel(filepaths,overwrite=True)

# #------------------------------------------------------------------------------

# times.append(datetime.now())
# ElapsedTime(times)
# print("="*80,"\n")
# print("EVALUATING:\n")

# # Predict the volume using the normal, tflite and quantised models.............
# predicted_volume_normal = PredictNormal(NeurComp,input_data,network_config)
# predicted_volume_tflite = PredictTFLite(filepaths,input_data)
# predicted_volume_quantd = PredictQuantd(filepaths,input_data)

# #------------------------------------------------------------------------------

# times.append(datetime.now())
# ElapsedTime(times)
# print("="*80,"\n")
# print("SAVING FILES:\n")

# # Obtain the save filepath, save the predicted volumes for evaluation..........
# save_filepath = filepaths.output_volume_path

# for extension in [".npy",".txt",".bin"]:
#     input_data.SaveValues(save_filepath,"original",extension)
#     predicted_volume_normal.SaveValues(save_filepath,"normal",extension)
#     predicted_volume_tflite.SaveValues(save_filepath,"tflite",extension)
#     predicted_volume_quantd.SaveValues(save_filepath,"quantd",extension)

# #------------------------------------------------------------------------------

# times.append(datetime.now())
# ElapsedTime(times)
# print("="*80,"\n")

#=============================================================================#