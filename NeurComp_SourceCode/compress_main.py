""" Created: 18.07.2022  \\  Updated: 26.10.2022  \\   Author: Robert Sales """

#==============================================================================
# Import user-defined libraries 

# from callback_functions      import *
# from network_load            import *
# from network_save            import *
# from predict_volumes         import *
# from utility_functions       import *

from data_management         import DataClass
from file_management         import FileStructureClass
from loss_functions          import LossPred, LossGrad
from network_configuration   import NetworkConfigClass
from network_make            import BuildNeurComp

#==============================================================================
# Import libraries and set flags

import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)

#from datetime import datetime
#times=[]

#from utility_functions import LoggerClass
#sys.stdout = LoggerClass()

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
# Main script

print("="*80,"\n")
print("NEURCOMP COMPRESSION: IMPLICIT NEURAL REPRESENTATIONS (Version 2.0) \n")

#------------------------------------------------------------------------------

# times.append(datetime.now())
print("="*80,"\n")
print("INITIALISING FILES AND MODEL:\n")

#==============================================================================
# Create a 'DataClass' object to store input data and load input data from file
input_data = DataClass()
input_data.LoadValues(input_data_filepath)

#==============================================================================
# Create a 'NetworkConfigClass' object to store the network hyperparameters and
# generate the network structure based on the input dimensions
network_config = NetworkConfigClass()
network_config.NetworkStructure(input_data)

#==============================================================================
# Use the '.MakeDataset' method to generate a Tensorflow '.data' dataset to use
# for supplying volume and values batches during training 
dataset = input_data.MakeDataset(network_config)

#==============================================================================
# Create a 'FileStructureClass' object, and use it's initialisation constructor
# function to generate new folders, filepaths and filenames 
filepaths = FileStructureClass(parent_directory,network_config.network_name)

#==============================================================================
# Build NeurComp from the information carried in 'network_config' and define an
# optimiser for training
neur_comp = BuildNeurComp(network_config)
optimiser = tf.keras.optimizers.Adam(learning_rate=network_config.learn_rate)

#==============================================================================
# Enter the outer training loop: epochs
for epoch in range(network_config.num_training_epochs):
    
    print("\nEpoch: {}/{}".format(epoch,network_config.num_training_epochs))
    
    #==========================================================================
    # Enter the inner training loop: batches 
    for batch, (volume_minibatch,values_minibatch) in enumerate(dataset):
        
        print("\nBatch: {}".format(batch))
        
        #======================================================================
        # Open a 'GradientTape' to record the operations run during the forward
        # pass, which enables auto-differentiation
        with tf.GradientTape() as tape:
            
            # Compute a forward pass for the current mini-batch
            values_predicted = neur_comp(volume_minibatch,training=True)
            
            # Compute the losses for the current mini-batch
            loss_on_predicted = LossPred(values_minibatch,values_predicted)
            loss_on_gradients = LossGrad(values_minibatch,values_predicted)
            total_loss = loss_on_predicted

        #======================================================================
        # Use the gradient tape to automatically retrieve the gradients for the
        # trainable variables (weights/biases) with respect to loss
        gradients = tape.gradient(total_loss,neur_comp.trainable_weights)
        
        # Run a single step of gradient descent by updating the variable values
        # in order to minimise the total loss per mini-batch
        optimiser.apply_gradients(zip(gradients,neur_comp.trainable_weights))
                
    #==========================================================================

#==============================================================================

TRIP_ERROR # <- CONTINUE FROM HERE <- CONTINUE FROM HERE <- CONTINUE FROM HERE

#------------------------------------------------------------------------------

#times.append(datetime.now())
#ElapsedTime(times)
print("="*80,"\n")
print("COMPRESSING DATA:")

# Retrieve training callbacks, commence training ------------------------------

NeurComp.fit(dataset,
             batch_size=network_config.batch_size,
             epochs=network_config.max_epochs,
             verbose=2,
             callbacks=TrainingCallbacks(network_config,filepaths),
             initial_epoch=0)

# #------------------------------------------------------------------------------

# #times.append(datetime.now())
# #ElapsedTime(times)
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