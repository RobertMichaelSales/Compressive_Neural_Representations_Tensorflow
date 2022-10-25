""" Created: 18.07.2022  \\  Updated: 25.10.2022  \\   Author: Robert Sales """

trip_error

#==============================================================================
# Import user-defined libraries 

from callback_functions      import *
from error_functions         import *
from network_load            import *
from network_make            import *
from network_save            import *
from predict_volumes         import *
from utility_functions       import *

from data_management         import DataClass
from network_configuration   import NetworkConfigClass
from file_management         import FileStructureClass

#==============================================================================
# Import libraries and set flags

import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0],True)

from datetime import datetime
times=[]

from utility_functions import LoggerClass
sys.stdout = LoggerClass()

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
print("NEURCOMP COMPRESSION: IMPLICIT NEURAL REPRESENTATIONS (Version 1.0) \n")

#------------------------------------------------------------------------------

times.append(datetime.now())
print("="*80,"\n")
print("INITIALISING FILES AND MODEL:\n")

# Declare input data, load input data from file -------------------------------

input_data=DataClass()
input_data.LoadValues(input_data_filepath)

# Declare network_config, set all network_config ----------------------------

network_config=NetworkConfigClass()
network_config.NetworkStructure(input_data)

# Create a Tensorflow dataset for training ------------------------------------

dataset=input_data.MakeDataset(network_config)

# Declare filepaths, create the training folders, create filepaths ------------

filepaths=FileStructureClass(parent_directory,network_config.network_save_name)

# Build NeurComp, select the error function, compile the network --------------

NeurComp = BuildNeurComp(network_config)

NeurComp.compile(
    loss=GetErrorFunction(network_config),
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=network_config.learn_rate),
    metrics=[tf.keras.metrics.MeanSquaredError(name='mse',dtype=None)]
)

#------------------------------------------------------------------------------

times.append(datetime.now())
ElapsedTime(times)
print("="*80,"\n")
print("COMPRESSING DATA:")

# Retrieve training callbacks, commence training ------------------------------

NeurComp.fit(dataset,
             batch_size=network_config.batch_size,
             epochs=network_config.max_epochs,
             verbose=2,
             callbacks=TrainingCallbacks(network_config,filepaths),
             initial_epoch=0)

#------------------------------------------------------------------------------

times.append(datetime.now())
ElapsedTime(times)
print("="*80,"\n")
print("SAVING MODEL:\n")

# Save the trained model, save network_config to CSV -------------------------

SaveTrainedModel(NeurComp,filepaths,overwrite=True,save_format="tf")
network_config.Savenetwork_config(filepaths)

#------------------------------------------------------------------------------

times.append(datetime.now())
ElapsedTime(times)
print("="*80,"\n")
print("QUANTISATION:\n")

# Make TFLite model, make quantised model, save both --------------------------

NeurComp_tflite = MakeTFLiteModel(filepaths,overwrite=True)
NeurComp_quantd = MakeQuantdModel(filepaths,overwrite=True)

#------------------------------------------------------------------------------

times.append(datetime.now())
ElapsedTime(times)
print("="*80,"\n")
print("EVALUATING:\n")

# Predict the volume using the normal, tflite and quantised models.............
predicted_volume_normal = PredictNormal(NeurComp,input_data,network_config)
predicted_volume_tflite = PredictTFLite(filepaths,input_data)
predicted_volume_quantd = PredictQuantd(filepaths,input_data)

#------------------------------------------------------------------------------

times.append(datetime.now())
ElapsedTime(times)
print("="*80,"\n")
print("SAVING FILES:\n")

# Obtain the save filepath, save the predicted volumes for evaluation..........
save_filepath = filepaths.output_volume_path

for extension in [".npy",".txt",".bin"]:
    input_data.SaveValues(save_filepath,"original",extension)
    predicted_volume_normal.SaveValues(save_filepath,"normal",extension)
    predicted_volume_tflite.SaveValues(save_filepath,"tflite",extension)
    predicted_volume_quantd.SaveValues(save_filepath,"quantd",extension)

#------------------------------------------------------------------------------

times.append(datetime.now())
ElapsedTime(times)
print("="*80,"\n")

#=============================================================================#