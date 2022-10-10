""" Created: 18.07.2022  \\  Updated: 18.07.2022  \\   Author: Robert Sales """

trip_error

#=# IMPORT LIBRARIES #========================================================#

from callback_functions import *
from data_management import *
from error_functions import *
from network_hyperparameters import *
from network_load import *
from network_make import *
from network_save import *
from predict_volumes import *
from file_management import *
from utility_functions import *

#=# SET RUNTIME FLAGS #=======================================================#

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

#=# SET FILEPATHS #===========================================================#

parent_dir_filepath = ("C:\\"
                       "Users\\"
                       "sales\\"
                       "Documents\\"
                       "Cambridge University\\"
                       "PPD - Project Proposal Dissertation\\"
                       "NeurComp Files")

input_data_filepath = ("C:\\"
                       "Users\\"
                       "sales\\"
                       "Documents\\"
                       "Cambridge University\\"
                       "PPD - Project Proposal Dissertation\\"
                       "NeurComp Files\\"
                       "Inputs\\"
                       "Generic\\"
                       "data_orig.npy")

#=# MAIN SCRIPT #=============================================================#

print("="*80,"\n")
print("NEURCOMP COMPRESSION: IMPLICIT NEURAL REPRESENTATIONS (Version 1.0) \n")

#------------------------------------------------------------------------------

times.append(datetime.now())
print("="*80,"\n")
print("INITIALISING FILES AND MODEL:\n")

# Declare input data, load input data from file -------------------------------

input_data=DataClass()
input_data.LoadValues(input_data_filepath)

# Declare hyperparameters, set all hyperparameters ----------------------------

hyperparameters=HyperparameterClass()
hyperparameters.NetworkDetails(input_data)

# Create a Tensorflow dataset for training ------------------------------------

dataset=MakeDataset(input_data,hyperparameters)

# Declare filepaths, create the training folders, create filepaths ------------

filepaths=FilepathClass(parent_dir_filepath,hyperparameters)

# Build NeurComp, select the error function, compile the network --------------

NeurComp = BuildNeurComp(hyperparameters)

NeurComp.compile(
    loss=GetErrorFunction(hyperparameters),
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=hyperparameters.learn_rate),
    metrics=[tf.keras.metrics.MeanSquaredError(name='mse',dtype=None)]
)

#------------------------------------------------------------------------------

times.append(datetime.now())
ElapsedTime(times)
print("="*80,"\n")
print("COMPRESSING DATA:")

# Retrieve training callbacks, commence training ------------------------------

NeurComp.fit(dataset,
             batch_size=hyperparameters.batch_size,
             epochs=hyperparameters.max_epochs,
             verbose=2,
             callbacks=TrainingCallbacks(hyperparameters,filepaths),
             initial_epoch=0)

#------------------------------------------------------------------------------

times.append(datetime.now())
ElapsedTime(times)
print("="*80,"\n")
print("SAVING MODEL:\n")

# Save the trained model, save hyperparameters to CSV -------------------------

SaveTrainedModel(NeurComp,filepaths,overwrite=True,save_format="tf")
hyperparameters.SaveHyperparameters(filepaths)

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
predicted_volume_normal = PredictNormal(NeurComp,input_data,hyperparameters)
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