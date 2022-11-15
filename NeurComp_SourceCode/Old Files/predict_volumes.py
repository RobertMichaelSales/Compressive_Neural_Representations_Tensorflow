""" Created: 18.07.2022  \\  Updated: 18.07.2022  \\   Author: Robert Sales """

#=# IMPORT LIBRARIES AND SET RUNTIME FLAGS #==================================#

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from tensorflow import keras 
from data_management import *
from utility_functions import *

#=# DEFINE FUNCTIONS #========================================================#

def PredictNormal(model,input_data,hyperparameters):
    
    print("Predicting Data: Normal Model.\n")
    
    # Declare the volume data class
    predicted_volume_normal = DataClass()
    
    # Copy over the input volume
    predicted_volume_normal.volume = input_data.volume
    
    # Evaluate the output values
    predicted_volume_normal.values = model.predict(input_data.volume,
                                                   batch_size=hyperparameters.batch_size,
                                                   verbose=0)
    
    return predicted_volume_normal


def PredictTFLite(filepaths,input_data,batch=True):
    
    print("Predicting Data: TFLite Model.\n")
    
    # Declare the volume data class
    predicted_volume_tflite = DataClass()
    
    # Copy over the input volume, set up the output values
    predicted_volume_tflite.volume = input_data.volume
    predicted_volume_tflite.values = np.zeros(shape=input_data.values.shape,dtype=np.float32)
    
    # Load the model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=filepaths.tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    # Iterate through all the volume coordinates
    for index,coordinate in enumerate(input_data.volume):
        
        # Construct an input tensor from coordinates
        input_tensor = np.array([coordinate],dtype=np.float32)
      
        # Set the input tensor within the network 
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
      
        # Invoke the interpreter
        interpreter.invoke()
      
        # Retreieve the output tensor
        output_tensor = interpreter.get_tensor(output_details[0]['index'])
      
        # Condition the output
        output_scalar = np.squeeze(output_tensor)
      
        # Store the output in the volume array
        predicted_volume_tflite.values[index] = output_scalar
      
        # Print loading bar
        #if (index % 10000 == 0): LoadingBar(index,input_data.values.size,30)
       
    #print("\n")    
       
    return predicted_volume_tflite


def PredictQuantd(filepaths,input_data,batch=True):
    
    print("Predicting Data: Quantised Model.\n")
    
    # Declare the volume data class
    predicted_volume_quantd = DataClass()
    
    # Copy over the input volume, set up the output values
    predicted_volume_quantd.volume = input_data.volume
    predicted_volume_quantd.values = np.zeros(shape=input_data.values.shape,
                                              dtype=np.float32)
    
    # Load the model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=filepaths.quantd_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    # Iterate through all the volume coordinates
    for index,coordinate in enumerate(input_data.volume):
        
        # Construct an input tensor from coordinates
        input_tensor = np.array([coordinate],dtype=np.float32)
      
        # Set the input tensor within the network 
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
      
        # Invoke the interpreter
        interpreter.invoke()
      
        # Retreieve the output tensor
        output_tensor = interpreter.get_tensor(output_details[0]['index'])
      
        # Condition the output
        output_scalar = np.squeeze(output_tensor)
      
        # Store the output in the volume array
        predicted_volume_quantd.values[index] = output_scalar
      
        # Print loading bar
        #if (index % 1000 == 0): LoadingBar(index,input_data.values.size,30)
        
    #print("\n")   
          
    return predicted_volume_quantd

#=============================================================================#