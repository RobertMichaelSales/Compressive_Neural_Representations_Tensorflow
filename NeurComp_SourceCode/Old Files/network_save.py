""" Created: 18.07.2022  \\  Updated: 18.07.2022  \\   Author: Robert Sales """

#=# IMPORT LIBRARIES #========================================================#

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from datetime import datetime

#=# DEFINE FUNCTIONS #========================================================#

def SaveTrainedModel(model,filepaths,overwrite,save_format):
    
    if save_format == "h5": 
        filepath = filepaths.normal_model_path + ".h5"
    else:
        filepath = filepaths.normal_model_path
        
    print("Saving Trained Model: '{}'.\n".format(filepath))
    
    tf.keras.models.save_model(model,
                               filepath=filepath,
                               overwrite=overwrite,
                               include_optimizer=True,
                               save_format=save_format,
                               signatures=None,
                               options=None,
                               save_traces=True)

    return None


def MakeTFLiteModel(filepaths,overwrite):
    
    # Convert to a TFLite model
    converter = tf.lite.TFLiteConverter.from_saved_model(filepaths.normal_model_path)
    NeurComp_tflite = converter.convert()
    
    # Save the TFLite model
    if not os.path.exists(filepaths.tflite_model_path) or overwrite:
        print("Saving TFLite Model: '{}'.\n".format(filepaths.tflite_model_path))
        open(filepaths.tflite_model_path, "wb").write(NeurComp_tflite)
    else:
        print("Will Not Overwrite: '{}'.".format(filepaths.tflite_model_path))
    
    return NeurComp_tflite


def MakeQuantdModel(filepaths,overwrite):
    
    # Convert to a TFLite model
    converter = tf.lite.TFLiteConverter.from_saved_model(filepaths.normal_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    NeurComp_quantd = converter.convert()
    
    # Save the Quantised model
    if not os.path.exists(filepaths.quantd_model_path) or overwrite:
        print("Saving Quantised Model: '{}'.\n".format(filepaths.quantd_model_path))
        open(filepaths.quantd_model_path, "wb").write(NeurComp_quantd)
    else:
        print("Will Not Overwrite: '{}'.".format(filepaths.quantd_model_path))
    
    return NeurComp_quantd

#=============================================================================#