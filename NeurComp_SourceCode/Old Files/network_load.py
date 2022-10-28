""" Created: 18.07.2022  \\  Updated: 18.07.2022  \\   Author: Robert Sales """

#=# IMPORT LIBRARIES #========================================================#

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from tensorflow import keras

#=# DEFINE FUNCTIONS #========================================================#

def ReconstructModel(filepath):
        
    if not os.path.exists(filepath):
        print("Cannot Reconstruct, Missing File'{}'.".format(filepath))
        return None
    else:
        print("Reconstructing Model: '{}'.".format(filepath))
        model = keras.models.load_model(filepath)

    return model
        
def LoadModelWeights():
    
    return None
#=============================================================================#