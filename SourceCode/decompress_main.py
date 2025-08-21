""" Created: 16.11.2022  \\  Updated: 23.10.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

#==============================================================================
# Import user-defined libraries 

from network_decoder         import DecodeParameters,DecodeArchitecture,AssignParameters
from network_model           import ConstructNetwork

#==============================================================================

architecture_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/AuxFiles/outputs/nodule_iso/architecture.bin"

parameters_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/AuxFiles/outputs/nodule_iso/parameters.bin"

##

layer_dimensions,frequencies = DecodeArchitecture(architecture_path=architecture_path)
  
SquashISO = ConstructNetwork(layer_dimensions=layer_dimensions,frequencies=frequencies) 

parameters,original_values_bounds,original_coords_centre,original_coords_radius = DecodeParameters(network=SquashISO,parameters_path=parameters_path)

AssignParameters(network=SquashISO,parameters=parameters)  

#==============================================================================

w = SquashISO.get_weights()[0::2]
b = SquashISO.get_weights()[1::2]
a = SquashISO.get_weights()

#==============================================================================

x = np.array([[1.0, 1.0, 1.0]])

np.dot(x,w[0]) + b[0]

#==============================================================================

SquashISO(x)
