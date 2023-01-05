""" Created: 16.11.2022  \\  Updated: 05.01.2023  \\   Author: Robert Sales """

# Separate volume and values (to be separate objects) in DataClass. Then move SaveData() from within DataClass to training_utils.py. Then rename training_utils.py to compress_utils.py.
# Separate volume and values (to be separate objects) in DataClass. Then move SaveData() from within DataClass to training_utils.py. Then rename training_utils.py to compress_utils.py.
# Separate volume and values (to be separate objects) in DataClass. Then move SaveData() from within DataClass to training_utils.py. Then rename training_utils.py to compress_utils.py.
# Separate volume and values (to be separate objects) in DataClass. Then move SaveData() from within DataClass to training_utils.py. Then rename training_utils.py to compress_utils.py.
# Separate volume and values (to be separate objects) in DataClass. Then move SaveData() from within DataClass to training_utils.py. Then rename training_utils.py to compress_utils.py.

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

def decompress(architecture_path,parameters_path,volume_path):
        
    print("-"*80,"\nSQUASHNET: IMPLICIT NEURAL REPRESENTATIONS (by Rob Sales)")
    
    #==========================================================================
    # Check whether hardware acceleration is enabled
   
    gpus = tf.config.list_physical_devices('GPU')
    
    if (len(gpus) != 0): 
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Enabled"))
        tf.config.experimental.set_memory_growth(gpus[0],True)
    else:        
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Disabled"))
        return None
           
    #==========================================================================
    # Construct network
    print("-"*80,"\nCONSTRUCTING NETWORK:")  
    
    layer_dimensions = DecodeArchitecture(architecture_path=architecture_path)
    print("\n{:30}{}".format("Loaded architecture from:",architecture_path.split("/")[-1]))
    
    SquashNet = ConstructNetwork(layer_dimensions=layer_dimensions)
    print("\n{:30}{}".format("Network dimensions:",layer_dimensions))
    
    #==========================================================================
    # Assign parameters 
    print("-"*80,"\nSETTING PARAMETERS:") 
    
    parameters,values_bounds = DecodeParameters(network=SquashNet,parameters_path=parameters_path)
    print("\n{:30}{}".format("Loaded parameters from:",parameters_path.split("/")[-1])) 

    AssignParameters(network=SquashNet,parameters=parameters)  
    print("\n{:30}{}".format("Total parameters:",np.sum([np.prod(x.shape) for x in SquashNet.get_weights()])))    
    
    #==========================================================================
    
    # Construct the volume
    print("-"*80,"\nCONSTRUCTING VOLUME:")  
    
    # Extract the input and output dimensions from the network dimensions
    i_dimensions = layer_dimensions[ 0]
    o_dimensions = layer_dimensions[-1]
    
    # Load the input volume only
    data = np.load(volume_path)    
    volume = data[...,:i_dimensions]
    resolution = volume.shape[:-1]  
        
    # Determine the maximum, minimum, average and range of 'volume'
    volume_max = np.array([volume[...,i].max() for i in np.arange(i_dimensions)])
    volume_min = np.array([volume[...,i].min() for i in np.arange(i_dimensions)])
    volume_avg = (volume_max+volume_min)/2.0
    volume_rng = abs(volume_max-volume_min)    
    
    # Normalise the volume 
    volume = (2.0*((volume-volume_avg)/(volume_rng)))

    # Flatten the input volume
    flat_volume = np.reshape(np.ravel(volume,order="C"),(-1,i_dimensions),order="C")

    #==========================================================================
    # Construct the values
    print("-"*80,"\nCONSTRUCTING VALUES:")  

    # redict values from the input volume
    flat_values = SquashNet.predict(flat_volume,batch_size=1024,verbose="1")
    values = np.reshape(flat_values,(resolution+(o_dimensions,)),order="C")
    
    #==========================================================================
    # Reverse normalise volume and values
    
    # Determine the maximum, minimum, average and range of 'values'
    values_max = values_bounds[0]
    values_min = values_bounds[1]
    values_avg = (values_max+values_min)/2.0
    values_rng = abs(values_max-values_min)
    
    # For now, denormalise the values
    values = (((values/2.0)*values_rng)+values_avg)
    
    #==========================================================================
    # Save decompressed output
    print("-"*80,"\nSAVING DECOMPRESSED RESULTS:")  

    # Save as Numpy file 
    output_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/test"
    np.save(output_path,np.concatenate((volume,values),axis=-1))
        
    #==========================================================================
    print("-"*80,"\n")
    
    return None

#==============================================================================
# Define the script to run when envoked from the terminal

if __name__=="__main__":
    
    # Set architecture filepath
    architecture_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/neurcomp_test/architecture.bin"
    
    # Set parameters filepath
    parameters_path   = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/neurcomp_test/parameters.bin"
    
    # Set input filepath
    volume_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/volume.npy"

    # Execute decompression
    decompress(architecture_path=architecture_path,parameters_path=parameters_path,volume_path=volume_path)   

else: pass
    
#==============================================================================