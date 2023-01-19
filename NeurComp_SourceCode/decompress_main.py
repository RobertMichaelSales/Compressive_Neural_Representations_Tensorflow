""" Created: 16.11.2022  \\  Updated: 19.01.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

#==============================================================================
# Import user-defined libraries 

from data_management         import DataClass,SaveData
from network_decoder         import DecodeParameters,DecodeArchitecture,AssignParameters
from network_model           import ConstructNetwork

#==============================================================================

def decompress(architecture_path,parameters_path,input_data_path):
        
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
    # Initialise i/o 
    
    print("-"*80,"\nINITIALISING DATA I/O:")
    
    # Create 'DataClass' objects to store i/o volume and values
    i_volume = DataClass(data_type="volume")
    o_values = DataClass(data_type="values")
    
    # Load and normalise input data
    i_volume.LoadData(input_data_path=input_data_path,dimensions=(3,1),normalise=True)
    
    #==========================================================================
    # Construct the values
    print("-"*80,"\nCONSTRUCTING VALUES:")  

    # redict values from the input volume
    o_values.flat = SquashNet.predict(i_volume.flat,batch_size=1024,verbose="1")
    o_values.data = np.reshape(o_values.flat,(i_volume.resolution+(1,)),order="C")
    
    # Determine the maximum, minimum, average and range of 'values'
    o_values.max = values_bounds[0]
    o_values.min = values_bounds[1]
    o_values.avg = (o_values.max+o_values.min)/2.0
    o_values.rng = abs(o_values.max-o_values.min)
    
    #==========================================================================
    # Save decompressed output
    print("-"*80,"\nSAVING DECOMPRESSED OUTPUT:")  

    # Save the output volume to ".npy" and ".vtk" files
    output_data_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/test"
    SaveData(output_data_path=output_data_path,volume=i_volume,values=o_values,reverse_normalise=True)
    print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1]))
    
    #==========================================================================
    print("-"*80,"\n")
    
    return None

#==============================================================================
# Define the script to run when envoked from the terminal

if __name__=="__main__":
    
    import sys
        
    # Set architecture filepath
    architecture_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/neurcomp_test/architecture.bin"
    # architecture_path = sys.argv[1]
    
    # Set parameters filepath
    parameters_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/neurcomp_test/parameters.bin"
    # parameters_path = sys.argv[2]
    
    # Set input filepath
    input_data_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/volume.npy"
    # input_data_path = sys.argv[3]
    
    # Execute decompression
    decompress(architecture_path=architecture_path,parameters_path=parameters_path,input_data_path=input_data_path)   

else: pass
    
#==============================================================================