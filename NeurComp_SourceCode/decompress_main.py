""" Created: 16.11.2022  \\  Updated: 16.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import user-defined libraries 

from network_decoder         import DecodeWeights,DecodeArchitecture
from network_make            import BuildNeurComp

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

#==============================================================================

def decompress(base_directory,architecture_filepath,weights_filepath):
    
    #==========================================================================
    # Enter the main script and check for hardware acceleration
    print("-"*80,"\nNEURCOMP: IMPLICIT NEURAL REPRESENTATIONS (Version 2.0)")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if (len(gpus) == 0):        
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Disabled"))
    else:        
        tf.config.experimental.set_memory_growth(gpus[0],True)
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Enabled"))
        
    #==========================================================================
    # Decoding network architecture
    print("-"*80,"\nDECODING ARCHITECTURE:")    
    
    print("\n{:30}{}".format("Loading architecture from:",architecture_filepath.split("/")[-1]))
    print("\n{:30}{} bytes".format("File size:",os.path.getsize(architecture_filepath)))
    layer_dimensions = DecodeArchitecture(filepath=architecture_filepath)
    print("\n{:30}{}".format("Network dimensions:",layer_dimensions))
    
    #==========================================================================
    # Rebuilding network from architecture
    print("-"*80,"\nREBUILDING NETWORK:")  
    
    print("\n{:30}{}".format("Reconstructed network:","neur_comp"))
    neur_comp_r = BuildNeurComp(layer_dimensions)
    
    #==========================================================================
    # Rebuilding network from architecture
    print("-"*80,"\nDECODING PARAMETERS:")  
    
    print("\n{:30}{}".format("Loading weights/biases fron:",weights_filepath.split("/")[-1]))
    print("\n{:30}{} bytes".format("File size:",os.path.getsize(weights_filepath)))
    DecodeWeights(network=neur_comp_r,filepath=weights_filepath)    
    
    #==========================================================================
    
    return None
    
#==============================================================================
# Define the script to run when envoked from the terminal

if __name__=="__main__":

    # Set base directory
    base_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"
    
    # Set architecture filepath
    architecture_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/neurcomp_test/network_architecture"
 
    # Set weights filepath
    weights_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/neurcomp_test/network_weights"
    
    # decompress(base_directory=base_directory,architecture_filepath=architecture_filepath,weights_filepath=weights_filepath)   

else:
    
    print("Please invoke 'decompress_main.py' correctly from the terminal!")
    
#==============================================================================