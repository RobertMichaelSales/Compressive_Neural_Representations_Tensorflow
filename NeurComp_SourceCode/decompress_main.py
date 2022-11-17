""" Created: 16.11.2022  \\  Updated: 16.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import user-defined libraries 

from network_decoder         import DecodeWeights,DecodeArchitecture,AssignWeights
from network_make            import BuildNeurComp

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

#==============================================================================

def decompress(base_directory,architecture_filepath,weights_filepath,input_filepath):
    
    # TODO:
        
        # Some way to convey batch size
        
    
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
    print("-"*80,"\nLOADING ARCHITECTURE:")    
    
    print("\n{:30}{}".format("Loaded architecture from:",architecture_filepath.split("/")[-1]))
    layer_dimensions = DecodeArchitecture(filepath=architecture_filepath)
    print("\n{:30}{} bytes".format("File size:",os.path.getsize(architecture_filepath)))
    
    #==========================================================================
    # Rebuilding network from architecture
    print("-"*80,"\nCONSTRUCTING NETWORK:")  
    
    print("\n{:30}{}".format("Constructed network:","done"))
    neur_comp = BuildNeurComp(layer_dimensions=layer_dimensions)
    print("\n{:30}{}".format("Network dimensions:",layer_dimensions))
    
    #==========================================================================
    # Decoding weights and biases
    print("-"*80,"\nLOADING WEIGHTS/BIASES:")  
    
    print("\n{:30}{}".format("Loaded weights/biases fron:",weights_filepath.split("/")[-1]))
    network_weights = DecodeWeights(network=neur_comp,filepath=weights_filepath)    
    print("\n{:30}{} bytes".format("File size:",os.path.getsize(weights_filepath)))
          
    #==========================================================================
    # Rebuilding network from architecture
    print("-"*80,"\nASSIGNING WEIGHTS/BIASES:")  
    
    print("\n{:30}{}".format("Assigned weights/biases:","done"))
    neur_comp = AssignWeights(network=neur_comp,weights_dict=network_weights)   
    print("\n{:30}{}".format("Total parameters:",np.sum([np.prod(x.shape) for x in neur_comp.get_weights()])))
    
    #==========================================================================
    # Construct the volume
    print("-"*80,"\nCONSTRUCTING VOLUME:")  
    
    # For now, just load the input volume and values
    data = np.load(input_filepath)
    input_volume = data[...,:-1]
    input_values = data[...,-1:]
    
    # For now, get the minimum and maximum volume values
    volume_max = np.amax(input_volume,axis=tuple(np.arange(3)))
    volume_min = np.amin(input_volume,axis=tuple(np.arange(3)))
    volume_avg = (volume_max+volume_min)/2.0
    volume_rng = abs(volume_max-volume_min)
    
    # For now, get the minimum and maximum values values
    values_max = input_values.max()
    values_min = input_values.min()
    values_avg = (values_max+values_min)/2.0
    values_rng = abs(values_max-values_max)
    
    # For now, normalise the volume and values
    input_volume = (2.0*((input_volume-volume_avg)/(volume_rng)))
    
    # Flatten the input volume
    flat_volume = np.reshape(np.ravel(input_volume,order="C"),(-1,input_volume.shape[-1]),order="C")

    #==========================================================================
    # Construct the values
    print("-"*80,"\nCONSTRUCTING VALUES:")  

    # For now, just predict values from the input volume
    flat_values = neur_comp.predict(flat_volume,batch_size=1024,verbose="1")
    output_values = np.reshape(flat_values,(input_volume.shape[:-1]+(1,)),order="C")
    
    # For now, denormalise the values
    output_values = (((output_values/2.0)*values_rng)+values_avg)
    
    #==========================================================================
    # Save decompressed output
    print("-"*80,"\nSAVING DECOMPRESSED RESULTS:")  

    # Save as Numpy file 
    np.save("test",np.concatenate((input_volume,output_values),axis=-1))
    
    #==========================================================================
    
    return flat_volume
    
#==============================================================================
# Define the script to run when envoked from the terminal

if __name__=="__main__":

    # Set base directory
    base_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"
    
    # Set architecture filepath
    architecture_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/neurcomp_test/network_architecture"
 
    # Set weights filepath
    weights_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/neurcomp_test/network_weights"
    
    # Set input filepath
    input_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/test_vol.npy"
    
    flatvol2 = decompress(base_directory=base_directory,architecture_filepath=architecture_filepath,weights_filepath=weights_filepath,input_filepath=input_filepath)   

else:
    
    print("Please invoke 'decompress_main.py' correctly from the terminal!")
    
#==============================================================================