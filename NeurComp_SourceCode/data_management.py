""" Created: 18.07.2022  \\  Updated: 25.10.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

#==============================================================================
# Define a class for managing datasets: i.e. loading, handling and storing data  

class DataClass():

    #==========================================================================
    # Define the initialisation constructor function for 'DataClass'   

    def __init__(self):
        
        # Initialise internal variables for n-dimensional positional data 
        self.volume = np.array          
        self.volume_max = np.array     
        self.volume_min = np.array     
        self.volume_avg = np.array
        self.volume_rng = np.array
        
        # Initialise internal variables for one-dimensional scalar values
        self.values = np.array          
        self.values_max = 0.0           
        self.values_min = 0.0  
        self.values_avg = 0.0
        self.values_rng = 0.0
        
        # Initialise internal variables for the input shape, size and dimension
        self.input_resolution = np.array      
        self.input_dimensions = 3     
        self.input_size = 0

        # Initialise internal variables for the output dimension
        self.output_dimensions = 1        
        
        return None
    
    #==========================================================================
    # Define a function to load an arbitrary dimension scalar field and extract 
    # useful meta-data. Note: 'filepath' must point to a file ending in '*.npy' 
    
    def LoadValues(self,filepath):
        
        print("Loading Data: '{}'.\n".format(filepath))
        
        # Determine the file extension (type) from the provided file path
        extension = filepath.split(".")[-1].lower()
        
        # If the extension matches ".npy" then load it, else throw an error
        if extension == "npy":  
            data = np.load(filepath)            
        else:
            print("Error: File Type Not Supported: '{}'. ".format(extension))
            return None
        
        # Extract the positional data (i.e. volume) and scalars (i.e. values)
        volume = data[...,:-1]                 
        values = data[...,-1:]                 
        
        # Determine the input resolution, number of dimensions and input size
        self.input_resolution = volume.shape[:-1]    
        self.input_dimensions = volume.shape[ -1] 
        self.input_size = values.size
        
        # Determine the output dimension
        self.output_dimensions = values.shape[-1]
        
        # Determine the maximum, minimum, average and range values of 'volume'
        self.volume_max = np.amax(volume,axis=tuple(np.arange(self.input_dimensions)))
        self.volume_min = np.amin(volume,axis=tuple(np.arange(self.input_dimensions)))
        self.volume_avg = (self.volume_max + self.volume_min) / 2.0
        self.volume_rng = abs(self.volume_max - self.volume_min)
        
        # Determine the maximum, minimum, average and range values of 'values'
        self.values_max = values.max()
        self.values_min = values.min()
        self.values_avg = (self.values_max + self.values_min) / 2.0
        self.values_rng = abs(self.values_max - self.values_min)
        
        # Normalise 'volume' and 'values' to the range [-1,+1]
        self.volume = 2.0 * ((volume - self.volume_avg) / (self.volume_rng))        
        self.values = 2.0 * ((values - self.values_avg) / (self.values_rng))
        
        return None
    
    #==========================================================================
    # Define a function to concatenate and save a scalar field to a '.npy' file
    
    # -> Note: 'self.volume' and 'self.values' should be appropriately reshaped
    # -> such that they have the same shape as the input scalar field.
    
    def SaveValues(self,filepath):
        
        print("Saving Data: '{}'.\n".format(filepath))
        
        # Determine the file extension (type) from the provided file path
        extension = filepath.split(".")[-1].lower()
        
        # If the extension matches ".npy" then save it else throw an error
        if extension == "npy":  
            data = np.concatenate((self.volume,self.values),axis=-1)
            np.save(filepath,data)
        else:
            print("Error: File Type Not Supported: '{}'. ".format(extension))
            return None

        return None
    
    #==========================================================================
    # Define a function to create and return a 'tf.data.Dataset' dataset object     
    
    # -> Note: 'AUTOTUNE' prompts 'tf.data' to tune the value  of 'buffer_size'
    # -> dynamically at runtime.
    
    def MakeDataset(self,hyperparameters):
          
        # Reshape 'volume' and 'values' into lists of vectors and scalars
        flat_volume = self.volume.reshape(-1,self.volume.shape[-1]) 
        flat_values = self.values.reshape(-1,self.values.shape[-1])
        
        # Create a dataset whose elements are slices of the given tensors
        dataset = tf.data.Dataset.from_tensor_slices((flat_volume,flat_values))
        
        # Cache the elements of the dataset to increase runtime performance
        dataset = dataset.cache()
        
        # Randomly shuffle the elements of the cached dataset 
        dataset = dataset.shuffle(buffer_size=flat_values.size,
                                  reshuffle_each_iteration=True)
        
        # Concatenate elements of the dataset into mini-batches
        dataset = dataset.batch(batch_size=hyperparameters.batch_size,
                                drop_remainder=False)
        
        # Pre-fetch elements from the dataset to increase throughput
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
                
        return dataset 
        
#=============================================================================#