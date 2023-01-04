""" Created: 18.07.2022  \\  Updated: 10.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from pyevtk.hl import gridToVTK

#==============================================================================
# Define a class for managing datasets: i.e. loading, handling and storing data  

class DataClass():

    #==========================================================================
    # Define the initialisation constructor function for 'DataClass'   

    def __init__(self):
        pass
    
    #==========================================================================
    # Define a function to load an arbitrary dimension scalar field and extract 
    # useful meta-data. 
    
    # -> Note: The input filepath must refer to a file ending in '.npy' 
    
    # -> Note: Make sure the flattening takes place after normalisation
    
    def LoadData(self,volume_path,i_dimensions,o_dimensions,normalise=True):
                
        print("\n{:30}{}".format("Loaded data:",volume_path.split("/")[-1]))
    
        # Set the input and output dimensions defined by user input
        self.i_dimensions = i_dimensions
        self.o_dimensions = o_dimensions
    
        # Load the data then extract the volume and values matrices
        data = np.load(volume_path)    
        self.volume = data[...,:self.i_dimensions]
        self.values = data[...,self.i_dimensions:]
        
        # Determine the input resolution and number of input values
        self.resolution = self.values.shape[:-1]  
        self.i_size = self.values.size
        
        # Determine the maximum, minimum, average and range of 'volume'
        self.volume_max = np.array([self.volume[...,i].max() for i in np.arange(self.i_dimensions)])
        self.volume_min = np.array([self.volume[...,i].min() for i in np.arange(self.i_dimensions)])
        self.volume_avg = (self.volume_max+self.volume_min)/2.0
        self.volume_rng = abs(self.volume_max-self.volume_min)
        
        # Determine the maximum, minimum, average and range of 'values'
        self.values_max = np.array([self.values[...,i].max() for i in np.arange(self.o_dimensions)])
        self.values_min = np.array([self.values[...,i].min() for i in np.arange(self.o_dimensions)])
        self.values_avg = (self.values_max+self.values_min)/2.0
        self.values_rng = abs(self.values_max-self.values_min)
        
        # Normalise 'volume' and 'values' to the range [-1,+1], or do nothing
        if normalise:
            self.volume = 2.0*((self.volume-self.volume_avg)/(self.volume_rng))        
            self.values = 2.0*((self.values-self.values_avg)/(self.values_rng))
        else: pass
    
        # Flatten 'volume' and 'values' into equal lists of vectors
        self.flat_volume = np.reshape(np.ravel(self.volume,order="C"),(-1,self.volume.shape[-1]))
        self.flat_values = np.reshape(np.ravel(self.values,order="C"),(-1,self.values.shape[-1]))
                
        return None
       
    #==========================================================================
    # Define a function to copy attributes from 'DataClassObject' to 'self' but
    # without referencing 'DataClassObject' itself (so attributes in 'self' can 
    # be safely changed without changing those in 'DataClassObject')
    
    # -> Note: 'getattr()' and 'setattr()' are used to copy without referencing 
    
    def CopyMetaData(self,DataClassObject):
        
        # Extract attribute keys from 'DataObject' and define exceptions
        exception_keys = ["values","flat_values"]
        attribute_keys = DataClassObject.__dict__.keys()
        
        # Iterate through the list of attribute keys
        for key in attribute_keys:
            
            # Copy the attribute if the key is not in 'exception_keys'
            if key not in exception_keys:
                setattr(self,key,getattr(DataClassObject,key))
            else: pass
            
        return None
    
    #==========================================================================
    # Define a function to concatenate and save a scalar field to a '.npy' file
    
    # -> Note: 'volume' and 'values' must be reshaped to match the input shapes
    
    def SaveData(self,output_volume_path,reverse_normalise=True):
                    
        # Reverse normalise 'volume' and 'values' to the initial ranges
        if reverse_normalise:
            self.volume = ((self.volume_rng*(self.volume/2.0))+self.volume_avg)
            self.values = ((self.values_rng*(self.values/2.0))+self.values_avg)
        else: pass

        # Save as Numpy file 
        np.save(output_volume_path,np.concatenate((self.volume,self.values),axis=-1))
        
        # Save as VTK file
        volume_list = [np.ascontiguousarray(self.volume[...,x]) for x in range(self.i_dimensions)]
        values_dict = {"values":np.ascontiguousarray(self.values[...,0])}
        gridToVTK(output_volume_path,*volume_list,pointData=values_dict)

        return None
    
    #==========================================================================
    # Define a function to create and return a 'tf.data.Dataset' dataset object
    
    # -> Note: 'AUTOTUNE' prompts 'tf.data' to tune the value  of 'buffer_size'
    # -> dynamically at runtime
    
    # -> Note: Moving 'dataset.cache()' up/down will reduce runtime performance
    # -> significantly

    def MakeDataset(self,batch_size,repeat=False):
        
        print("\n{:30}{}{}".format("Made TF dataset:","batch_size = ",batch_size))
        
        # Create a dataset whose elements are slices of the given tensors
        dataset = tf.data.Dataset.from_tensor_slices((self.flat_volume,self.flat_values))
        
        # Cache the elements of the dataset to increase runtime performance
        dataset = dataset.cache()
        
        # Makes the dataset infinitely iterable (i.e. infinitely repeating)
        if repeat: 
            dataset = dataset.repeat(count=None)
        else: pass
        
        # Randomly shuffle the elements of the cached dataset 
        dataset = dataset.shuffle(buffer_size=self.i_size,reshuffle_each_iteration=True)
                    
        # Concatenate elements of the dataset into mini-batches
        dataset = dataset.batch(batch_size=batch_size,drop_remainder=False)
        
        # Pre-fetch elements from the dataset to increase throughput
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
                
        return dataset 
        
#=============================================================================#
    