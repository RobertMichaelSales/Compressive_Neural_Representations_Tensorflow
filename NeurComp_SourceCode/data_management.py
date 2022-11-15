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
                
        # Initialise internal variables for n-dimensional positional data 
        self.volume,self.flat_volume = np.array,np.array    
        self.volume_max = np.array     
        self.volume_min = np.array     
        self.volume_avg = np.array
        self.volume_rng = np.array
        
        # Initialise internal variables for one-dimensional scalar values
        self.values,self.flat_values = np.array,np.array  
        self.values_max = 0.0           
        self.values_min = 0.0  
        self.values_avg = 0.0
        self.values_rng = 0.0
        
        # Initialise internal variables for input and output meta-data
        self.resolution = np.array      
        self.input_dimensions = 3     
        self.size = 0
        self.output_dimensions = 1  
        
        # Initialise the array of index positions for the positional data
        self.indices,self.flat_indices = np.array,np.array            
        
        return None
    
    #==========================================================================
    # Define a function to load an arbitrary dimension scalar field and extract 
    # useful meta-data. Note: 'filepath' must point to a file ending in '*.npy' 
    
    # Note: The '*' in the definition of 'self.indices' treats lists and tuples
    # as consecutive arguments so 'print(*[1,2,3])' gives '1 2 3' not '[1,2,3]'
    
    # Note: Do not swap the order of the flattening to before the normalisation
    
    def LoadData(self,filepath,normalise=True):
                
        print("\n{:30}{}".format("Loaded data:",filepath.split("/")[-1]))
        
        # Determine the file extension (type) from the provided file path
        extension = filepath.split(".")[-1].lower()
        
        # If the extension matches ".npy" then load it, else throw an error
        if extension == "npy":  
            data = np.load(filepath)            
        else:
            print("File Type Not Supported: {}. ".format(extension))
            return None
        
        # Extract the positional data (i.e. volume) and scalars (i.e. values)
        volume = data[...,:-1]                 
        values = data[...,-1:]
        
        # Determine the input/output size and dimension meta-data 
        self.resolution = volume.shape[:-1]  # (i.e. [150,150,150])
        self.input_dimensions = volume.shape[ -1]  # (i.e. [3])
        self.size = values.size              # (i.e. 3375000)
        self.output_dimensions = values.shape[-1]  # (i.e. [1])
        
        # Determine the maximum, minimum, average and range values of 'volume'
        self.volume_max = np.amax(volume,axis=tuple(np.arange(self.input_dimensions)))
        self.volume_min = np.amin(volume,axis=tuple(np.arange(self.input_dimensions)))
        self.volume_avg = (self.volume_max+self.volume_min)/2.0
        self.volume_rng = abs(self.volume_max-self.volume_min)
        
        # Determine the maximum, minimum, average and range values of 'values'
        self.values_max = values.max()
        self.values_min = values.min()
        self.values_avg = (self.values_max+self.values_min)/2.0
        self.values_rng = abs(self.values_max-self.values_min)
        
        # Normalise 'volume' and 'values' to the range [-1,+1]
        if normalise:
            self.volume = 2.0*((volume-self.volume_avg)/(self.volume_rng))        
            self.values = 2.0*((values-self.values_avg)/(self.values_rng))
        else:
            self.volume = volume
            self.volume = values
            
        # Form an array of indices where 'coords[x,y,...,z]' = '[x,y,...,z]'
        self.coords = np.stack(np.meshgrid(*[np.arange(x) for x in self.resolution],indexing="ij"),axis=-1)
                
        # Flatten 'volume', 'values' and 'coords' before creating the dataset
        self.flat_volume = np.reshape(np.ravel(self.volume,order="C"),(-1,self.volume.shape[-1]))
        self.flat_values = np.reshape(np.ravel(self.values,order="C"),(-1,self.values.shape[-1]))   
        self.flat_coords = np.reshape(np.ravel(self.coords,order="C"),(-1,self.coords.shape[-1]))
                
        return None
    
    #==========================================================================
    # Define a function to copy attributes from 'DataClassObject' to 'self' but
    # without referencing 'DataClassObject' itself (so attributes in 'self' can 
    # be safely changed without changing those in 'DataClassObject')
    
    # -> 'getattr' and 'setattr' can be used to copy values without referencing
    
    def CopyData(self,DataClassObject):
        
        # Extract attribute keys from 'DataObject' and define exceptions
        exception_keys = ["values","flat_values"]
        attribute_keys = DataClassObject.__dict__.keys()
        
        # Iterate through the list of attribute keys
        for key in attribute_keys:
            
            # Copy the attribute if the key is not in 'exception_keys'
            if key not in exception_keys:
                setattr(self,key,getattr(DataClassObject,key))
            else:
                pass
            
        return None

    #==========================================================================
    # Define a function to compute the gradient of 'values' with respect to the
    # mesh (spacings) described by the elements of 'volume'
    
    def GetGradients(self):
        
        # Determine the gradient of 'values' with respect to 'volume'
        # values_reduced = self.values[...,0]
        # volume_reduced = [self.volume[...,x].shape for x in np.arange(self.input_dimensions)]
        # self.volume_gradient = np.gradient(self.values,self.volume)
        
        return None
        
    #==========================================================================
    # Define a function to create and return a 'tf.data.Dataset' dataset object     
    
    # -> Note: 'AUTOTUNE' prompts 'tf.data' to tune the value  of 'buffer_size'
    # -> dynamically at runtime.
    
    # -> Note: Moving 'dataset.cache()' up/down will reduce runtime performance
    # -> significantly.
    
    def MakeDataset(self,network_config):
        
        print("\n{:30}{}{}".format("Made TF dataset:","batch_size = ",network_config.batch_size))

        # Create a dataset whose elements are slices of the given tensors
        dataset = tf.data.Dataset.from_tensor_slices((self.flat_volume,self.flat_values,self.flat_coords))
        
        # Cache the elements of the dataset to increase runtime performance
        dataset = dataset.cache()
        
        # Randomly shuffle the elements of the cached dataset 
        dataset = dataset.shuffle(buffer_size=self.size,reshuffle_each_iteration=True)
                    
        # Concatenate elements of the dataset into mini-batches
        dataset = dataset.batch(batch_size=network_config.batch_size,drop_remainder=False)
        
        # Pre-fetch elements from the dataset to increase throughput
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
                
        return dataset 
    
    #==========================================================================
        
    def PredictValues(self,network,network_config):
        
        # Predict 'flat_values' using the Neurcomp's learned weights and biases
        self.flat_values = network.predict(self.flat_volume,batch_size=network_config.batch_size,verbose="0")
        
        # Reshape 'flat_values' into the shape of the original input dimensions
        self.values = np.reshape(self.flat_values,(self.resolution+(self.output_dimensions,)),order="C")

        return None        
    
    #==========================================================================
    # Define a function to concatenate and save a scalar field to a '.npy' file
    
    # -> Note: 'self.volume' and 'self.values' should be appropriately reshaped
    # -> such that they have the same shape as the input scalar field.
    
    def SaveData(self,output_volume_path,reverse_normalise=True):
                    
        # Reverse normalise 'volume' and 'values' to the initial range
        if reverse_normalise:
            self.volume = ((self.volume_rng*(self.volume/2.0))+self.volume_avg)
            self.values = ((self.values_rng*(self.values/2.0))+self.values_avg)
        else:
            self.volume = self.volume
            self.values = self.values 
        
        # Save as Numpy file 
        np.save(output_volume_path,np.concatenate((self.volume,self.values),axis=-1))
        
        # Save as VTK file
        volume_list = [np.ascontiguousarray(self.volume[...,x]) for x in range(self.input_dimensions)]
        values_dict = {"values":np.ascontiguousarray(self.values[...,0])}
        gridToVTK(output_volume_path,*volume_list,pointData=values_dict)

        return None
        
#=============================================================================#
# Potential class for storing training data

class TrainingDataClass():
    
    def __init__(self):
        
        pass
    
    