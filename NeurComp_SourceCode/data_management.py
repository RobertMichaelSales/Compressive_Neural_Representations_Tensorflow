""" Created: 18.07.2022  \\  Updated: 02.11.2022  \\   Author: Robert Sales """

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

    def __init__(self,name):
        
        # Initialise the name of the 'DataClass' instance
        self.name = name
        print("Creating DataClass Object: {}".format(self.name))
        
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
        self.input_resolution = np.array      
        self.input_dimensions = 3     
        self.input_size = 0
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
        
        print("Loading Data: {} > {}\n".format(filepath.split("/")[-1],self.name))
        
        # Determine the file extension (type) from the provided file path
        extension = filepath.split(".")[-1].lower()
        
        # If the extension matches ".npy" then load it, else throw an error
        if extension == "npy":  
            data = np.load(filepath)            
        else:
            print("Error: File Type Not Supported: '{}'. ".format(extension))
            return None
        
        # Extract the positional data (i.e. volume) and scalars (i.e. values)
        self.volume = data[...,:-1]                 
        self.values = data[...,-1:]
        
        # Determine the input/output meta-data 
        self.input_resolution = self.volume.shape[:-1]  # (i.e. [150,150,150])
        self.input_dimensions = self.volume.shape[ -1]  # (i.e. [3])
        self.input_size = self.values.size              # (i.e. 3375000)
        self.output_dimensions = self.values.shape[-1]  # (i.e. [1])
        
        # Form an array of indices where 'coords[x,y,...,z]' = '[x,y,...,z]'
        self.coords = np.stack(np.meshgrid(*[np.arange(x) for x in self.input_resolution],indexing="ij"),axis=-1)
        
        # Determine the maximum, minimum, average and range values of 'volume'
        self.volume_max = np.amax(self.volume,axis=tuple(np.arange(self.input_dimensions)))
        self.volume_min = np.amin(self.volume,axis=tuple(np.arange(self.input_dimensions)))
        self.volume_avg = (self.volume_max+self.volume_min)/2.0
        self.volume_rng = abs(self.volume_max-self.volume_min)
        
        # Determine the maximum, minimum, average and range values of 'values'
        self.values_max = self.values.max()
        self.values_min = self.values.min()
        self.values_avg = (self.values_max+self.values_min)/2.0
        self.values_rng = abs(self.values_max-self.values_min)
        
        # Normalise 'volume' and 'values' to the range [-1,+1]
        if normalise:
            self.volume = 2.0*((self.volume-self.volume_avg)/(self.volume_rng))        
            self.values = 2.0*((self.values-self.values_avg)/(self.values_rng))
        else:
            pass
                
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
        
        print("Copying Data: {} > {}\n".format(DataClassObject.name,self.name))
        
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
    
    def MakeDataset(self,network_config):
        
        print("Creating Tensorflow Training Dataset\n")
        
        # Create a dataset whose elements are slices of the given tensors
        dataset = tf.data.Dataset.from_tensor_slices((self.flat_volume,self.flat_values,self.flat_coords))
        
        # Cache the elements of the dataset to increase runtime performance
        dataset = dataset.cache()
        
        # Randomly shuffle the elements of the cached dataset 
        dataset = dataset.shuffle(buffer_size=self.input_size,reshuffle_each_iteration=True)
        
        # Concatenate elements of the dataset into mini-batches
        dataset = dataset.batch(batch_size=network_config.batch_size,drop_remainder=False)
        
        # Pre-fetch elements from the dataset to increase throughput
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
                
        return dataset 
    
    #==========================================================================
    # Define a function to concatenate and save a scalar field to a '.npy' file
    
    # -> Note: 'self.volume' and 'self.values' should be appropriately reshaped
    # -> such that they have the same shape as the input scalar field.
    
    def SaveData(self,filepath,normalise=True):
        
        print("Saving Data: {}.\n".format(filepath.split("/")[-1]))
        
        # Determine the file extension (type) from the provided file path
        extension = filepath.split(".")[-1].lower()
            
        # Reverse normalise 'volume' and 'values' to the initial range
        if normalise:
            self.volume = ((self.volume_rng*(self.volume/2.0))+self.volume_avg)
            self.values = ((self.values_rng*(self.values/2.0))+self.values_avg)
        else:
            pass
        
        # If the extension matches ".npy" then save it else throw an error
        if extension == "npy":  
            
            data = np.concatenate((self.volume,self.values),axis=-1)
            np.save(filepath,data)
        else:
            print("Error: File Type Not Supported: '{}'. ".format(extension))
            return None

        return None
        
#=============================================================================#