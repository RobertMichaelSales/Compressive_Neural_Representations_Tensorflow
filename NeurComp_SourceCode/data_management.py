""" Created: 18.07.2022  \\  Updated: 19.01.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
import tensorflow as tf
from pyevtk.hl import gridToVTK

#==============================================================================
# Define a class for managing datasets: i.e. loading, handling and storing data  

class DataClass():

    #==========================================================================
    # Define the initialisation constructor function for 'DataClass'   

    def __init__(self,data_type):
        
        # Set the object data type
        self.data_type = data_type
        
        return None
    
    #==========================================================================
    # Define a function to load an arbitrary dimension scalar field and extract 
    # useful meta-data. 
    
    # -> Note: The input filepath must refer to a file ending in '.npy' 
    
    # -> Note: Make sure the flattening takes place after normalisation
    
    def LoadData(self,input_data_path,indices,normalise=True):
        
        # Unpack the indices for the coordinate axes and scalar fields
        coord_indices,field_indices = indices
          
        # Load the entire data block then extract the volume tensor
        if (self.data_type=="volume"): 
            print("\n{:30}{}".format("Loaded volume:",input_data_path.split("/")[-1]))
            self.dimensions = len(coord_indices)
            # self.data = np.load(input_data_path)[...,coord_indices]   
            
            self.data = np.lib.format.open_memmap(filename=input_data_path,mode="r",dtype='float32',shape=(100520000,10))[0:10052,coord_indices]      # <<<<<<<<<
            
            
        else: pass
            
        # Load the entire data block then extract the values tensor
        if (self.data_type=="values"): 
            print("\n{:30}{}".format("Loaded values:",input_data_path.split("/")[-1]))
            self.dimensions = len(field_indices)
            # self.data = np.load(input_data_path)[...,field_indices]  
            
            self.data = np.lib.format.open_memmap(filename=input_data_path,mode="r",dtype='float32',shape=(100520000,10))[0:10052,field_indices]      # <<<<<<<<<
            
            
        else: pass
    
        # Determine the field resolution
        self.resolution = self.data.shape[:-1]
    
        # Determine the number of values
        self.size = self.data.size
        
        # Determine the maximum 
        self.max = np.array([self.data[...,i].max() for i in range(self.dimensions)])
        
        # Determine the minimum
        self.min = np.array([self.data[...,i].min() for i in range(self.dimensions)])
        
        # Determine the average
        self.avg = (self.max+self.min)/2.0
        
        # Determine the range
        self.rng = abs(self.max-self.min)
 
        # Normalise each tensor field to the range [-1,+1]
        if normalise:
            self.data = 2.0*((self.data-self.avg)/(self.rng))        
        else: pass
    
        # Flatten each tensor fields into equal lists of vectors
        self.flat = np.reshape(np.ravel(self.data,order="C"),(-1,self.dimensions))
                
        return None       
        
    #==========================================================================
    # Define a function to copy attributes from 'DataClassObject' to 'self' but
    # without referencing 'DataClassObject' itself (so attributes in 'self' can 
    # be safely changed without changing those in 'DataClassObject')
    
    # -> Note: 'getattr()' and 'setattr()' are used to copy without referencing 
    
    def CopyData(self,DataClassObject,exception_keys):
        
        # Extract attribute keys from 'DataObject' and define exceptions
        attribute_keys = DataClassObject.__dict__.keys()
        
        # Iterate through the list of attribute keys
        for key in attribute_keys:
            
            # Copy the attribute if the key is not in 'exception_keys'
            if key not in exception_keys:
                setattr(self,key,getattr(DataClassObject,key))
            else: pass
            
        return None
        
#==============================================================================
# Define a function to create and return a 'tf.data.Dataset' dataset object

# -> Note: 'AUTOTUNE' prompts 'tf.data' to tune the value  of 'buffer_size'
# -> dynamically at runtime

# -> Note: Moving 'dataset.cache()' up/down will reduce runtime performance
# -> significantly


def MakeDataset(volume,values,batch_size,repeat=False):
    
    print("\n{:30}{}{}".format("Made TF dataset:","batch_size = ",batch_size))
    
    # Create a dataset whose elements are slices of the given tensors
    dataset = tf.data.Dataset.from_tensor_slices((volume.flat,values.flat))
    
    # Cache the elements of the dataset to increase runtime performance
    dataset = dataset.cache()
    
    # Makes the dataset infinitely iterable (i.e. infinitely repeating)
    if repeat: 
        dataset = dataset.repeat(count=None)
    else: pass

    # Set the shuffle buffer size to equal the number of scalars
    buffer_size = values.size
    
    # Randomly shuffle the elements of the cached dataset 
    dataset = dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=True)
                
    # Concatenate elements of the dataset into mini-batches
    dataset = dataset.batch(batch_size=batch_size,drop_remainder=False)
    
    # Pre-fetch elements from the dataset to increase throughput
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            
    return dataset    

#==============================================================================
# Define a generator function to supply the dataset with a datastream of inputs

def CreateDataGen(data,indices):
    
    # Unpack the indices for the coordinate axes and scalar fields
    coord_indices,field_indices = indices
    
    def DataGen():
        
        for row in data:
            
            yield row[coord_indices],row[field_indices]
            
    return DataGen

#==============================================================================

def MakeDatasetFromGenerator(data,indices,batch_size,repeat=False):
        
    generator = CreateDataGen(data,indices)
    
    output_types = (tf.float32,tf.float32)
    
    output_shapes = (tf.TensorShape((len(indices[0]),)),tf.TensorShape(len(indices[1]),))
        
    dataset = tf.data.Dataset.from_generator(generator=generator,output_types=output_types,output_shapes=output_shapes)
    
    dataset = dataset.cache()
    
    if repeat: 
        dataset = dataset.repeat(count=None)
    else: pass

    buffer_size = min(values.size,int(1e6))
    
    dataset = dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=True)
                
    dataset = dataset.batch(batch_size=batch_size,drop_remainder=False)
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            
    return dataset    

#==============================================================================
# Define a function to concatenate and save a scalar field to a '.npy' file

# -> Note: 'volume' and 'values' must be reshaped to match the input shapes

def SaveData(output_data_path,volume,values,reverse_normalise=True):
                
    # Reverse normalise 'volume' and 'values' to the initial ranges
    if reverse_normalise:
        volume.data = ((volume.rng*(volume.data/2.0))+volume.avg)
        values.data = ((values.rng*(values.data/2.0))+values.avg)
    else: pass

    # Save as Numpy file 
    np.save(output_data_path,np.concatenate((volume.data,values.data),axis=-1))
    
    # Save as VTK file
    volume_list, values_dict = [],{}
    
    for dimension in range(volume.dimensions):
        volume_list.append(np.ascontiguousarray(volume.data[...,dimension]))
        
    for dimension in range(values.dimensions):
        key = "field" + str(dimension)
        values_dict[key] = np.ascontiguousarray(values.data[...,dimension])
                                                
    gridToVTK(output_data_path,*volume_list,pointData=values_dict)

    return None

#=============================================================================#
import numpy as np
import tensorflow as tf

# filename = "/Data/Compression_Datasets/combustor_les_compressible_time/combustor_les_compressible_time_all_data.npy"

# volume = DataClass(data_type="volume")
# values = DataClass(data_type="values")

# volume.LoadData(input_data_path=filename, indices=([0,2,3],[4]), normalise=False)
# values.LoadData(input_data_path=filename, indices=([0,2,3],[4]), normalise=False)

#=============================================================================#
import numpy as np
import tensorflow as tf

filename = "/Data/Compression_Datasets/combustor_les_compressible_time/combustor_les_compressible_time_all_data.npy"

data = np.lib.format.open_memmap(filename=filename,mode="r",dtype='float32',shape=(100520000,10))

# dataset = MakeDataset(volume=volume, values=values, batch_size=1)

# next(iter(dataset))
#=============================================================================#