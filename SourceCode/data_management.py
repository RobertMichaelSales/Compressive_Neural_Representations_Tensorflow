""" Created: 18.07.2022  \\  Updated: 29.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
import tensorflow as tf
from pyevtk.hl import gridToVTK,pointsToVTK

#==============================================================================
# Define a class for managing datasets: i.e. loading, handling and storing data  

class DataClass():

    #==========================================================================
    # The constructor function for 'DataClass'   

    def __init__(self,data_type,tabular):
        
        # Set the object data type and input data structure
        self.data_type = data_type
        self.tabular = tabular
    
        return None
    
    ##
    
    #==========================================================================
    # Loads and returns pre-processed scalar fields for use creating Tensorflow DataSet objects
    # Note: The input filepath must refer to a file ending in '.npy' 
    # Note: Make sure the flattening takes place after normalisation
    
    def LoadData(self,input_data_path,columns,shape,dtype,normalise=True):
                
        # Unpack the columns for the coordinate axes and scalar fields
        volume_columns,values_columns,weight_column = columns
                
        # Load the entire dataset then extract the desired tensor
        if (self.data_type=="volume"): 
            
            self.dimensions = len(volume_columns)
            self.data = np.load(input_data_path)[...,volume_columns].astype('float32')
            
        ##
            
        # Load the entire dataset then extract the desired tensor
        if (self.data_type=="values"): 
            
            self.dimensions = len(values_columns)
            self.data = np.load(input_data_path)[...,values_columns].astype('float32')
            
        ##           
        
        # Load the entire dataset then extract the desired tensor
        if (self.data_type=="weights"):
            
            self.dimensions = len(weight_column)

            if not self.dimensions: 
                
                self.data = np.array(None)
                self.flat = np.array(None)
                return None
            
            else: 
        
                self.data = np.load(input_data_path)[...,weight_column].astype('float32')
                
            ##     
        ##        
    
        # Determine the field resolution and number of values
        self.resolution = self.data.shape[:-1] 
        self.size = self.data.size
        
        # Determine the maximum, minimum, average and range
        self.max = np.array([self.data[...,i].max() for i in range(self.dimensions)])
        self.min = np.array([self.data[...,i].min() for i in range(self.dimensions)])
        self.avg = (self.max+self.min)/2.0
        self.rng = abs(self.max-self.min)
        
        # Adjust zero-entries
        self.rng[self.max==self.min] = 1.0        
        
        # Normalise each tensor field to the range [-1,+1] or [0,+1]
        if normalise:
            if self.data_type in ["volume","values"]:
                self.data = 2.0*((self.data-self.avg)/(self.rng))  
            else:pass
            if self.data_type in ["weights"]:
                self.data = 0.5+((self.data-self.avg)/(self.rng)) + 1e-9
            else:pass
        else: pass
    
        # Flatten each tensor fields into equal lists of vectors
        self.flat = np.reshape(np.ravel(self.data,order="C"),(-1,self.dimensions))
                
        return None 
    
    ##      
        
    #==========================================================================
    # Indepenently copies / deep copies attributes from 'DataClassObject' without referencing
    # Note: 'getattr()' and 'setattr()' are used to copy without referencing 
    
    def CopyData(self,DataClassObject,exception_keys):
        
        # Extract attribute keys from 'DataObject'
        attribute_keys = DataClassObject.__dict__.keys()
        
        # Iterate through the list of attribute keys
        for key in attribute_keys:
            
            # Copy the attribute if key not in 'exception_keys'
            if key not in exception_keys:
                setattr(self,key,getattr(DataClassObject,key))
            else: pass
            
        return None
    
    ##

#==============================================================================
# Creates a 'tf.data.Dataset' object from input volume, input values and optional weight data
# Note: 'AUTOTUNE' prompts 'tf.data' to tune the value  of 'buffer_size' dynamically at runtime
# Note: Moving 'dataset.cache()' up or down will damage the runtime performance significantly

def MakeDatasetFromTensorSlc(volume,values,weights,batch_size,cache_dataset):
        
    # Handle the case where there are no weights -> set them all to 1
    if not weights.flat.any(): weights.flat = np.ones(shape=((np.prod(volume.resolution),)+(1,))).astype("float32")
    
    # Extend the weights to apply to each element of the output vector
    weights.flat = np.repeat(weights.flat,values.dimensions,axis=-1)     
    
    # Convert all numpy arrays to tensorflow tensors, then pre-shuffle
    shuffle_order = tf.random.shuffle(tf.range(start=0,limit=values.size,delta=1,dtype=tf.int32))
    volume_flat = tf.gather(tf.convert_to_tensor(volume.flat),  shuffle_order)
    values_flat = tf.gather(tf.convert_to_tensor(values.flat),  shuffle_order)
    weights_flat = tf.gather(tf.convert_to_tensor(weights.flat),shuffle_order)
    
    # Create a dataset whose elements are slices of the given tensors
    dataset = tf.data.Dataset.from_tensor_slices((volume_flat,values_flat,weights_flat))
    
    # Cache the elements of the dataset to increase runtime performance
    if cache_dataset: dataset = dataset.cache()

    # Set the shuffle buffer size to equal the number of scalars
    # buffer_size = np.prod(values.resolution)
    
    # Randomly shuffle the elements of the dataset 
    # dataset = dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=False)
                
    # Concatenate elements of the dataset into mini-batches
    dataset = dataset.batch(batch_size=batch_size,drop_remainder=False,num_parallel_calls=tf.data.AUTOTUNE)
    
    # Pre-fetch elements from the dataset to increase throughput
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Assign a size attribute to store the batches per data pass
    dataset.size = len(dataset)
            
    return dataset    

#==============================================================================

# Saves output data to both '.npy' and '.vtk' files

def SaveData(output_data_path,volume,values,reverse_normalise=True):
                    
    # Reverse normalise 'volume' and 'values' to the initial ranges
    if reverse_normalise:
        volume.data = ((volume.rng*(volume.data/2.0))+volume.avg)
        values.data = ((values.rng*(values.data/2.0))+values.avg)
    else: pass

    # Save as Numpy file 
    np.save(output_data_path,np.concatenate((volume.data,values.data),axis=-1))
    
    # Create volume list and values dict for VTK/VTS
    volume_list, values_dict = [],{}
    
    # Add volume fields to list
    for dimension in range(volume.dimensions):
        volume_list.append(np.ascontiguousarray(volume.data[...,dimension]))
    ##
    
    # Add values fields to dict
    for dimension in range(values.dimensions):
        key = "field" + str(dimension)
        values_dict[key] = np.ascontiguousarray(values.data[...,dimension])
    ##
    
    # Save to '.vtk'/'.vts' using VTK library
    if (volume.tabular == values.tabular == False):                                        
        gridToVTK(output_data_path,*volume_list,pointData=values_dict)  
    else:
        pointsToVTK(output_data_path,*volume_list,data=values_dict)
    ##
        
    return None

##

#=============================================================================#