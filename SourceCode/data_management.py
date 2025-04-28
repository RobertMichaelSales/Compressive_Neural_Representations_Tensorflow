""" Created: 18.07.2022  \\  Updated: 29.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import trimesh
import numpy as np
import tensorflow as tf
from pyevtk.hl import gridToVTK,pointsToVTK

#==============================================================================

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
                
        # Load the volume data and normalise
        if (self.data_type=="volume"): 
            
            self.dimensions = len(volume_columns)
            self.data = np.load(input_data_path)[...,volume_columns].astype('float32')
            self.original_centre, self.original_radius, self.original_bounds = self.GetVolumeBoundingSphere(self.data)
            
            if (normalise):
                self.data = self.data - self.original_centre
                self.data = self.data / self.original_radius
            ##
            
        ##
            
        # Load the values data and normalise
        if (self.data_type=="values"): 
            
            self.dimensions = len(values_columns)
            self.data = np.load(input_data_path)[...,values_columns].astype('float32')
            self.original_average, self.original_range, self.original_bounds = self.GetValuesBoundingValues(self.data)
            
            if (normalise):
                self.data = self.data - self.original_average
                self.data = self.data / (self.original_range / 2.0)
            ##
            
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
    
        # Flatten each tensor fields into equal lists of vectors
        self.flat = np.reshape(np.ravel(self.data,order="C"),(-1,self.dimensions))
                
        return None 
    
    ##      
        
    #==========================================================================
    
    # Returns the bounding sphere centre and radius for a given set of points
    # Note: Using the Trimesh package means these are the equal for SDFs/ISOs
    # Note: 
    
    def GetVolumeBoundingSphere(self,points):
           
        # Create a trimesh PointCloud object for a set of points
        point_cloud = trimesh.PointCloud(points.reshape(-1,3))
                
        # Extract sphere centre and radius
        original_centre = np.array(point_cloud.bounding_sphere.center)
        original_radius = np.cbrt((3.0*point_cloud.bounding_sphere.volume)/(4.0*np.pi))
        original_bounds = np.array(point_cloud.bounding_sphere.bounds).T

        return original_centre, original_radius, original_bounds
        
    #==========================================================================
    
    # Returns the bounding sphere centre and radius for a given set of points
    # Note: Using the Trimesh package means these are the equal for SDFs/ISOs
    
    def GetVolumeBoundingCuboid(self,points):
           
        # Create a trimesh PointCloud object for a set of points
        point_cloud = trimesh.PointCloud(points.reshape(-1,3))
        
        # Extract cuboid centre and bounds
        original_centre = point_cloud.bounding_box.centroid
        original_extent = point_cloud.bounding_box.extent
        original_bounds = point_cloud.bounding_box.bounds.T
           
        return original_centre, original_extent, original_bounds
        
    #==========================================================================
    
    # Returns the bounding values average and range for a given set of points
    # Note: bounds are transposed so bounds[0] returns the bounds for field 0
    # Note: the transpose changes "axis=0" to "axis=-1"
    
    def GetValuesBoundingValues(self,points):
        
        # Determine the maximum and minimum values per axis
        original_minimum = points.min(axis=tuple(range(points.ndim - 1)))
        original_maximum = points.max(axis=tuple(range(points.ndim - 1)))
        original_bounds = np.array([original_minimum,original_maximum]).T
        
        # Determine the average and range values per axis
        original_average = np.mean(original_bounds,axis=-1)
        original_range = np.ptp(original_bounds,axis=-1)
        
        return original_average, original_range, original_bounds
    
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
        volume.data = volume.data * volume.original_radius
        volume.data = volume.data + volume.original_centre
        values.data = values.data * (values.original_range / 2.0)
        values.data = values.data + values.original_average
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