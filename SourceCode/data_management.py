""" Created: 18.07.2022  \\  Updated: 29.07.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import trimesh, os
import numpy as np
import pyvista as pv
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
   
    # Loads and returns data for use in creating Tensorflow DataSet objects
    
    def LoadData(self,input_data_path,columns,normalise):
                
        # Unpack the columns for the coordinate axes and scalar fields
        coords_columns,values_columns,scales_column = columns
                
        # Load the coords data and normalise
        if (self.data_type=="coords"): 
            
            # Load data
            self.dimensions = len(coords_columns)
            self.data = np.load(input_data_path)[...,coords_columns].astype(np.float32)
            self.original_centre, self.original_radius = self.GetVolumeBoundingSphere(self.data)
            
            # Normalise
            self.data = self.data - self.original_centre
            self.data = self.data / self.original_radius            
        ##
            
        # Load the values data and normalise
        if (self.data_type=="values"): 
            
            # Load data
            self.dimensions = len(values_columns)
            self.data = np.load(input_data_path)[...,values_columns].astype(np.float32)           
            self.original_average, self.original_range, self.original_bounds = self.GetValuesBoundingValues(self.data)
 
            # Normalise
            self.data = self.data - self.original_average
            self.data = self.data / (self.original_range / 2.0)            
        ##           
        
        # Load the entire dataset then extract the desired tensor
        if (self.data_type=="scales"):
            
            # Load data
            self.dimensions = len(scales_column)
            self.data = np.load(input_data_path)[...,scales_column].astype(np.float32)

            # Clip problematic values and normalise
            self.data = np.maximum(self.data, np.abs(self.data).min())
            self.data = np.maximum(self.data, np.finfo(np.float64).eps)
            self.data = self.data / np.average(self.data)
            
        ##        
    
        # Determine the field resolution and number of values
        self.resolution = self.data.shape[:-1] 
        self.size = self.data.size
    
        # Flatten each tensor fields into equal lists of vectors
        self.flat = np.reshape(np.ravel(self.data,order="F"),(-1,self.dimensions),order="F")
                
        return None 
    ##      
        
    #==========================================================================
    
    # Returns the bounding sphere centre and radius for a given set of points
    
    def GetVolumeBoundingSphere(self,points):
        
        # Create a trimesh PointCloud object for a set of points
        point_cloud = trimesh.PointCloud(points.reshape(-1,3))
      
        # Extract sphere centre and radius
        original_centre, original_radius = trimesh.nsphere.minimum_nsphere(point_cloud)
                          
        return original_centre, original_radius
    ##
    
    #==========================================================================
    
    # Returns the bounding values average and range for a given set of points
    
    def GetValuesBoundingValues(self,points):
        
        # Determine the original bounds, average and range
        original_bounds = np.array([points.min(),points.max()])
        original_average = np.mean(original_bounds)
        original_range = np.ptp(original_bounds)
        
        return original_average, original_range, original_bounds
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

# Creates a 'tf.data.Dataset' from input coords, input values and weight data

def MakeDataset(coords,values,scales,batch_size,cache_dataset):
        
    # Convert all numpy arrays to tensorflow shuffled tensorflow tensors on CPU
    with tf.device('/CPU:0'):
        shuffle_order = tf.random.shuffle(tf.range(coords.flat.shape[0],dtype=tf.int32))
        coords_tensor = tf.gather(tf.convert_to_tensor(coords.flat,dtype=tf.float32),shuffle_order)
        values_tensor = tf.gather(tf.convert_to_tensor(values.flat,dtype=tf.float32),shuffle_order)
        scales_tensor = tf.gather(tf.convert_to_tensor(scales.flat,dtype=tf.float32),shuffle_order)    
    ##
    
    # Create a dataset whose elements are slices of the given tensors
    dataset = tf.data.Dataset.from_tensor_slices((coords_tensor,values_tensor,scales_tensor))
                
    # Concatenate elements of the dataset into mini-batches
    dataset = dataset.batch(batch_size=batch_size,drop_remainder=False,num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache if requested (optional, may increase memory usage)
    if cache_dataset: dataset = dataset.cache()
    
    # Pre-fetch elements from the dataset to increase throughput
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Assign a size attribute to store the batches per data pass
    dataset.size = int(np.ceil(np.prod(coords.data.shape[:-1]) / batch_size))
            
    return dataset    
##

#==============================================================================

def SaveData(output_data_path,template_vtk_path,coords,values,normalise):
    
    # Reverse normalise coords and values (data and flat)
    if normalise:
        
        coords.data = coords.data * coords.original_radius
        coords.data = coords.data + coords.original_centre
        values.data = values.data * (values.original_range / 2.0)
        values.data = values.data + values.original_average
        
        coords.flat = coords.flat * coords.original_radius
        coords.flat = coords.flat + coords.original_centre
        values.flat = values.flat * (values.original_range / 2.0)
        values.flat = values.flat + values.original_average
        
    ##
    
    # Save coords and values concatenated as npy file 
    np.save(output_data_path,np.concatenate((coords.data,values.data),axis=-1).astype(np.float32))

    # Read in template and clear existing data points
    vtk_file = pv.read(template_vtk_path)
    vtk_file.point_data.clear()

    # Insert new flattened arrays
    for dimension in range(values.dimensions): 
        key = "var_{}".format(dimension)
        vtk_file.point_data[key] = values.data[..., dimension].ravel(order="F")
    ##
    
    # Choose correct extension and save
    vtk_file.save(output_data_path + os.path.splitext(template_vtk_path)[-1])
    
    return None
##

#==============================================================================