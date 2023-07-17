""" Created: 18.07.2022  \\  Updated: 24.03.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
import tensorflow as tf
from pyevtk.hl import gridToVTK,pointsToVTK

#==============================================================================
# Define a class for managing datasets: i.e. loading, handling and storing data  

class DataClass():

    #==========================================================================
    # Define the initialisation constructor function for 'DataClass'   

    def __init__(self,data_type,tabular,exceeds_memory):
        
        # Set the object data type
        self.data_type = data_type
        
        # Set input data structure
        self.tabular = tabular
        
        # Set input exceeds memory
        self.exceeds_memory = exceeds_memory
    
        return None
    
    #==========================================================================
    # Define a function to load an arbitrary dimension scalar field and extract 
    # useful meta-data. 
    
    # -> Note: The input filepath must refer to a file ending in '.npy' 
    
    # -> Note: Make sure the flattening takes place after normalisation
    
    def LoadData(self,input_data_path,columns,shape,dtype,normalise=True):
                
        # Unpack the columns for the coordinate axes and scalar fields
        coord_columns,field_columns,weight_column = columns
                
        # Load the full dataset then extract the desired volume tensor
        if (self.data_type=="volume"):     
            
            print("\n{:30}{}".format("Loaded volume:",input_data_path.split("/")[-1]))
            print("{:30}{}".format("Fields:",coord_columns))
            self.dimensions = len(coord_columns)
            
            if self.tabular:
                
                # if self.exceeds_memory:
                #     self.data = np.lib.format.open_memmap(filename=input_data_path,mode="r",dtype=dtype,shape=shape)[...,coord_columns].astype('float32')
                # else:
                #     self.data = np.load(input_data_path)[...,coord_columns].astype('float32')
                # ##
                
                self.data = np.load(input_data_path)[...,coord_columns].astype('float32')
                
            else: 
                
                # if self.exceeds_memory:
                #     self.data = np.lib.format.open_memmap(filename=input_data_path,mode="r",dtype=dtype,shape=shape).astype('float32')
                # else:
                #     self.data = np.load(input_data_path)[...,coord_columns].astype('float32')
                # ##
                
                self.data = np.load(input_data_path)[...,coord_columns].astype('float32')
            ##
        ##
            
        # Load the entire data block then extract the values tensor
        if (self.data_type=="values"): 
            
            print("\n{:30}{}".format("Loaded values:",input_data_path.split("/")[-1]))
            print("{:30}{}".format("Fields:",field_columns))
            self.dimensions = len(field_columns)
            
            if self.tabular:
                
                # if self.exceeds_memory:
                #     self.data = np.lib.format.open_memmap(filename=input_data_path,mode="r",dtype=dtype,shape=shape)[...,field_columns].astype('float32')
                # else:
                #     self.data = np.load(input_data_path)[...,field_columns].astype('float32')
                # ##
                
                self.data = np.load(input_data_path)[...,field_columns].astype('float32')
                
            else: 
                
                # if self.exceeds_memory:
                #     self.data = np.lib.format.open_memmap(filename=input_data_path,mode="r",dtype=dtype,shape=shape).astype('float32')
                # else:
                #     self.data = np.load(input_data_path)[...,field_columns].astype('float32')
                # ##
                
                self.data = np.load(input_data_path)[...,field_columns].astype('float32')
            ##     
        ##           
        
        # Load the full dataset then extract the desired weight tensor
        if (self.data_type=="weights"):
            
            print("\n{:30}{}".format("Loaded weights:",input_data_path.split("/")[-1]))
            print("{:30}{}".format("Fields:",weight_column))
            self.dimensions = len(weight_column)

            if not self.dimensions: 
                
                self.data = np.array(None)
                self.flat = np.array(None)
                return None
            
            else: 
            
                if self.tabular:
                    
                    # if self.exceeds_memory:
                    #     self.data = np.lib.format.open_memmap(filename=input_data_path,mode="r",dtype=dtype,shape=shape)[...,field_columns].astype('float32')
                    # else:
                    #     self.data = np.load(input_data_path)[...,field_columns].astype('float32')
                    # ##
                    
                    self.data = np.load(input_data_path)[...,weight_column].astype('float32')
                    
                else: 
                    
                    # if self.exceeds_memory:
                    #     self.data = np.lib.format.open_memmap(filename=input_data_path,mode="r",dtype=dtype,shape=shape).astype('float32')
                    # else:
                    #     self.data = np.load(input_data_path)[...,field_columns].astype('float32')
                    # ##
                    
                    self.data = np.load(input_data_path)[...,weight_column].astype('float32')
                ##
            ##     
        ##        
    
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

#============================================================================== <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Define a function to create and return a 'tf.data.Dataset' dataset object

# -> Note: 'AUTOTUNE' prompts 'tf.data' to tune the value  of 'buffer_size'
# -> dynamically at runtime

# -> Note: Moving 'dataset.cache()' up/down will reduce runtime performance
# -> significantly

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
    
    # Randomly shuffle the elements of the cached dataset 
    # dataset = dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=False)
                
    # Concatenate elements of the dataset into mini-batches
    dataset = dataset.batch(batch_size=batch_size,drop_remainder=False,num_parallel_calls=tf.data.AUTOTUNE)
    
    # Pre-fetch elements from the dataset to increase throughput
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Assign a size attribute to store the batches per data pass
    dataset.size = len(dataset)
            
    return dataset    

#==============================================================================
# Define a function to create and return a 'tf.data.Dataset' dataset object

# -> Note: 'AUTOTUNE' prompts 'tf.data' to tune the value  of 'buffer_size'
# -> dynamically at runtime

# -> Note: Moving 'dataset.cache()' up/down will reduce runtime performance
# -> significantly

def MakeDatasetFromGenerator(volume,values,weights,batch_size,cache_dataset):
    
    # Handle the case where there are no weights -> set them all to 1
    if not weights.flat.any(): weights.flat = np.ones(shape=((np.prod(volume.resolution),)+(1,))).astype("float32")
    
    # Extend the weights to apply to each element of the output vector
    weights.flat = np.repeat(weights.flat,values.dimensions,axis=-1)

    # Get the data generator as an iterable and pass it its arguments
    generator = GetDataGenerator(tf.convert_to_tensor(volume.flat),tf.convert_to_tensor(values.flat),tf.convert_to_tensor(weights.flat))
    
    # Set the expected output types
    output_types = (tf.float32,tf.float32,tf.float32)
    
    # Set the expected output shapes
    output_shapes = (tf.TensorShape(volume.flat.shape[-1]),tf.TensorShape(values.flat.shape[-1]),tf.TensorShape(weights.flat.shape[-1]))
    
    # Create a dataset whose elements are retrieved via a data generator
    dataset = tf.data.Dataset.from_generator(generator=generator,output_types=output_types,output_shapes=output_shapes)
    
    # Cache the elements of the dataset to increase runtime performance
    if cache_dataset: dataset = dataset.cache()

    # Set the shuffle buffer size to equal the number of scalars
    # buffer_size = np.prod(values.resolution)
    
    # Randomly shuffle the elements of the cached dataset 
    # dataset = dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=True)
                
    # Concatenate elements of the dataset into mini-batches
    dataset = dataset.batch(batch_size=batch_size,drop_remainder=False,num_parallel_calls=tf.data.AUTOTUNE)
    
    # Pre-fetch elements from the dataset to increase throughput
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Assign a size attribute to store the batches per data pass
    dataset.size = int(np.ceil(volume.flat.shape[0]/batch_size))
                
    return dataset    

#============================================================================== <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Define a generator function to supply the dataset with a stream of data pairs

def GetDataGenerator(volume,values,weights):
    
    def DataGenerator():
        
        for vol,val,wht in zip(volume,values,weights):
            
            yield vol,val,wht
            
    return DataGenerator

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
    
    # Create dictionaries for saving a VTK
    volume_list, values_dict = [],{}
    
    # Check volume and values are both 3-D
    if (volume.dimensions == 3):
        pass
    else:
        return None
    ##
        
    for dimension in range(volume.dimensions):
        volume_list.append(np.ascontiguousarray(volume.data[...,dimension]))
    ##
    
    for dimension in range(values.dimensions):
        key = "field" + str(dimension)
        values_dict[key] = np.ascontiguousarray(values.data[...,dimension])
    ##
    
    if (volume.tabular == values.tabular == False):                                        
        gridToVTK(output_data_path,*volume_list,pointData=values_dict)  
    else:
        pointsToVTK(output_data_path,*volume_list,data=values_dict)
    ##
        
    return None

#==============================================================================

def Benchmark(dataset, num_epochs=2):
    
    import time
    
    # Start timing the entire training loop
    time_total_tick = time.perf_counter()
    
    for epoch_num in range(num_epochs):
        
        # Start timing the epoch training loop
        time_epoch_tick = time.perf_counter()
        
        # Performing an artificial training step
        for sample in dataset: time.sleep(0.001)

        # Stop timing the epoch training loop
        time_epoch_tock = time.perf_counter()
        
        print("Epoch time:",time_epoch_tock-time_epoch_tick)
        
    # Stop timing the entire training loop
    time_total_tock = time.perf_counter()
    
    print("Total time:",time_total_tock-time_total_tick)

#=============================================================================#