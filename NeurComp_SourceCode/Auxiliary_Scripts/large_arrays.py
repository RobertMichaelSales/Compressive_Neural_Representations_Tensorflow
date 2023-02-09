""" Created: 25.01.2023  \\  Updated: 27.01.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

#==============================================================================

def MakeLargeArrayNpy(filename="/home/rms221/Documents/Miscellaneous/large_array.npy"):
    
    import numpy as np
    
    data = np.random.rand(100,100,100,100,15).astype("float32")
    
    np.save(filename,data)
    
    return None


def MakeLargeArrayBin(filename="/home/rms221/Documents/Miscellaneous/large_array.bin"):
    
    import numpy as np
    
    resolution = (100,100,100,100,15)
    resolution_as_bytes = np.array(resolution).astype("uint32").tobytes()
    
    dimensions = len(resolution)
    dimensions_as_bytes = np.array(dimensions).astype("uint32").tobytes()
    
    dtype_size = len(np.array([1.0]).astype("float32").tobytes())
    dtype_size_as_bytes = np.array(dtype_size).astype("uint32").tobytes()
    
    header = dimensions_as_bytes + resolution_as_bytes + dtype_size_as_bytes
    
    with open (filename,"wb") as file:
        
        file.write(header)
        file.flush()
        
    file.close()
    
    data = np.memmap(filename=filename,dtype="float32",mode="r+",offset=len(header),shape=resolution,order="C")
    
    for i in range(resolution[-1]):
        data[...,i] = np.random.rand(100,100,100,100).astype("float32")
        data.flush()
    
    return None


def MakeSmallArrayNpy(filename="/home/rms221/Documents/Miscellaneous/small_array.npy"):
    
    import numpy as np
    
    data = np.random.rand(10 ,10 ,10 ,10 ,15).astype("float32")
    
    np.save(filename,data)
    
    return None


def MakeSmallArrayBin(filename="/home/rms221/Documents/Miscellaneous/small_array.bin"):
    
    import numpy as np
        
    resolution = (10 ,10 ,10 ,10 ,15)
    resolution_as_bytes = np.array(resolution).astype("uint32").tobytes()
    
    dimensions = len(resolution)
    dimensions_as_bytes = np.array(dimensions).astype("uint32").tobytes()
    
    dtype_size = len(np.array([1.0]).astype("float32").tobytes())
    dtype_size_as_bytes = np.array(dtype_size).astype("uint32").tobytes()
    
    header = dimensions_as_bytes + resolution_as_bytes + dtype_size_as_bytes
    
    data = np.random.rand(10 ,10 ,10 ,10 ,15).astype("float32")
    data_as_bytes = np.ravel(data,order="C").astype("float32").tobytes()    
    
    with open (filename,"wb") as file:
        
        file.write(header)
        file.write(data_as_bytes)
        file.flush()
        
    file.close()
        
    return None


def LoadArray(filename,dtype,shape):
    
    import psutil, os
        
    # Determine available/threshold memory and compare the minimum to file size
    available_memory = psutil.virtual_memory().available
    print("Available Memory: {:12d} Bytes ({:5.2f} GigaBytes)".format(available_memory,(available_memory/1024**3)))
    
    threshold_memory = int(0.01 * 1024 * 1024 * 1024)
    print("Threshold Memory: {:12d} Bytes ({:5.2f} GigaBytes)".format(threshold_memory,(threshold_memory/1024**3)))
    
    file_size = os.path.getsize(filename)
    print("File Size:        {:12d} Bytes ({:5.2f} GigaBytes)".format(file_size,(file_size/1024**3)))
    
    file_larger_than_memory = (file_size > min(available_memory,threshold_memory))
    
        
    # Determine the file type to load. Accepted types include '.npy' and '.bin'
    file_type = filename.split(".")[-1] 
        
    
    # Load the file, either directly, or using a memory mapping, i.e. np.memmap
    if file_larger_than_memory:
        
        print("\nFile size > available/threshold memory. Loading as memmap.")
        
        if file_type == "npy":
            data = LoadLargeArrayNpy(filename=filename,shape=shape)
            
        elif file_type == "bin":
            data = LoadLargeArrayBin(filename=filename,shape=shape)
            
        else: print("File type '.{}' not supported. Try using '.npy' or '.bin'")
   
    else:
        
        print("\nFile size < available/threshold memory. Loading as normal.")
        
        if file_type == "npy":
            data = LoadSmallArrayNpy(filename=filename,shape=shape)
            
        elif file_type == "bin":
            data = LoadSmallArrayBin(filename=filename,shape=shape)
            
        else: print("File type '.{}' not supported. Try using '.npy' or '.bin'")
    
    ##    
        
    return data
        
    
def LoadSmallArrayNpy(filename,shape):
    
    import numpy as np
    
    data = np.ascontiguousarray(np.load(filename).astype("float32").reshape(shape))
    
    return data


def LoadSmallArrayBin(filename,shape):
    
    with open(filename,"rb") as file:
        
        dimensions_as_bytes = file.read(4)
        dimensions = int(np.frombuffer(dimensions_as_bytes,dtype="uint32"))
        
        resolution_as_bytes = file.read(4 * dimensions)
        resolution = tuple(np.frombuffer(resolution_as_bytes,dtype="uint32"))
        
        dtype_size_as_bytes = file.read(4)
        dtype_size = int(np.frombuffer(dtype_size_as_bytes,dtype="uint32"))
        
        data_as_bytes = file.read(int(np.prod(resolution)*dtype_size))
        data = np.reshape(np.frombuffer(data_as_bytes,dtype="float32"),newshape=resolution,order="C")
        
    ##
    
    return data


def LoadLargeArrayNpy(filename,shape):
    
    import numpy as np
    
    data = np.lib.format.open_memmap(filename=filename,mode="r",dtype="float32",shape=shape)
    
    return data


def LoadLargeArrayBin(filename,shape):
    
    with open(filename,"rb") as file:
        
        dimensions_as_bytes = file.read(4)
        dimensions = int(np.frombuffer(dimensions_as_bytes,dtype="uint32"))
        
        resolution_as_bytes = file.read(4 * dimensions)
        resolution = tuple(np.frombuffer(resolution_as_bytes,dtype="uint32"))
        
        dtype_size_as_bytes = file.read(4)
        dtype_size = int(np.frombuffer(dtype_size_as_bytes,dtype="uint32"))
        
        header = dimensions_as_bytes + resolution_as_bytes + dtype_size_as_bytes
    ##
        
    data = np.memmap(filename=filename,dtype="float32",mode="r",offset=len(header),shape=resolution,order="C")
          
    return data


#==============================================================================


def CreateDataGen(volume,values):
    
    def DataGen():
        
        for volume_item,values_item in zip(volume,values):
            
            yield volume_item,values_item
            
    return DataGen
        

def MakeTensorflowDataset(volume,values,batch_size,repeat=False):
    
    import tensorflow as tf
    
    generator = CreateDataGen(volume,values)
    
    output_types = (tf.float32,tf.float32)
    
    output_shapes = (tf.TensorShape((volume.shape[-1],)),tf.TensorShape(values.shape[-1],))
        
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

# data = LoadArray(filename="/home/rms221/Documents/Miscellaneous/large_array.npy",dtype=np.float32,shape=(100,100,100,100,15))
# data = LoadArray(filename="/home/rms221/Documents/Miscellaneous/large_array.bin",dtype=np.float32,shape=(100,100,100,100,15))
# data = LoadArray(filename="/home/rms221/Documents/Miscellaneous/small_array.npy",dtype=np.float32,shape=(10 ,10 ,10 ,10 ,15))
# data = LoadArray(filename="/home/rms221/Documents/Miscellaneous/small_array.bin",dtype=np.float32,shape=(10 ,10 ,10 ,10 ,15))

filename = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/cube.npy"

data = LoadArray(filename=filename,dtype=np.float32,shape=(150,150,150,4))

volume,values = data[...,:3].reshape(-1,3),data[...,3:].reshape(-1,1)

dataset = MakeTensorflowDataset(volume=volume,values=values,batch_size=1024)

del data, volume, values, filename
