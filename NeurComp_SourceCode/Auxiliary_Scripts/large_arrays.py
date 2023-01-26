""" Created: 25.01.2023  \\  Updated: 26.01.2023  \\   Author: Robert Sales """

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
    
    threshold_memory = 4 * 1024 * 1024 * 1024
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
            data = LoadLargeArrayNpy(filename=filename,dtype=dtype,shape=shape)
            
        elif file_type == "bin":
            data = LoadLargeArrayBin(filename=filename,dtype=dtype,shape=shape)
            
        else: print("File type '.{}' not supported. Try using '.npy' or '.bin'")
   
    else:
        
        print("\nFile size < available/threshold memory. Loading as normal.")
        
        if file_type == "npy":
            data = LoadSmallArrayNpy(filename=filename,dtype=dtype,shape=shape)
            
        elif file_type == "bin":
            data = LoadSmallArrayBin(filename=filename,dtype=dtype,shape=shape)
            
        else: print("File type '.{}' not supported. Try using '.npy' or '.bin'")
    
    ##    
        
    return data
        
    
def LoadSmallArrayNpy(filename,dtype,shape):
    
    import numpy as np
    
    data = np.ascontiguousarray(np.load(filename).astype(dtype).reshape(shape))
    
    return data


def LoadSmallArrayBin(filename,dtype,shape):
    
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


def LoadLargeArrayNpy(filename,dtype,shape):
    
    import numpy as np
    
    data = np.lib.format.open_memmap(filename=filename,mode="r",dtype=dtype,shape=shape)
    
    return data


def LoadLargeArrayBin(filename,dtype,shape):
    
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

import numpy as np

# data = LoadArray(filename="/home/rms221/Documents/Miscellaneous/large_array.npy",dtype=np.float32,shape=(100,100,100,100,15))
# data = LoadArray(filename="/home/rms221/Documents/Miscellaneous/large_array.bin",dtype=np.float32,shape=(100,100,100,100,15))
# data = LoadArray(filename="/home/rms221/Documents/Miscellaneous/small_array.npy",dtype=np.float32,shape=(10 ,10 ,10 ,10 ,15))
# data = LoadArray(filename="/home/rms221/Documents/Miscellaneous/small_array.bin",dtype=np.float32,shape=(10 ,10 ,10 ,10 ,15))
