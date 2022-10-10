""" Created: 18.07.2022  \\  Updated: 18.07.2022  \\   Author: Robert Sales """

#=# IMPORT LIBRARIES #========================================================#

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

#=# DEFINE FUNCTIONS #========================================================#

def LoadTXT(filepath):
    
    from numpy import loadtxt
    combined_data = loadtxt(filepath,delimiter=',')
    
    x_points = combined_data[:,0]
    y_points = combined_data[:,1]
    z_points = combined_data[:,2]
    values   = combined_data[:,3]
      
    volume = np.array(list(zip(x_points,y_points,z_points)),dtype=np.float32)
    values = np.array(values,dtype=np.float32)
      
    return volume,values


def LoadNPY(filepath):

    from numpy import load
    combined_data = load(filepath)
      
    x_points = combined_data[:,0]
    y_points = combined_data[:,1]
    z_points = combined_data[:,2]
    values   = combined_data[:,3]
    
    volume = np.array(list(zip(x_points,y_points,z_points)),dtype=np.float32)
    values = np.array(values,dtype=np.float32)
    
    return volume,values


def LoadBIN(filepath):

    import numpy as np
    
    data_vector = np.fromfile(filepath,dtype=np.float32,count=-1,sep='')
    combined_data = np.reshape(data_vector,(int(data_vector.size/4),4))
    
    x_points = combined_data[:,0]
    y_points = combined_data[:,1]
    z_points = combined_data[:,2]
    values   = combined_data[:,3]
    
    volume = np.array(list(zip(x_points,y_points,z_points)),dtype=np.float32)
    values = np.array(values,dtype=np.float32)
    
    return volume,values


def LoadEXT(filepath):

    ### Insert code here for any other file extension.
    volume,values = None,None 
    ### Insert code here for any other file extension.
    
    return volume,values


def LoadData(filepath):
    
    extension = filepath.split(".")[-1].lower()
    volume,values = None,None
    
    if extension not in ["npy","txt"]:
        print("Error: Data file extension is not accepted.")
    
    elif extension == "npy":
        volume,values = LoadNPY(filepath)    
        
    elif extension == "txt":
        volume,values = LoadTXT(filepath)
        
    return volume,values


def SaveNPY(volume,values,filename):
    
    from numpy import save
    
    x_points = volume[:,0].flatten()
    y_points = volume[:,1].flatten()
    z_points = volume[:,2].flatten()
    values   = values.flatten()
    
    data = np.array([x_points,y_points,z_points,values]).transpose()
    
    save(filename, data)
    
    return None   


def SaveTXT(volume,values,filename):
        
    from numpy import savetxt
    
    x_points = volume[:,0].flatten()
    y_points = volume[:,1].flatten()
    z_points = volume[:,2].flatten()
    values   = values.flatten()
    
    data = np.array([x_points,y_points,z_points,values]).transpose()
    
    fmt = '%16.15e', '%16.15e', '%16.15e', '%+16.15e'

    savetxt(filename,data,delimiter=',',fmt=fmt)
    
    return None   


def SaveBIN(volume,values,filename):
        
    from numpy import savetxt
    
    x_points = volume[:,0].flatten()
    y_points = volume[:,1].flatten()
    z_points = volume[:,2].flatten()
    values   = values.flatten()
    
    data = np.array([x_points,y_points,z_points,values],dtype=np.float32).transpose()
    
    data.tofile(filename,sep='',format='%s')
    
    return None   


def SaveEXT(volume,values,filename):
        
    from numpy import savetxt
    
    x_points = volume[:,0].flatten()
    y_points = volume[:,1].flatten()
    z_points = volume[:,2].flatten()
    values   = values.flatten()
    
    data = np.array([x_points,y_points,z_points,values]).transpose()
    
    ### Insert code here for any other file extension.
    
    return None   


def MakeDataset(input_data,hyperparameters):
    
    values = input_data.values
    volume = input_data.volume
        
    if hyperparameters.normalise: 
        values = 2.0*(((values-values.min())/(values.max()-values.min()))-0.5)
        
    dataset = tf.data.Dataset.from_tensor_slices((volume,values))
    
    dataset = dataset.shuffle(buffer_size=values.size,
                              seed=12345,
                              reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size=hyperparameters.batch_size)
    
    return dataset


# def GetPrimeFactors(n):
    
#     prime_factors = []
    
#     i = 2
    
#     while i <= n:
        
#         if (n%i == 0):
#             n = n / i
#             prime_factors.append(i)
            
#         else:
#             i = i + 1
        
#     return prime_factors   
    
    
#=# DEFINE CLASSES #==========================================================#

class DataClass():

    def __init__(self):
        
        self.volume = np.array
        self.values = np.array
        self.volume_max = None
        self.volume_min = None
        
        return None
    
    def LoadValues(self,filepath):
        
        print("Loading Data: '{}'.\n".format(filepath))
        
        extension = filepath.split(".")[-1].lower()
        volume,values = None,None
        
        if extension not in ["npy","txt"]:
            print("Extension Not Supported: '{}'. ".format(extension),end='')
            print("Cannot Load: '{}'.\n".format(filepath))
            return None
    
        elif extension == "npy":
            volume,values = LoadNPY(filepath)    
            
        elif extension == "txt":
            volume,values = LoadTXT(filepath)
        
        self.volume = volume
        self.values = values
        self.volume_max = self.volume.max()
        self.volume_min = self.volume.min()
        
        return None
    
    def SaveValues(self,filepath,name,extension):
        
        filename = os.path.join(filepath,name+extension)
        
        if extension not in [".npy",".txt",".bin"]:
            print("Extension Not Supported: '{}'. ".format(extension),end='')
            print("Cannot Save: '{}'.\n".format(filename))
            return None            
    
        if extension == ".npy": 
            SaveNPY(self.volume,self.values,filename)
            
        if extension == ".txt": 
            SaveTXT(self.volume,self.values,filename)
            
        if extension == ".bin": 
            SaveBIN(self.volume,self.values,filename)
        
        print("Saving Data: '{}'.\n".format(filename))

        return None

#=============================================================================#