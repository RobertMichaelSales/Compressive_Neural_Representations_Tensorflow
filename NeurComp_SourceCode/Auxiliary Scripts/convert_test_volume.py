""" Created: 01.06.2022  \\  Updated: 09.11.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def ConvertTestVol(filename_in,filename_out):
    
    # Load data from numpy array
    import numpy as np
    
    dtype = np.float32
    linsp = np.linspace(0,1,150)
    
    x,y,z = np.meshgrid(linsp,linsp,linsp,indexing="ij")
    values = np.load(filename_in)
    
    data = np.stack((x,y,z,values),axis=-1).astype(dtype)
    
    np.save(filename_out, data)
    
    return None
    
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>