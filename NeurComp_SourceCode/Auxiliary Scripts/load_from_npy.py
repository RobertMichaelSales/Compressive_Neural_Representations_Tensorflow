""" Created: 01.06.2022  \\  Updated: 01.06.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def LoadNPY(filename,size_x,size_y,size_z):
    
    # Load data from numpy array
    import numpy as np
    
    from numpy import load
    
    dtype = np.float32
    
    data = load(filename)
     
    x_points = data[:,0].reshape(size_x,size_y,size_z)
    y_points = data[:,1].reshape(size_x,size_y,size_z)
    z_points = data[:,2].reshape(size_x,size_y,size_z)
    volume   = data[:,3].reshape(size_x,size_y,size_z)
    
    return x_points,y_points,z_points,volume  
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>