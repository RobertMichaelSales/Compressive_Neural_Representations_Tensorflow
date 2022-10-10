""" Created: 01.06.2022  \\  Updated: 01.06.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def SaveNPY(filename,x_points,y_points,z_points,values):
    
    import numpy as np

    from numpy import save
    
    dtype = np.float32
    
    x_points = x_points.flatten()
    y_points = y_points.flatten()
    z_points = z_points.flatten()
    values   = values.flatten()
    
    data = np.array([x_points,y_points,z_points,values],dtype=dtype).transpose()
    
    save(filename, data)
    
    return None    
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>