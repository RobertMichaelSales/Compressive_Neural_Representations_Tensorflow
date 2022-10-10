""" Created: 31.07.2022  \\  Updated: 31.07.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def LoadBIN(filename,x_points,y_points,z_points,values):

    import numpy as np
    
    from numpy import savetxt
    
    x_points = x_points.flatten()
    y_points = y_points.flatten()
    z_points = z_points.flatten()
    values   = values.flatten())
    
    data = np.array([x_points,y_points,z_points,values],dtype=np.float32).transpose()
    
    data.tofile(filename,sep='',format='%s')
    
    return None    
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>