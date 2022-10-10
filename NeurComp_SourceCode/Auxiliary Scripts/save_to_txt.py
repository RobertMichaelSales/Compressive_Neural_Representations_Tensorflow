""" Created: 01.06.2022  \\  Updated: 01.06.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def SaveTXT(filename,x_points,y_points,z_points,values):
    
    from numpy import savetxt
    
    x_points = x_points.flatten()
    y_points = y_points.flatten()
    z_points = z_points.flatten()
    values   = values.flatten()
    
    data = np.array([x_points,y_points,z_points,values],dtype=np.float32).transpose()
    
    fmt = '%16.15e', '%16.15e', '%16.15e', '%+16.15e'

    savetxt(filename + ".txt",data,delimiter=',',fmt=fmt)
    
    return None    
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>