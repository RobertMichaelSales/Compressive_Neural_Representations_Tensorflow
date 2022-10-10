""" Created: 01.06.2022  \\  Updated: 01.06.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def LoadTXT(filename,size_x,size_y,size_z):
    
    from numpy import loadtxt

    data = loadtxt(filename,delimiter=',')
        
    x_points = combined_data[:,0].reshape(size_x,size_y,size_z)
    y_points = combined_data[:,1].reshape(size_x,size_y,size_z)
    z_points = combined_data[:,2].reshape(size_x,size_y,size_z)
    volume   = combined_data[:,3].reshape(size_x,size_y,size_z)
    
    return x_points,y_points,z_points,volume  
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>