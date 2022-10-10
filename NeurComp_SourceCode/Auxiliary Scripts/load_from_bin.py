""" Created: 31.07.2022  \\  Updated: 31.07.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def LoadBIN(filename,size_x,size_y,size_z):

    import numpy as np
    
    raw_data = np.fromfile(filepath,dtype=np.float32,count=-1,sep='')
    combined_data = np.reshape(raw_data,((size_x*size_y*size_z),4))
    
    x_points = combined_data[:,0]
    y_points = combined_data[:,1]
    z_points = combined_data[:,2]
    values   = combined_data[:,3]
        
    return x_points,y_points,z_points,volume  
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>