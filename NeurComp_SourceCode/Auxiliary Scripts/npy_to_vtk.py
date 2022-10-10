""" Created: 01.06.2022  \\  Updated: 13.06.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def npy2vtk(npy_filename,vtk_filename,size_x,size_y,size_z):
    
    # Load data from numpy array
    from numpy import load
    
    data = load(npy_filename)
     
    x_points = data[:,0].reshape(size_x,size_y,size_z)
    y_points = data[:,1].reshape(size_x,size_y,size_z)
    z_points = data[:,2].reshape(size_x,size_y,size_z)
    volume   = data[:,3].reshape(size_x,size_y,size_z)
        
    # Save data to a vts filetype
    from pyevtk.hl import gridToVTK
    
    # Condition filename
    vtk_filename = vtk_filename.replace(".vts","")
    
    gridToVTK(vtk_filename,x_points,y_points,z_points,pointData={"data": volume})
    
    return None    
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import os

parent_filepath = "C:/Users/sales/Documents/Cambridge University/PPD - Project Proposal Dissertation/NeurComp Files/Outputs/Volumes/NeurComp_V6"
input_filenames = ["difference.npy"] # ["original.npy","normal.npy","tflite.npy","quantd.npy"]

size_x,size_y,size_z = 150,150,150 #49,133,49

# Iterate through all specified files
for input_filename in input_filenames:

    # Create absolute path for input and output files
    file_npy = os.path.join(parent_filepath,input_filename)
    file_vtk = file_npy.replace(".npy",".vts")

    if os.path.exists(file_npy):
        print("Converting: '{}' -> '{}'.\n".format(file_npy,file_vtk))
        npy2vtk(file_npy,file_vtk,size_x,size_y,size_z)
    else: 
        print("Could not find file '{}'.\n".format(file_npy))
    
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>