""" Created: 13.06.2022  \\  Updated: 15.06.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def vtk2npy(vtk_filename,npy_filename,var_name):
    
    import vtk
    
    import numpy as np
    
    from vtk.util.numpy_support import vtk_to_numpy
    
    from save_to_npy import SaveNPY
    
    # Set up reader
    reader = vtk.vtkXMLStructuredGridReader()
    
    # Set file name for reader
    reader.SetFileName(vtk_filename)
    
    # Update the reader object
    reader.Update()
        
    # Get the coordinates of the nodes in the mesh
    field = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
    
    # Initialise the 'volume' dictionary
    scalar_volume = {}
    
    # Iterate through all of the scalar fields
    for index in range(reader.GetOutput().GetPointData().GetNumberOfArrays()):
        
        # Get the scalars stored at nodes in the mesh
        array_field = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(index))
        
        # Get the name of the scalar field
        array_name = reader.GetOutput().GetPointData().GetArrayName(index)    
        
        # Add the fields to the 'volume' dictionary
        scalar_volume[array_name] = array_field
        
    # Check the desired variable exists
    if var_name in scalar_volume.keys():
        
        # Extract the desired variable scalar field
        volume = scalar_volume[var_name]
        
        # Save the data to a file
        SaveNPY(npy_filename,field[:,0],field[:,1],field[:,2],volume)
        
        print("Converting: Success!")
        
    else:
        
        print("Converting: Error!")
        
    return None
        
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# Options: "r, rt, ro, mut, Vx, Vy, Vz, V, T, M, To, P, Po, alpha, beta, sfunc"

import os

# Specify names of files to convert
filenames = [] #["out1","out2","out3","out4","out5"]

# Specify variable name 
var_names = ["Po","sfunc"]

# Specify the parent directory
parent_directory = 'C:\\Users\\sales\\Documents\\Cambridge University\\PPD - Project Proposal Dissertation\\Test Fields\\Multall Volume'

# Iterate through all specified files
for filename in filenames:
    
    # Iterate through all specified variables
    
    for var_name in var_names:

        # Create absolute path for input and output files
        file_vtk = os.path.join(parent_directory,filename+".vts")    
        file_npy = os.path.join(parent_directory,filename+"_"+var_name+".npy")
        
        if os.path.exists(file_vtk):
            print("Converting: {} -> {}.".format(filename+".vts",filename+"_"+var_name+".npy"))
            vtk2npy(file_vtk,file_npy,var_name)
        else: 
            print("Could not find file '{}'.".format(filename+".vts"))
        
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>