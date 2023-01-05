""" Created: 13.06.2022  \\  Updated: 09.11.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def VTKtoNPY_3D(vtk_filename,variable_name,shape):
    
    # Import libraries
    import vtk
    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy
    
    # Configure filename
    npy_filename = vtk_filename.replace(".vts","_test")
        
    # Set up a vtk file reader
    reader = vtk.vtkXMLStructuredGridReader()
    
    # Set file name for reader
    reader.SetFileName(vtk_filename)
    
    # Update the reader object
    reader.Update()
        
    # Get the coordinates of the grid
    flat_volume = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
    volume = np.reshape(flat_volume,(output_shape+(-1,)),order="F")
    
    # Initialise the 'values' dictionary
    values_dictionary = {}
    
    # Iterate through all of the fields
    for index in range(reader.GetOutput().GetPointData().GetNumberOfArrays()):
        
        # Get the scalars stored at grid points
        array = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(index))
        
        # Get the name of the scalar field
        label = reader.GetOutput().GetPointData().GetArrayName(index)    
        
        # Add the fields to the 'values' dictionary
        values_dictionary[label] = array
        
    # Check that the desired variable exists
    if variable_name in values_dictionary.keys():
        
        # Extract the desired variable scalar field
        flat_values = values_dictionary[variable_name]
        values = np.reshape(flat_values,(output_shape+(-1,)),order="F")
        
        # Save the data to a file
        np.save(npy_filename,np.concatenate((volume,values),axis=-1))
        
        print("Converting: Success!")
        
    else:        
        print("No such variable: {}".format(variable_name))
        
    return None
        
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# Options: "r, rt, ro, mut, Vx, Vy, Vz, V, T, M, To, P, Po, alpha, beta, sfunc"

import os

# Specify the parent directory
input_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes"

# Specify file names
input_filenames = ["test_vol.vts"]

# Specify variable name 
variable_names = ["values"]

# Specify volume shape
output_shape = (150,150,150)

# Iterate through all specified files
for input_filename in input_filenames:
    
    # Iterate through all specified variables
    for variable_name in variable_names:

        # Create absolute path for input and output files
        vtk_filename = os.path.join(input_directory,input_filename)    

        if os.path.exists(vtk_filename):
            print("Converting File: {}".format(vtk_filename))
            VTKtoNPY_3D(vtk_filename,variable_name,output_shape)
        else: 
            print("Could not find file '{}'.".format(vtk_filename))
        
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>