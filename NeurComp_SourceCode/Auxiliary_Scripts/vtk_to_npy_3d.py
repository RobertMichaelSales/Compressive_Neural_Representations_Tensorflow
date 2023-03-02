""" Created: 13.06.2022  \\  Updated: 18.01.2023  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def GetVarsFromVTK(vtk_filename):
        
    # Set up a vtk file reader
    reader = vtk.vtkXMLStructuredGridReader()
    
    # Set file name for reader
    reader.SetFileName(vtk_filename)
    
    # Update the reader object
    reader.Update()
    
    # Create a list for variable names
    variables = []
    
    print("\nThe point-data variables stored in {} are:".format(vtk_filename.split("/")[-1]))
    
    # Iterate through all of the fields
    for index in range(reader.GetOutput().GetPointData().GetNumberOfArrays()):

        variables.append(reader.GetOutput().GetPointData().GetArrayName(index))
        
        print("'{}'".format(variables[-1]))
    
    return variables

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def PickVarFromList(variables):
    
    variable = ""
    
    while(variable not in variables):
        
        variable = input("\nSelect the point-data variable to convert: ")
    
    print("'{}'".format(variable))
    
    return variable 

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def VTKtoNPY_3D(vtk_filename,variable_name,shape,check_save):
    
    npy_filename = vtk_filename.replace(".vts","") # + "_" + variable_name
    
    print("\nConverting file: {} -> {}.npy".format(vtk_filename.split("/")[-1],npy_filename.split("/")[-1]))    
            
    # Create a vtk file reader object to read structured grids
    reader = vtk.vtkXMLStructuredGridReader()
    
    # Set the input filename for the VTK reader
    reader.SetFileName(vtk_filename)
    
    # Update the reader object to obtain values
    reader.Update()
    
    # Initialise a dictionary to store the scalars 
    variables = {}
    
    # Iterate through each point array
    for index in range(reader.GetNumberOfPointArrays()):
        
        # Get the variable name
        label = reader.GetOutput().GetPointData().GetArrayName(index)
        
        # Get the scalar values
        value = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(index))
        
        # Add the name and values to a dictionary
        variables[label] = value
    ##
        
    # Get the x,y,z coordinates of each grid point
    volume = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
        
    # Get the scalar values at each grid point 
    values = np.expand_dims(variables[variable_name],axis=-1)
        
    # Cast the arrays to float32 precision
    volume = volume.astype(np.float32)
    values = values.astype(np.float32)
    
    # Concatenate volume and values arrays
    npy_data = np.concatenate((volume,values),axis=-1)   
        
    # Reshape the array according to the user's resolution
    npy_data = np.reshape(npy_data,(shape[2],shape[0],shape[1],-1),order="F")
    
    # Transpose from z,x,y -> x,y,z index ordering
    npy_data = np.transpose(npy_data,(1,2,0,3))
    
    # Save the data to a file
    np.save(npy_filename, npy_data)
    
    if check_save: 
        CheckSavedAsVTK(vtk_filename,variable_name,npy_data,shape)
    else: pass
    
    print("\nSuccess!")
    
    return npy_data
        
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
def CheckSavedAsVTK(vtk_filename,variable_name,npy_data,shape):
    
    # Add to the vtk filename
    new_vtk_filename = vtk_filename.replace(".vts","_check_") + variable_name
    
    # Create a list of the coordinate positions
    volume_list = [np.ascontiguousarray(npy_data[...,x]) for x in range(len(shape))]
    
    # Create a dictionary of the scalar values
    values_dict = {variable_name: np.ascontiguousarray(npy_data[...,-1])}
    
    # Save point data to VTK file    
    gridToVTK(new_vtk_filename,*volume_list,pointData=values_dict)

    return None

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# Import libraries
import os, vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from pyevtk.hl import gridToVTK

# Specify the vtk file name
vtk_filename = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/passage.vts"

# Specify volume shape
shape = (133,49,49)

if os.path.exists(vtk_filename):

    # Pick which variables
    variable_name = PickVarFromList(GetVarsFromVTK(vtk_filename))
    
    # Convert specifed variable
    npy_data = VTKtoNPY_3D(vtk_filename,variable_name,shape,check_save=False)    
    
else: 
    print("File not found '{}'.".format(vtk_filename.split("/")[-1]))
        
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>