""" Created: 01.06.2022  \\  Updated: 09.11.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def NPYtoVTK_3D(npy_filename):
    
    # Import libraries
    import numpy as np
    from pyevtk.hl import gridToVTK
    
    # Configure filename
    vtk_filename = npy_filename.replace(".npy","")
    
    # Load data from numpy array
    npy_data = np.load(npy_filename)
        
    # Create a list of the coordinate positions
    volume_list = [np.ascontiguousarray(npy_data[...,x]) for x in range(3)]
    
    # Create a dictionary of the scalar values
    values_dict = {"values" : np.ascontiguousarray(npy_data[...,-1])}
    
    # Save point data to VTK file    
    gridToVTK(vtk_filename,*volume_list,pointData=values_dict)
    
    return None    
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import os

input_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes"

input_filenames = ["test_vol.npy"]

# Iterate through all specified files
for input_filename in input_filenames:
    
    npy_filename = os.path.join(input_directory,input_filename)

    if os.path.exists(npy_filename):
        print("Converting file: {}".format(npy_filename))
        NPYtoVTK_3D(npy_filename)
    else: 
        print("Could not find file '{}'.\n".format(npy_filename))
        
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>