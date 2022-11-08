""" Created: 01.06.2022  \\  Updated: 13.06.2022  \\   Author: Robert Sales """

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def npy2vtk(npy_filename,dimensions=3):
    
    
    from numpy import load
    from pyevtk.hl import gridToVTK
    
    # Load data from numpy array
    data = load(npy_filename)
    
    # Create an empty list for coordinate data i.e. [x,y,...,z]
    coords = []
     
    # Extract coordinate data
    for dim in range(dimensions):coords.append(data[...,dim].flatten())
        
    # Extract volume data
    volume = data[...,3].flatten()
    
    # Configure filename
    vtk_filename = npy_filename.replace(".npy","")
    
    # Save to vtk type file
    gridToVTK(vtk_filename,*coords,pointData={"data":volume})
    
    return None    
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

import os

input_filenames = ["/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/neurcomp_1/output_volume.npy"]

# Iterate through all specified files
for input_file in input_filenames:

    if os.path.exists(input_file):
        print("Converting file: {}".format(input_file))
        npy2vtk(input_file)
    else: 
        print("Could not find file '{}'.\n".format(input_file))
        
    
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>