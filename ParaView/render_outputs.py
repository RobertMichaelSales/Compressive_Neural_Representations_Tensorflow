""" Created: 22.08.2023  \\  Updated: 22.08.2023  \\   Author: Robert Sales """

# Note: Run this file from the command line using "pvpython filename.py"

#==============================================================================

import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 9

from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()
ResetSession()

import numpy as np
import os, json, glob, vtk, sys

from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter

#==============================================================================

def RenderOutput(vtk_filename,values_range,output_type):
    
    #==========================================================================
    ## Load the dataset

    # Extract filename extension
    vtk_filename_ext = os.path.splitext(vtk_filename)[-1]
    
    # Unstructured reader
    if vtk_filename_ext == ".vtu":
        vtk_data = XMLUnstructuredGridReader(FileName=[vtk_filename])
    ##
    
    # Structured reader
    if vtk_filename_ext == ".vts":
        vtk_data = XMLStructuredGridReader(FileName=[vtk_filename])
    ##
    
    #==========================================================================
    
    # Set output filepath
    render_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/ParaView/renders"

    #==========================================================================
    ## Set the variable
        
    # Get the data array and values range
    dataset = servermanager.Fetch(vtk_data)
    variable_name = dataset.GetPointData().GetArray(0).GetName()
    vtk_data = PassArrays(Input=vtk_data)
    vtk_data.PointDataArrays = [variable_name]
    if not values_range: values_range = vtk_data.PointData[variable_name].GetRange()
    UpdatePipeline(time=0.0, proxy=vtk_data)
    
    #==========================================================================
    ## Show the dataset
    
    # Create render view
    view = GetActiveViewOrCreate('RenderView')
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0 
    view.AxesGrid = None
    view.CenterAxesVisibility = 0
    
    # Display the surface
    display = Show(vtk_data, view)
    display.Representation = 'Surface'
    
    # Set lighting
    display.SpecularPower = 80
    display.Diffuse = 0.5
    display.Ambient = 0.8
    display.Specular = 0.0
    
    # Colour by variable
    ColorBy(display, ('POINTS', variable_name))
    display.RescaleTransferFunctionToDataRange(True, False)
    
    # Apply color preset
    lut = GetColorTransferFunction(variable_name)
    lut.ApplyPreset('Cool to Warm', True)
    lut.RescaleTransferFunction(values_range[0],values_range[1])
    
    #==========================================================================
    ## Configure camera
    
    # Compute centre, range, and view distance
    bounds = vtk_data.GetDataInformation().GetBounds()    
    centre = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
    ranges = [(bounds[1]-bounds[0])  , (bounds[3]-bounds[2])  , (bounds[5]-bounds[4])  ]
    view_distance = 2.5 * max(ranges)
         
    #==========================================================================
    ## Render surfaces
    
    # Set camera to +X
    view.CameraPosition = (np.array(centre) + np.array([+view_distance,0,0])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,0,1]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_surface_+x.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)
    
    # Set camera to -X
    view.CameraPosition = (np.array(centre) + np.array([-view_distance,0,0])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,0,1]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_surface_-x.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)
    
    # Set camera to +Y
    view.CameraPosition = (np.array(centre) + np.array([0,+view_distance,0])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,0,1]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_surface_+y.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)
    
    # Set camera to +X
    view.CameraPosition = (np.array(centre) + np.array([0,-view_distance,0])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,0,1]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_surface_-y.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)
    
    # Set camera to +Z
    view.CameraPosition = (np.array(centre) + np.array([0,0,+view_distance])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,1,0]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_surface_+z.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)
    
    # Set camera to -Z
    view.CameraPosition = (np.array(centre) + np.array([0,0,-view_distance])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,1,0]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_surface_-z.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)
    
    display.Visibility = 0
    
    #==========================================================================
    ## Display contours
    
    vtk_data = Contour(Input=vtk_data)
    vtk_data.ContourBy = ['POINTS', variable_name]
    vtk_data.Isosurfaces = np.linspace(values_range[0],values_range[1],20).tolist()
    vtk_data.PointMergeMethod = "Uniform Binning"
    
    # Display the surface
    display = Show(vtk_data, view)
    display.Representation = 'Surface'
    
    # Set lighting
    display.SpecularPower = 80
    display.Diffuse = 0.5
    display.Ambient = 0.8
    display.Specular = 0.0
    
    # Colour by variable
    ColorBy(display, ('POINTS', variable_name))
    display.RescaleTransferFunctionToDataRange(True, False)
    
    # Apply color preset
    lut = GetColorTransferFunction(variable_name)
    lut.ApplyPreset('Cool to Warm', True)
    lut.RescaleTransferFunction(values_range[0],values_range[1])
    
    #==========================================================================
    ## Render contours
    
    # Set camera to +X
    view.CameraPosition = (np.array(centre) + np.array([+view_distance,0,0])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,0,1]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_contour_+x.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)
    
    # Set camera to -X
    view.CameraPosition = (np.array(centre) + np.array([-view_distance,0,0])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,0,1]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_contour_-x.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)
    
    # Set camera to +Y
    view.CameraPosition = (np.array(centre) + np.array([0,+view_distance,0])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,0,1]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_contour_+y.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)
    
    # Set camera to +X
    view.CameraPosition = (np.array(centre) + np.array([0,-view_distance,0])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,0,1]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_contour_-y.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)
    
    # Set camera to +Z
    view.CameraPosition = (np.array(centre) + np.array([0,0,+view_distance])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,1,0]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_contour_+z.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)
    
    # Set camera to -Z
    view.CameraPosition = (np.array(centre) + np.array([0,0,-view_distance])).tolist()
    view.CameraFocalPoint = centre
    view.CameraViewUp = [0,1,0]
    Render()    
    SaveScreenshot(os.path.join(render_filepath,"temp_{:}_contour_-z.png".format(output_type)),view,ImageResolution=[2000,2000],TransparentBackground=True,CompressionLevel=0)

    #==========================================================================    

    return values_range
    
##

#==============================================================================

if __name__ == "__main__":
    
    # This block will run in the event that this script is called in an IDE
    if (len(sys.argv) == 1):
                
        experiment_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/AuxFiles/outputs/test/"
      
    # This block will run in the event that this script is run via terminal        
    else: 

        experiment_filepath  = sys.argv[1]

    ##
        
    values_range = None
    
    vtk_filename = os.path.join(experiment_filepath,"i_dataset.vtu")
    if not os.path.isfile(vtk_filename): raise FileExistsError("File '{:}' does not exist!".format(vtk_filename))
    values_range = RenderOutput(vtk_filename,values_range,"true")
    
    vtk_filename = os.path.join(experiment_filepath,"o_dataset.vtu")
    if not os.path.isfile(vtk_filename): raise FileExistsError("File '{:}' does not exist!".format(vtk_filename))
    values_range = RenderOutput(vtk_filename,values_range,"pred")

else: pass

#==============================================================================