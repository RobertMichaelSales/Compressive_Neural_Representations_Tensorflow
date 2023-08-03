""" Created: 17.07.2023  \\  Updated: 17.07.2023  \\   Author: Robert Sales """

# Note: Run this file from the command line using "pvpython filename.py"

#==============================================================================
# trace generated using paraview version 5.10.1
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

#### import any other modules
import numpy as np
import os, json, glob

#==============================================================================

def ParaviewLoadData(filename,variable):
    
    # create a new 'XML Structured Grid Reader'
    volume = XMLStructuredGridReader(FileName=filename)
        
    # get the range of the pointdata
    volume_range = volume.PointData[variable].GetRange()
    
    return volume,volume_range

#==============================================================================
    
def ParaviewRender(volume,volume_range,savename,render_isom,render_orth,render_zoom):
    
    render_view = GetActiveViewOrCreate('RenderView')

    position_isom = ((1/render_zoom)*np.array([2.1443472477763703, 1.4141082648137087, 2.1443472477763703])).tolist()
    position_orth = ((1/render_zoom)*np.array([1.5721363955790437, 1.1444217043946554, 2.7230201135711063])).tolist()

    #==============================================================================

    # extract the bounds of volume   
    bounds = volume.GetDataInformation().DataInformation.GetBounds()
    
    # create a new 'Transform': translate
    translate = Transform(registrationName='Translate', Input=volume)
    translate.Transform = 'Transform'
    translate.Transform.Translate = [-((bounds[0]+bounds[1])/2),-((bounds[2]+bounds[3])/2),-((bounds[4]+bounds[5])/2)]
    
    # create a new 'Transform': new_scale
    new_scale = Transform(registrationName='Scale', Input=translate)
    new_scale.Transform = 'Transform'
    new_scale.Transform.Scale = [(1/(bounds[0]-bounds[1])),(1/(bounds[2]-bounds[3])),(1/(bounds[4]-bounds[5]))]

    #==============================================================================    
    # Edges (for structured grids)

    # show data in view
    volumeDisplay = Show(new_scale, render_view, 'StructuredGridRepresentation')
    
    # change representation type
    volumeDisplay.SetRepresentationType('Feature Edges')
    
    # properties modified on volumeDisplay
    volumeDisplay.LineWidth = 2.0
    
    # set scalar coloring
    ColorBy(volumeDisplay, ('POINTS', 'field0'))
    
    # rescale color and/or opacity maps used to include current data range
    volumeDisplay.RescaleTransferFunctionToDataRange(True, False)
    
    # show color bar/color legend
    volumeDisplay.SetScalarBarVisibility(render_view, False)
    
    # get color transfer function/color map for 'field0'
    field0LUT = GetColorTransferFunction('field0')
    
    # get opacity transfer function/opacity map for 'field0'
    field0PWF = GetOpacityTransferFunction('field0')
    
    #==========================================================================
    # Volume
    
    # create a new 'Resample To Image'
    resampleToImage = ResampleToImage(registrationName='ResampleToImage', Input=new_scale)
    
    # show data in view
    resampleToImageDisplay = Show(resampleToImage, render_view, 'UniformGridRepresentation')
    
    # trace defaults for the display properties.
    resampleToImageDisplay.Representation = 'Outline'
    
    # show color bar/color legend
    resampleToImageDisplay.SetScalarBarVisibility(render_view, False)
    
    # change representation type
    resampleToImageDisplay.SetRepresentationType('Volume')
    
    # properties modified on field0PWF
    field0PWF.Points = [volume_range[0], 0.25, 0.5, 0.0, volume_range[1], 0.25, 0.5, 0.0]
    
    #==========================================================================
    # Contours
    
    # create a new 'Contour'
    contour = Contour(registrationName='Contour', Input=new_scale)
    
    # properties modified on contour1
    contour.Isosurfaces = np.linspace(volume_range[0],volume_range[1],15)
    
    # create a new 'Extract Surface'
    extractSurface = ExtractSurface(registrationName='ExtractSurface', Input=contour)
    
    # create a new 'Subdivide'
    subdivide = Subdivide(registrationName='Subdivide', Input=extractSurface)
    
    # Properties modified on subdivide
    subdivide.NumberofSubdivisions = 1
    
    # show data in view
    subdivideDisplay = Show(subdivide, render_view, 'GeometryRepresentation')
    
    # change representation type
    subdivideDisplay.SetRepresentationType('Surface With Edges')
    
    # properties modified on subdivideDisplay
    subdivideDisplay.LineWidth = 2.0
    
    # properties modified on subdivideDisplay
    subdivideDisplay.EdgeColor = [1.0, 1.0, 0.0]
    
    # properties modified on subdivideDisplay
    subdivideDisplay.Specular = 0.5
    
    # show color bar/color legend
    subdivideDisplay.SetScalarBarVisibility(render_view, False)
    
    #==========================================================================
    
    # update the view to ensure updated data information
    render_view.Update()
    
    #==========================================================================
    # modify scene properties
    
    # properties modified on render_view
    render_view.UseColorPaletteForBackground = 0
    
    # properties modified on render_view
    render_view.Background = [1.0, 1.0, 1.0]
    
    # properties modified on render_view
    render_view.OrientationAxesVisibility = 0
    
    # properties modified on render_view
    render_view.EnableRayTracing = 1
    
    # properties modified on render_view
    render_view.AmbientSamples = 5
    
    # properties modified on render_view
    render_view.SamplesPerPixel = 5
    
    # properties modified on render_view
    render_view.Shadows = 1
    
    # apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    field0LUT.ApplyPreset('Black, Blue and White', True)
    
    #==========================================================================
    # sort out the lighting   
    
    # turn off the light kit
    render_view.UseLight = 0
    
    # create a new light
    light = AddLight(view=render_view)
    
    # properties modified on light
    light.DiffuseColor = [1.0, 0.0, 0.0]
    
    # properties modified on light
    light.Coords = 'Scene'
    light.Position = position_isom

    #==========================================================================
    # current camera placement for render_view
    
    if render_isom:
        render_view.CameraPosition = position_isom
        render_view.CameraViewUp = [-0.29883623873011983, 0.906307787036650, -0.2988362387301198]
        render_view.CameraParallelScale = 0.6748609001214854
        SaveScreenshot(savename.replace(".png","_isom.png"),render_view,ImageResolution=[2000,2000],TransparentBackground=False,CompressionLevel=0)
    else:pass
        
    if render_orth:
        render_view.CameraPosition = position_orth
        render_view.CameraViewUp = [-0.17101007166283452, 0.9396926207859084, -0.296198132726024]
        render_view.CameraParallelScale = 0.8660254037844386
        SaveScreenshot(savename.replace(".png","_orth.png"),render_view,ImageResolution=[2000,2000],TransparentBackground=False,CompressionLevel=0)
    else:pass
    
    #==========================================================================
    
    Delete(render_view)

    return None
    
#==============================================================================

if __name__=="__main__": 
    
    if (len(sys.argv) == 1):
        
        #======================================================================
        # This block will run in the event that this script is called in an IDE
        
        filepath    = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs/test_profiling"
        render_isom = True
        render_orth = True
        render_zoom = 1.0
          
    else: 

        #======================================================================
        # This block will run in the event that this script is run via terminal
        
        plotting_config = json.loads(sys.argv[1])
        
        filepath    = plotting_config["filepath"]
        render_isom = plotting_config["render_isom"]
        render_orth = plotting_config["render_orth"]
        render_zoom = plotting_config["render_zoom"]
        
    ##    
    
    #==========================================================================
    
    true_filename = os.path.join(filepath,"i_volume.vts")
    true_savename = true_filename.replace(".vts",".png")
    
    pred_filename = os.path.join(filepath,"o_volume.vts")
    pred_savename = pred_filename.replace(".vts",".png")
    
    true_volume, true_volume_range = ParaviewLoadData(filename=true_filename,variable="field0")
    pred_volume, pred_volume_range = ParaviewLoadData(filename=pred_filename,variable="field0")
    
    #==========================================================================

    # Check if the checkpoint file already exists
    if (len(glob.glob(true_filename.replace(".vts","*.png"))) == 0): 
    
        print("\nRendering '{}'".format(true_savename))
        ParaviewRender(volume=true_volume,volume_range=true_volume_range,savename=true_savename,render_isom=render_isom,render_orth=render_orth,render_zoom=render_zoom)
    
    else: 
        
        print("\nFile '{}' already exists: skipping.".format(true_savename))
        
    ##
    
    # Check if the checkpoint file already exists
    if (len(glob.glob(pred_filename.replace(".vts","*.png"))) == 0): 
        
        print("\nRendering '{}'".format(pred_savename))
        ParaviewRender(volume=pred_volume,volume_range=true_volume_range,savename=pred_savename,render_isom=render_isom,render_orth=render_orth,render_zoom=render_zoom)
        
    else: 
        
        print("\nFile '{}' already exists: skipping.".format(pred_savename))
        
    ##
    
else:pass
#==============================================================================




