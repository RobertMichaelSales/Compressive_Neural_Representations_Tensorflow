""" Created: 17.07.2023  \\  Updated: 17.07.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
from pyevtk.hl import gridToVTK

#==============================================================================

def Spectrum2D(matrix):

    fourier_transform = np.fft.fftn(matrix,norm="forward")
    fourier_transform = np.fft.fftshift(fourier_transform)
    
    power_spectrum = np.abs(fourier_transform)
    power_spectrum = np.expand_dims(power_spectrum,axis=-1)
    
    x = np.arange(-(power_spectrum.shape[0]/2),+(power_spectrum.shape[0]/2))
    y = np.arange(-(power_spectrum.shape[1]/2),+(power_spectrum.shape[1]/2))
    z = np.array([0.0])
    
    grid = np.meshgrid(x,y,z,indexing="ij")
    
    return power_spectrum,grid

#==============================================================================

def Spectrum3D(matrix):

    fourier_transform = np.fft.fftn(matrix,norm="forward")
    fourier_transform = np.fft.fftshift(fourier_transform)
    
    power_spectrum = np.abs(fourier_transform)
    
    x = np.arange(-(power_spectrum.shape[0]/2),+(power_spectrum.shape[0]/2))
    y = np.arange(-(power_spectrum.shape[1]/2),+(power_spectrum.shape[1]/2))
    z = np.arange(-(power_spectrum.shape[2]/2),+(power_spectrum.shape[2]/2))
    
    grid = np.meshgrid(x,y,z,indexing="ij")
    
    return power_spectrum,grid

#==============================================================================

def Normalise(matrix,maximum,minimum):
    
    avg = (maximum+minimum)/2
    rng = (maximum-minimum)
    
    matrix = 2*((matrix-avg)/(rng))
    
    return matrix

#==============================================================================

def SaveSpectrum(output_data_path,values,maximum,minimum):
        
    matrix = Normalise(matrix=values.data.squeeze(),maximum=maximum,minimum=minimum)
    
    if (values.data.squeeze().ndim == 2):        
        power_spectrum,grid = Spectrum2D(matrix=matrix)
    else:pass
    
    if (values.data.squeeze().ndim == 3):
        power_spectrum,grid = Spectrum3D(matrix=matrix)
    else:pass   
    
    np.save(output_data_path,power_spectrum)
        
    gridToVTK(output_data_path,*grid,pointData={"power_spectrum":np.ascontiguousarray(power_spectrum)})

    return None

#==============================================================================