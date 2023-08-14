""" Created: 17.07.2023  \\  Updated: 17.07.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
from pyevtk.hl import gridToVTK

#==============================================================================
# Normalise the input matrix into the range [-1,+1]

def Normalise(matrix,maximum,minimum):
    
    avg = (maximum+minimum)/2
    rng = (maximum-minimum)
    
    matrix = 2*((matrix-avg)/(rng))
    
    return matrix

#==============================================================================
# Get a quasi-2D mesh of the power spectrum (FOR SaveSpectrum())

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
# Get a fully-2D mesh of the power spectrum (FOR SaveSpectrum())

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
# Save the power spectrum as .npy and .vts files 

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
# Get the power component of the frequency spectrum

def SpectralPower(matrix):

    fourier_transform = np.fft.fftn(matrix,norm="forward")
    fourier_transform = np.fft.fftshift(fourier_transform)
    
    spectral_power = np.abs(fourier_transform).astype("float32")
    
    return spectral_power

#==============================================================================
# Get the phase component of the frequency spectrum

def SpectralPhase(matrix):

    fourier_transform = np.fft.fftn(matrix,norm="forward")
    fourier_transform = np.fft.fftshift(fourier_transform)
    
    spectral_phase = np.angle(fourier_transform).astype("float32")
    
    return spectral_phase

#==============================================================================
# Calculate the radially averaged power spectrum in 2D

def RadiallyAveragedPowerSpectrum2D(matrix,norm_max,norm_min):
    
    # Normalise the matrix according to user-stated maximum/minimum
    matrix = Normalise(matrix,norm_max=norm_max,norm_min=norm_min)

    # Determine the spectral power and phase
    spectral_power = SpectralPower(matrix=matrix)
    spectral_phase = SpectralPhase(matrix=matrix)

    # Make a matrix showing the radii of each entry from the origin 
    radii = np.fromfunction(lambda i,j: np.sqrt((i-(matrix.shape[0]/2))**2+(j-(matrix.shape[1]/2))**2),matrix.shape,dtype=float).astype(int)

    # Count the number of each integer radius: i.e. 1 element for 0
    radial_bins = np.bincount(radii.ravel())[:int(min(matrix.shape)/2)]

    # Sum the spectral powers for each integer radius going outward
    radial_sums = np.bincount(radii.ravel(),weights=spectral_power.ravel())[:int(min(matrix.shape)/2)]

    # Make an array of the frequencies in units of 1/pixel (PxErtz)
    frequencies = np.arange(start=0,stop=int(len(radial_bins)))/(2*len(radial_bins))
    
    return frequencies,radial_sums,spectral_power,spectral_phase

#==============================================================================
# Calculate the radially averaged power spectrum in 3D

def RadiallyAveragedPowerSpectrum3D(matrix,norm_max,norm_min):
    
    # Normalise the matrix according to user-stated maximum/minimum
    matrix = Normalise(matrix,norm_max=norm_max,norm_min=norm_min)

    # Determine the spectral power and phase
    spectral_power = SpectralPower(matrix=matrix)
    spectral_phase = SpectralPhase(matrix=matrix)

    # Make a matrix showing the radii of each entry from the origin 
    radii = np.fromfunction(lambda i,j,k: np.sqrt((i-(matrix.shape[0]/2))**2+(j-(matrix.shape[1]/2))**2+(k-(matrix.shape[2]/2))**2),matrix.shape,dtype=float).astype(int)

    # Count the number of each integer radius: i.e. 1 element for 0
    radial_bins = np.bincount(radii.ravel())[:int(min(matrix.shape)/2)]

    # Sum the spectral powers for each integer radius going outward
    radial_sums = np.bincount(radii.ravel(),weights=spectral_power.ravel())[:int(min(matrix.shape)/2)]

    # Make an array of the frequencies in units of 1/pixel (PxErtz)
    frequencies = np.arange(start=0,stop=int(len(radial_bins)))/(2*len(radial_bins))
    
    return frequencies,radial_sums,spectral_power,spectral_phase

#==============================================================================

def GetComplexity(values,maximum,minimum):
    
    matrix = Normalise(matrix=values.data.squeeze(),maximum=maximum,minimum=minimum)
    
    if (values.data.squeeze().ndim == 2):        
        frequencies,radial_sums,spectral_power,spectral_phase = RadiallyAveragedPowerSpectrum2D(matrix,norm_max=maximum,norm_min=minimum)
    else:pass
    
    if (values.data.squeeze().ndim == 3):
        frequencies,radial_sums,spectral_power,spectral_phase = RadiallyAveragedPowerSpectrum3D(matrix,norm_max=maximum,norm_min=minimum)
    else:pass   

    raps_avg_frequency = np.sum(frequencies*radial_sums)/np.sum(radial_sums)
    
    complexity_density = raps_avg_frequency / (matrix.size ** (1/matrix.ndim))

    return raps_avg_frequency,complexity_density

#==============================================================================