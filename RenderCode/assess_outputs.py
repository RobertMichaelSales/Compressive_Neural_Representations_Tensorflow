""" Created: 22.08.2023  \\  Updated: 22.08.2023  \\   Author: Robert Sales """

#==============================================================================

import numpy as np
import os, json, glob, vtk, sys, skimage

#==============================================================================

def AssessRenders(experiment_filepath):
    
    #==========================================================================
    
    # Set output filepath
    render_filepath = os.path.join(experiment_filepath,"renders")
    
    if os.path.exists(render_filepath): 
        if (not len(os.listdir(render_filepath))): 
            render_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/RenderCode/renders"
        ##
    ##
    
    #==========================================================================
    ## Assess surface renderings
    
    true_surface_img_filepaths = sorted(glob.glob(os.path.join(render_filepath,"temp_true_surface_*.png")))
    pred_surface_img_filepaths = sorted(glob.glob(os.path.join(render_filepath,"temp_pred_surface_*.png")))
    
    surface_nrmss, surface_ssims = [], []
    
    for true_img_filepath,pred_img_filepath in zip(true_surface_img_filepaths,pred_surface_img_filepaths):
                
        true_img = skimage.io.imread(true_img_filepath)
        pred_img = skimage.io.imread(pred_img_filepath)
        
        nrms = skimage.metrics.normalized_root_mse(true_img,pred_img,normalization="euclidean")
        ssim = skimage.metrics.structural_similarity(true_img,pred_img,channel_axis=2,data_range=np.ptp(true_img),full=False)
        
        surface_nrmss.append(nrms)
        surface_ssims.append(ssim)
        
    ##
    
    #==========================================================================
    ## Assess contour renderings
    
    true_contour_img_filepaths = sorted(glob.glob(os.path.join(render_filepath,"temp_true_contour_*.png")))
    pred_contour_img_filepaths = sorted(glob.glob(os.path.join(render_filepath,"temp_pred_contour_*.png")))
    
    contour_nrmss, contour_ssims = [], []
    
    for true_img_filepath,pred_img_filepath in zip(true_contour_img_filepaths,pred_contour_img_filepaths):
                
        true_img = skimage.io.imread(true_img_filepath)
        pred_img = skimage.io.imread(pred_img_filepath)
        
        nrms = skimage.metrics.normalized_root_mse(true_img,pred_img,normalization="euclidean")
        ssim = skimage.metrics.structural_similarity(true_img,pred_img,channel_axis=2,data_range=np.ptp(true_img),full=False)
        
        contour_nrmss.append(nrms)
        contour_ssims.append(ssim)
        
    ##
    
    #==========================================================================
    ## Compute average statistics
    
    average_surface_nrms = np.average(surface_nrmss)
    print("Average surface NRMS: {:.6f}".format(average_surface_nrms))
    
    average_surface_ssim = np.average(surface_ssims)
    print("Average surface SSIM: {:.6f}".format(average_surface_ssim))
    
    average_contour_nrms = np.average(contour_nrmss)
    print("Average contour NRMS: {:.6f}".format(average_contour_nrms))
    
    average_contour_ssim = np.average(contour_ssims)
    print("Average contour SSIM: {:.6f}".format(average_contour_ssim))
    
    #==========================================================================
    # Save to experiment filepath
      
    training_metadata_filepath = os.path.join(experiment_filepath,"training_metadata.json")
    if not os.path.isfile(training_metadata_filepath): raise FileExistsError("File '{:}' does not exist!".format(training_metadata_filepath))
    with open(training_metadata_filepath,"r") as file: training_metadata = json.load(file)
    
    training_metadata["average_surface_nrms"] = [average_surface_nrms]
    training_metadata["average_surface_ssim"] = [average_surface_ssim]
    training_metadata["average_contour_nrms"] = [average_contour_nrms]
    training_metadata["average_contour_ssim"] = [average_contour_ssim]

    with open(training_metadata_filepath,"w") as file: json.dump(training_metadata,file,indent=4,sort_keys=False)
    
    #==========================================================================
    ## Remove temporary image files
    
    render_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/RenderCode/renders"
    all_img_filepaths = sorted(glob.glob(os.path.join(render_filepath,"*.png")))
    for img_filepath in all_img_filepaths: os.remove(img_filepath)
    
    #==========================================================================
    
    return None
    
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
        
    print("-"*80,"\nASSESSING IMAGES:\n") 
    
    print("Assessing Outputs: '{}'\n".format(experiment_filepath))
    
    AssessRenders(experiment_filepath)
    
    print("-"*80,"\n")

else: pass

#==============================================================================