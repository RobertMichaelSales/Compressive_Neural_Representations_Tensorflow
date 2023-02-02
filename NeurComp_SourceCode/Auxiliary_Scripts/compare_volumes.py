""" Created: 12.08.2022  \\  Updated: 12.08.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, csv, matplotlib, json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

#==============================================================================
# Set plot aesthetics

plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

params_contour = {
    'image.origin'            : 'lower',
    'image.interpolation'     : 'nearest',
    'image.aspect'            : 'equal',
    'savefig.dpi'             : 600,
    'figure.figsize'          : (18,6.6),
    #
    'axes.axisbelow'          : True,
    'axes.edgecolor'          : 'black',
    'axes.grid'               : True,
    'axes.labelweight'        : 'normal',
    'axes.linewidth'          : 1.25,
    'axes.titlepad'           : 12,
    'axes.titlesize'          : 22,
    #
    'grid.color'              : (0.2500, 0.2500, 0.2500),
    'grid.linestyle'          : 'solid',
    'grid.linewidth'          : 1.00,
    'grid.alpha'              : 0.00,          
    #
    'font.family'             : 'serif',
    'font.size'               : 18,
    'font.style'              : 'normal',
    'font.variant'            : 'small-caps',
    'font.weight'             : 'normal',
    #
    'lines.marker'            : "",
    'lines.markersize'        : 14,
    #
    'xtick.labelsize'         : 16,
    'ytick.labelsize'         : 16,
    'xtick.major.width'       : 1.25,
    'ytick.major.width'       : 1.25,
    'ytick.major.size'        : 7.00,
    'xtick.major.size'        : 7.00,
    'xtick.direction'         : "in",
    'ytick.direction'         : "in",
    'xtick.top'               : True,
    'xtick.bottom'            : True,
    'ytick.left'              : True,
    'ytick.right'             : True,
    'xtick.major.top'         : False,
    'xtick.major.bottom'      : False,
    'ytick.major.left'        : False,
    'ytick.major.right'       : False,
    #
    'legend.title_fontsize'   : 18,
    'legend.fontsize'         : 16,
    'legend.labelspacing'     : 0.3,
    'legend.fancybox'         : False,
    'legend.facecolor'        : 'white',          
    "legend.frameon"          : True,
    "legend.shadow"           : False,
    "legend.framealpha"       : 1.0,
    "legend.edgecolor"        : "black",
    "legend.handlelength"     : 1.0,
    "legend.columnspacing"    : 1.2,
    #
    'savefig.bbox'            : 'tight'}


params_plot = {
    'image.origin'            : 'lower',
    'image.interpolation'     : 'nearest',
    'image.aspect'            : 'equal',
    'savefig.dpi'             : 600,
    'figure.figsize'          : (18,6.6),
    #
    'axes.axisbelow'          : True,
    'axes.edgecolor'          : 'black',
    'axes.grid'               : True,
    'axes.labelweight'        : 'normal',
    'axes.linewidth'          : 1.25,
    'axes.titlepad'           : 12,
    'axes.titlesize'          : 22,
    #
    'grid.color'              : (0.2500, 0.2500, 0.2500),
    'grid.linestyle'          : 'solid',
    'grid.linewidth'          : 1.00,
    'grid.alpha'              : 0.60,          
    #
    'font.family'             : 'serif',
    'font.size'               : 18,
    'font.style'              : 'normal',
    'font.variant'            : 'small-caps',
    'font.weight'             : 'normal',
    #
    'lines.marker'            : "",
    'lines.markersize'        : 14,
    #
    'xtick.labelsize'         : 16,
    'ytick.labelsize'         : 16,
    'xtick.major.width'       : 1.25,
    'ytick.major.width'       : 1.25,
    'ytick.major.size'        : 7.00,
    'xtick.major.size'        : 7.00,
    'xtick.direction'         : "in",
    'ytick.direction'         : "in",
    'xtick.top'               : True,
    'xtick.bottom'            : True,
    'ytick.left'              : True,
    'ytick.right'             : True,
    'xtick.major.top'         : True,
    'xtick.major.bottom'      : True,
    'ytick.major.left'        : True,
    'ytick.major.right'       : True,
    #
    'legend.title_fontsize'   : 18,
    'legend.fontsize'         : 16,
    'legend.labelspacing'     : 0.5,
    'legend.fancybox'         : False,
    'legend.facecolor'        : 'white',          
    "legend.frameon"          : True,
    "legend.shadow"           : False,
    "legend.framealpha"       : 1.0,
    "legend.edgecolor"        : "black",
    "legend.handlelength"     : 1.8,
    "legend.columnspacing"    : 1.2,
    #
    'savefig.bbox'            : 'tight'}

# Matlab Default Colours
colours = {"blue"   : (0.0000, 0.4470, 0.7410),
           "orange" : (0.8500, 0.3250, 0.0980),
           "yellow" : (0.9290, 0.6940, 0.1250),
           "purple" : (0.4940, 0.1840, 0.5560),
           "green"  : (0.4660, 0.6740, 0.1880),
           "lblue"  : (0.3010, 0.7450, 0.9330),
           "red"    : (0.6350, 0.0780, 0.1840),
           "black"  : (0.0000, 0.0000, 0.0000),
           "dgrey"  : (0.2500, 0.2500, 0.2500),
           "grey"   : (0.5000, 0.5000, 0.5000),
           "lgrey"  : (0.7500, 0.7500, 0.7500)}

# Matplotlib blues colourmap
c_map = cm.get_cmap("Blues",20)

#==============================================================================

def CompareContours(with_grad=True):
    
    matplotlib.rcParams.update(params_contour)   
    
    #==========================================================================
    # Set filepaths and directories
    
    # Set base directory
    base_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"
    
    # Set input filepath
    input_data_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/test_vol.npy"
    
    # Set current directory
    plot_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/plots"
        
    # Get a sorted list of all the experiements (subdirectories)
    experiment_directories = sorted(os.listdir(base_directory),key = lambda x: int(x.split("_")[-1]))
        
    #==========================================================================
    # Iterate through all the experiments
    for experiment_directory in experiment_directories:
        
        # Set the current experiment directory
        experiment_directory = os.path.join(base_directory,experiment_directory)
        
        # Set the config file, output file, and training data file paths
        config_filepath = os.path.join(experiment_directory,"network_configuration.json")
        output_data_filepath = os.path.join(experiment_directory,"output_volume.npy")
        training_data_filepath = os.path.join(experiment_directory,"training_data.json")
        
        # Load the config dictionary and training data from .json files
        with open(config_filepath) as config_file: config = json.load(config_file)
        with open(training_data_filepath) as training_data_file: training_data = json.load(training_data_file)
        
        #==========================================================================
        # Data for plotting
        
        # Load the input volume from .npy file
        input_data  = np.load(input_data_filepath)
        
        # Load the output volume from .npy file
        output_data = np.load(output_data_filepath)
        
        # Make 2D cuts from 3D data
        axis,cut = 'z',75
        varx_i,vary_i,vals_i = MakeSliceFrom3D(data=input_data, cut=cut,axis=axis)
        varx_o,vary_o,vals_o = MakeSliceFrom3D(data=output_data,cut=cut,axis=axis)
        
        # Normalise values
        vals_i = Normalise(vals_i)
        vals_o = Normalise(vals_o)
        
        # Compute gradients from values and coordinates
        grad_i = np.sum(np.gradient(vals_i,edge_order=1),axis=0)
        grad_o = np.sum(np.gradient(vals_o,edge_order=1),axis=0)
        
        # Compute gradient components
        grad_x_i,grad_y_i = np.gradient(vals_i,edge_order=1)[0],np.gradient(vals_i,edge_order=1)[1]
        grad_x_o,grad_y_o = np.gradient(vals_o,edge_order=1)[0],np.gradient(vals_o,edge_order=1)[1]
        
        # Normalise gradients
        grad_i = Normalise(grad_i)
        grad_o = Normalise(grad_o)
        
        # Calculate errors
        varx_e,vary_e,vals_e,grad_e = varx_i,vary_i,abs(vals_i-vals_o),abs(grad_i-grad_o)
        
        #==========================================================================
        # Matplotlib code 
        
        # Make figure and plot axes
        fig, ax = plt.subplots(1,3,constrained_layout=True)
        
        labels = ["Normalised Input","Normalised Prediction","Relative Error (Magnified)"]
        
        if with_grad: 
            alpha = 0.6
        else:
            alpha = 1.0
        
        
        if with_grad:
            space,scale = 15,0.8
            ax[0].quiver(varx_i[space::space,space::space],vary_i[space::space,space::space],grad_x_i[space::space,space::space],grad_y_i[space::space,space::space],pivot="mid",scale=scale,alpha=1.0,width=0.01)
        ax[0].set_title(labels[0],horizontalalignment="left",x=0)
        
        # Plot output
        vmin,vmax,layers = 0.0,1.0,100
        ax[1].contourf(varx_o,vary_o,vals_o,np.linspace(vmin,vmax,layers),vmin=vmin,vmax=vmax,cmap=plt.get_cmap('Blues'),extend="both",alpha=alpha)
        if with_grad:
            space,scale = 15,0.8
            ax[1].quiver(varx_o[space::space,space::space],vary_o[space::space,space::space],grad_x_o[space::space,space::space],grad_y_o[space::space,space::space],pivot="mid",scale=scale,alpha=1.0,width=0.01)
        ax[1].set_title(labels[1],horizontalalignment="left",x=0)
            
        # Plot errors
        vmin,vmax,layers = 0.0,0.1,50
        ax[2].contourf(varx_e,vary_e,vals_e,np.linspace(vmin,vmax,layers),vmin=vmin,vmax=vmax,cmap=plt.get_cmap('Blues'),extend="both")
        ax[2].set_title(labels[2],horizontalalignment="left",x=0)
        
        if with_grad:
            title = r"Contour Slices With Gradient: Actual Compression Ratio = {:.2f} ".format(config["actual_compression_ratio"])
        else:   
            title = r"Contour Slices: Actual Compression Ratio = {:.2f} ".format(config["actual_compression_ratio"])
        plt.suptitle(title,weight="heavy",horizontalalignment="left",x=0.0025)
        
        compression_ratio = config["target_compression_ratio"]
        
        if with_grad:
            savename = os.path.join(plot_filepath,"contours_grad_"+str(compression_ratio)+".png")
        else:
            savename = os.path.join(plot_filepath,"contours_"+str(compression_ratio)+".png")
        
        plt.savefig(savename)
        plt.show()
        
        #======================================================================
    
    return None

#==============================================================================

def PlotVersusCompression():
    
    matplotlib.rcParams.update(params_plot)
    
    #==========================================================================
    # Set filepaths and directories
    
    # Set base directory
    base_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"
    
    # Set input filepath
    input_data_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/test_vol.npy"
    
    # Set current directory
    plot_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/plots"
    
    # Get a sorted list of all the experiements (subdirectories)
    experiment_directories = sorted(os.listdir(base_directory),key = lambda x: int(x.split("_")[-1]))
        
    # Make figure and plot axes
    fig, ax = plt.subplots(1,2,constrained_layout=True)
    
    # Make empty lists to add data to
    compression_ratio = []
    training_duration = []
    mean_squared_loss = []    
    
    #==========================================================================
    # Iterate through all the experiments
    for experiment_directory in experiment_directories:
        
        # Set the current experiment directory
        experiment_directory = os.path.join(base_directory,experiment_directory)
        
        # Set the config file, output file, and training data file paths
        config_filepath = os.path.join(experiment_directory,"network_configuration.json")
        output_data_filepath = os.path.join(experiment_directory,"output_volume.npy")
        training_data_filepath = os.path.join(experiment_directory,"training_data.json")
        
        # Load the config dictionary and training data from .json files
        with open(config_filepath) as config_file: config = json.load(config_file)
        with open(training_data_filepath) as training_data_file: training_data = json.load(training_data_file)
        
        #==========================================================================
        # Data for plotting
        
        compression_ratio.append(config["actual_compression_ratio"])
        training_duration.append(np.sum(training_data["time"]))   
        mean_squared_loss.append(training_data["loss"][-1])
        
        #======================================================================
        # Matplotlib code
        
        ax[1].scatter(compression_ratio,training_duration,marker="s",color=colours["blue"])
        ax[1].plot(compression_ratio,training_duration,linestyle="dotted",color=colours["blue"])
        
        ax[0].scatter(compression_ratio,mean_squared_loss,marker="s",color=colours["blue"])
        ax[0].plot(compression_ratio,mean_squared_loss,linestyle="dotted",color=colours["blue"])
        
    # Set x- and y-ticks
    ax[1].set_xticks(np.linspace(0,800,5),minor=False)
    ax[1].set_yticks(np.linspace(170,240,8),minor=False)
    
    # Set x- and y-ticks
    ax[0].set_xticks(np.linspace(0,800,5),minor=False)
    ax[0].set_yticks(np.linspace(0,0.5,6),minor=False)
    
    # Set x- and y-limits
    ax[1].set_ylim(180,230)
    
    # Set x- and y-limits
    ax[0].set_ylim(0.000001,0.1)
    
    # Set x- and y- scale
    ax[1].set_xscale('log')
    
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    # Set x- and y-labels
    ax[1].set_xlabel(r"Compression Ratio")
    ax[1].set_ylabel(r"Total Training Time / s")
    
    # Set x- and y-labels
    ax[0].set_xlabel(r"Compression Ratio")
    ax[0].set_ylabel(r"Mean Squared Error")
    
    # Set the figure title
    ax[1].set_title("Training Time vs. Compression Ratio")
    ax[0].set_title("Training Loss vs. Compression Ratio")

    savename = os.path.join(plot_filepath,"compression_comparison.png")
    plt.savefig(savename)
    plt.show()
        
    #==========================================================================
            
    return None

#==============================================================================

def PlotTrainingMetrics():
    
    matplotlib.rcParams.update(params_plot)
    
    blues = c_map(np.linspace(1.0,0.3,12))
    
    #==========================================================================
    # Set filepaths and directories
    
    # Set base directory
    base_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"
    
    # Set input filepath
    input_data_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/test_vol.npy"
    
    # Set current directory
    plot_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/plots"
    
    # Get a sorted list of all the experiements (subdirectories)
    experiment_directories = sorted(os.listdir(base_directory),key = lambda x: int(x.split("_")[-1]))
        
    # Make figure and plot axes
    fig, ax = plt.subplots(1,2,constrained_layout=True)
    
    #==========================================================================
    # Iterate through all the experiments
    for index,experiment_directory in enumerate(experiment_directories):
        
        if index not in [0,1,3,4,9,11]: continue
        
        # Set the current experiment directory
        experiment_directory = os.path.join(base_directory,experiment_directory)
        
        # Set the config file, output file, and training data file paths
        config_filepath = os.path.join(experiment_directory,"network_configuration.json")
        output_data_filepath = os.path.join(experiment_directory,"output_volume.npy")
        training_data_filepath = os.path.join(experiment_directory,"training_data.json")
        
        # Load the config dictionary and training data from .json files
        with open(config_filepath) as config_file: config = json.load(config_file)
        with open(training_data_filepath) as training_data_file: training_data = json.load(training_data_file)
        
        #==========================================================================
        # Data for plotting
        
        learning_rate = training_data["learning_rate"]
        loss = training_data["loss"]
        epoch = training_data["epoch"]
        label = config["target_compression_ratio"]
        
        #======================================================================
        # Matplotlib code
        
        ax[0].plot(epoch,loss,linestyle="dotted",color=blues[index],marker="s",label=label)
        
    ax[1].scatter(epoch,learning_rate,marker="s",color=colours["blue"])
    ax[1].plot(epoch,learning_rate,linestyle="dotted",color=colours["blue"])
        
    # Set x- and y-ticks
    ax[0].set_xticks(np.linspace(0,30,7),minor=False)
    
    ax[1].set_xticks(np.linspace(0,30,7),minor=False)
    
    # Set x- and y-limits
    ax[0].set_xlim(-1,31)
    ax[0].set_ylim(5e-6,2e-1)
    
    # Set x- and y-limits
    ax[1].set_xlim(-1,31)    
    ax[1].set_ylim(6e-6,1.7e-2)
    
    # Set x- and y- scale
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    # Set x- and y-labels
    ax[0].set_xlabel(r"Epoch No. / Pass")
    ax[0].set_ylabel(r"Mean Squared Loss")
    
    # Set x- and y-labels
    ax[1].set_xlabel(r"Epoch No. / Pass")
    ax[1].set_ylabel(r"Learning Rate")
    
    # Set the figure title
    ax[0].set_title("Training Loss vs. Training Epoch")
    ax[1].set_title("Learning Rate vs. Training Epoch")
    
    ax[0].legend(title="Target Compression Ratios",loc="upper right",ncols=3)
        
    savename = os.path.join(plot_filepath,"compression_training.png")
    plt.savefig(savename)
    plt.show()
        
    #==========================================================================
            
    return None


#==============================================================================
    
def MakeSliceFrom3D(data,cut,axis='x'):
    
    if axis == 'x': var1,var2,vals = data[cut,:,:,1],data[cut,:,:,2],data[cut,:,:,3]
    
    if axis == 'y': var1,var2,vals = data[:,cut,:,0],data[:,cut,:,2],data[:,cut,:,3]
    
    if axis == 'z': var1,var2,vals = data[:,:,cut,0],data[:,:,cut,1],data[:,:,cut,3]
        
    return var1,var2,vals

#==============================================================================
    
def Normalise(data):
    
    data = (data-((data.max()+data.min())/2))/(abs(data.max()-data.min()))+0.5
        
    return data

#==============================================================================

CompareContours(with_grad=False)
CompareContours(with_grad=True )
PlotVersusCompression()
PlotTrainingMetrics()