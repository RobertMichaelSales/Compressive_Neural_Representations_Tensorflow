""" Created: 18.01.2022  \\  Updated: 18.01.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, csv, matplotlib, json, math
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
    'figure.figsize'          : (14,6.6),
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

def PlotVersusBatchSize(save=False):
    
    matplotlib.rcParams.update(params_plot)   
    
    #==========================================================================
    # Set filepaths and directories
    
    # Set base directory
    base_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"
    
    # Set current directory
    plot_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/plots"
    
    #======================================================================
    # Make figure and plot axes
    fig, ax1 = plt.subplots(1,1,constrained_layout=True)
    
    ax2 = ax1.twinx()
    
    #==========================================================================
    # Iterate through each data set
    for dataset in ["cube","passage"]:
    
        # Set input filepath
        input_volume_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/" + dataset + ".npy"
        
        # Get a sorted list of all the experiements directories
        experiment_directories = [x for x in os.listdir(base_directory) if dataset in x]
        experiment_directories.sort(key = lambda x: float(x.split("_")[-2]))
                    
        #======================================================================
        # Set up storage for results
        
        # Record once per experiment
        target_compression_ratio = []
        actual_compression_ratio = []
        initial_lr = []
        batch_size = []
        batch_fraction = []
        
        training_duration = []
        mean_squared_loss = []
        peak_signal_noise = []
                
        #======================================================================
        # Iterate through all the experiments
        for experiment_directory in experiment_directories:
            
            print(experiment_directory)
            
            # Set the current experiment directory
            experiment_directory = os.path.join(base_directory,experiment_directory)
            
            # Set the config, output, and training data filepaths
            configuration_filepath = os.path.join(experiment_directory,"configuration.json")
            output_volume_filepath = os.path.join(experiment_directory,"output_volume.npy")
            training_data_filepath = os.path.join(experiment_directory,"training_data.json")
            
            # Load the config and training data 
            with open(configuration_filepath) as configuration_file: configuration = json.load(configuration_file)
            with open(training_data_filepath) as training_data_file: training_data = json.load(training_data_file)
            
            # Load the input and output volumes
            input_data = np.load(input_volume_filepath)
            output_data = np.load(output_volume_filepath)
            
            #==================================================================
            # Extract data for plotting
            
            # Record once per experiment
            target_compression_ratio.append(configuration["target_compression_ratio"])
            actual_compression_ratio.append(configuration["actual_compression_ratio"])
            initial_lr.append(configuration["initial_lr"])
            batch_size.append(configuration["batch_size"])
            batch_fraction.append(configuration["batch_fraction"])
            
            # Record every training epoch
            learning_rate = training_data["learning_rate"]
            epoch = training_data["epoch"]
            error = training_data["error"]
            time = training_data["time"]
            
            # Derived data from training
            training_duration.append(np.sum(time))
            mean_squared_loss.append(error[-1])   
            peak_signal_noise.append(training_data["psnr"][-1])
            
            #==================================================================
            # Specific data wrangling
            
            mean_squared_loss_percentage = 100 * np.array(mean_squared_loss)
                    
        #======================================================================
        # Matplotlib code
            
        ax1.scatter(batch_fraction,mean_squared_loss_percentage,marker="s",color=colours["blue"])
        ax1.plot(batch_fraction,mean_squared_loss_percentage,linestyle="solid",color=colours["blue"])
        
        ax2.scatter(batch_fraction,training_duration,marker="s",color=colours["red"])
        ax2.plot(batch_fraction,training_duration,linestyle="solid",color=colours["red"])

    # Set x- and y-ticks
    # ax1.set_xticks(,minor=False)
    # ax1.set_yticks(,minor=False)
    # ax2.set_yticks(,minor=False)
    
    # # Set x- and y-limits
    # ax1.set_xlim()
    # ax1.set_ylim()
    # ax2.set_ylim()
    
    # # Set x- and y- scale
    ax1.set_xscale('linear')#'log',base=10)
    ax1.set_yscale('linear')
    ax2.set_yscale('linear')

    # Set x- and y-labels
    ax1.set_xlabel(r"Batch Size")
    ax1.set_ylabel(r"Mean-Squared Error / %",color=colours["blue"])
    ax2.set_ylabel(r"Training Duration / seconds",color=colours["red"])

    # Set the figure title
    ax1.set_title("Batch Size vs. Mean-Squared Error vs. Training Duration")

    if save:
        savename = os.path.join(plot_directory,"")
        plt.savefig(savename)
    else: pass

    plt.show()
        
    #==========================================================================
            
    return None

#==============================================================================

# PlotMSEandTimevsBatchSize()
PlotVersusBatchSize()