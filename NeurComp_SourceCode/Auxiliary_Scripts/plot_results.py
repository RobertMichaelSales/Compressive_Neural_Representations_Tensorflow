""" Created: 18.01.2022  \\  Updated: 18.01.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, csv, matplotlib, json, math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

#==============================================================================
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

markers = ["s","o","*"]
lstyles = ["solid","dashed","dotted",]


# Matplotlib blues colourmap
c_map = cm.get_cmap("Blues",20)

#==============================================================================

def PlotVersusBatchSize(save=False):
    
    plt.style.use("plot.mplstyle")  
    
    params_plot = {'text.latex.preamble': [r'\usepackage{amsmath}',r'\usepackage{amssymb}'],
                   'axes.grid': True}
    
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
    for index1,dataset in enumerate(["cube","passage"]):
    
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
        num_of_parameters = []
        
        training_duration = []
        mean_squared_loss = []
        peak_signal_noise = []
                
        #======================================================================
        # Iterate through all the experiments
        for index2,experiment_directory in enumerate(experiment_directories):
                        
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
            
        #======================================================================
        # Specific data wrangling      
        
        duration_of_epoch = np.array(training_duration)/30
                    
        #======================================================================
        # Matplotlib code
            
        ax1.plot(batch_fraction,mean_squared_loss,linestyle=lstyles[index1],marker=markers[index1],color=colours["blue"],label=dataset.capitalize())
        ax2.plot(batch_fraction,duration_of_epoch,linestyle=lstyles[index1],marker=markers[index1],color=colours["red"])

    #==========================================================================
    # Matplotlib code
    
    # Vertical rectangle
    ax1.axvspan(0.00025,0.00075, alpha=0.25, color='gray',hatch=None)
        
    # # Set x- and y- scale
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    ax2.set_yscale('linear')
    
    # # Set x- and y-limits
    ax1.set_xlim(-0.0000,+0.0026)
    ax1.set_ylim(-0.0000,+0.0080)
    ax2.set_ylim(-0.0000,+16.000)
    
    # Set x- and y-ticks
    # ax1.set_xticks(,minor=False)
    ax1.set_yticks(np.linspace(0,0.008,5),minor=False)
    ax2.set_yticks(np.linspace(0,16.00,5),minor=False)

    # Set x- and y-labels
    ax1.set_xlabel(r"Batch Fraction (Batch Size / Volume Size)")
    ax1.set_ylabel(r"Mean-Squared Error $(MSE)$",color=colours["blue"])
    ax2.set_ylabel(r"Per-Epoch Training Time / $s$",color=colours["red"])

    # Set the figure title
    ax1.set_title(r"Batch Fraction vs. Mean-Squared Error vs. Training Duration")
    
    # Set the figure legend
    ax1.legend(title=r"Input Volume",loc="upper right",ncol=2,labelcolor="black")


    if save:
        savename = os.path.join(plot_directory,"batch_size.png")
        plt.savefig(savename)
    else: pass

    plt.show()
        
    #==========================================================================
            
    return None

#==============================================================================

def PlotVersusCompression(save=False):
    
    plt.style.use("plot.mplstyle")  
    
    params_plot = {'text.latex.preamble': [r'\usepackage{amsmath}',r'\usepackage{amssymb}'],
                   'axes.grid': True}
    
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
    
    #==========================================================================
    # Iterate through each data set
    for index1,dataset in enumerate(["cube"]):
    
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
        num_of_parameters = []
        
        training_duration = []
        mean_squared_loss = []
        peak_signal_noise = []
                
        #======================================================================
        # Iterate through all the experiments
        for index2,experiment_directory in enumerate(experiment_directories):            

            if not (index2 % 2): continue

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
            
            #==================================================================
            # Matplotlib code
            
            ax1.plot(epoch,error,linestyle="solid",marker=None,color=colours["blue"],label=dataset.capitalize())

        #======================================================================
        # Specific data wrangling      
       
        #======================================================================
        # Matplotlib code
            
    #==========================================================================
    # Matplotlib code
    
    # # Set x- and y- scale
    # ax1.set_xscale('linear')
    ax1.set_yscale('log')
    # ax2.set_yscale('linear')
    
    # Set x- and y-limits
    ax1.set_xlim(0,30)
    ax1.set_ylim(0.0001,0.1)
    
    # Set x- and y-ticks
    # ax1.set_xticks(,minor=False)
    # ax1.set_yticks(np.linspace(0,0.008,5),minor=False)

    # Set x- and y-labels
    ax1.set_xlabel(r"Training Epoch")
    ax1.set_ylabel(r"Mean-Squared Error $(MSE)$")

    # Set the figure title
    # ax1.set_title(r"Batch Fraction vs. Mean-Squared Error vs. Training Duration")
    
    # Set the figure legend
    # ax1.legend(title=r"Input Volume",loc="upper right",ncol=2,labelcolor="black")


    if save:
        savename = os.path.join(plot_directory,"compression.png")
        plt.savefig(savename)
    else: pass

    plt.show()
        
    #==========================================================================
            
    return None

#==============================================================================

# PlotVersusBatchSize()
PlotVersusCompression()