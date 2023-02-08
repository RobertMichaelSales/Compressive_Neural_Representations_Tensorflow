""" Created: 18.01.2022  \\  Updated: 18.01.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, csv, matplotlib, json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

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
blues = cm.get_cmap("Blues",20)

#==============================================================================

def PlotErrorDurationBatch(save=False):
    
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
        experiment_directories = []
        
        for folder_name in os.listdir(base_directory):
            if (dataset in folder_name) and ("batchfraction" in folder_name):
                experiment_directories.append(folder_name)
                
                
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
            
        ax1.plot(batch_fraction,mean_squared_loss,linestyle=lstyles[index1],marker=markers[index1],color="blue",label=dataset.capitalize())
        ax2.plot(batch_fraction,duration_of_epoch,linestyle=lstyles[index1],marker=markers[index1],color="red")

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
    ax1.set_ylabel(r"Mean-Squared Error $(MSE)$",color="blue")
    ax2.set_ylabel(r"Per-Epoch Training Time / $s$",color="red")

    # Set the figure title
    # ax1.set_title(r"Batch Fraction vs. Mean-Squared Error vs. Training Duration")
    
    # Set the figure legend
    leg = ax1.legend(title=r"Input Volume",loc="upper right",ncol=2,labelcolor="black")
    leg.legendHandles[0].set_color("dimgray")
    leg.legendHandles[1].set_color("dimgray")

    if save:
        savename = os.path.join(plot_directory,"error_vs_time_vs_batchfraction.png")
        plt.savefig(savename)
    else: pass

    plt.show()
        
    #==========================================================================
            
    return None

#==============================================================================

def PlotConvergenceBatch(save=False):
    
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
    
    #==========================================================================
    # Iterate through each data set
    for index1,dataset in enumerate(["passage","cube"]):
        
        #======================================================================
        # Make figure and plot axes
        fig, ax1 = plt.subplots(1,1,constrained_layout=True)
        
        #======================================================================
    
        # Set input filepath
        input_volume_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/" + dataset + ".npy"
        
        # Get a sorted list of all the experiements directories
        experiment_directories = []
        
        for folder_name in os.listdir(base_directory):
            if (dataset in folder_name) and ("batchfraction" in folder_name):
                experiment_directories.append(folder_name)
                
        experiment_directories.sort(key = lambda x: float(x.split("_")[-2]))
        
        # Get rid of the last data point so the graph fits better
        experiment_directories.pop(0)
                    
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
            
            epoch = np.array(epoch) + 0.5
            
            #==================================================================
            # Matplotlib code

            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["blue","indigo","red"])

            color = cmap(np.linspace(0.0,1.0,len(experiment_directories))[index2])
            
            label = "{:.1f}".format(batch_fraction[-1]/1e-4)

            ax1.plot(epoch,error,linestyle="solid",marker="None",color=color,label=label)
            
        ##

        #======================================================================
        # Specific data wrangling      
       
        #======================================================================
        # Matplotlib code
        
        # ax1.annotate(r'Increasing Batch Fraction: [0.0001 $\to$ 0.0025]', color='black',
        #     xy=(0.25,0.10), xycoords='axes fraction',
        #     xytext=(0.67, 0.80), textcoords='axes fraction',
        #     arrowprops=dict(arrowstyle="<|-",connectionstyle="arc3,rad=-0.3",linewidth=1.2,color='black'),
        #     horizontalalignment='center', verticalalignment='center')
        
        # # Set x- and y- scale
        # ax1.set_xscale('linear')
        ax1.set_yscale('log')
        # ax2.set_yscale('linear')
        
        # Set x- and y-limits
        if dataset == "cube":
            ax1.set_xlim(0,30)
            ax1.set_ylim(0.0001,0.1)
        if dataset == "passage":
            ax1.set_xlim(0,30)
            ax1.set_ylim(0.0010,0.1)
        
        # Set x- and y-ticks
        # ax1.set_xticks(,minor=False)
        # ax1.set_yticks(np.linspace(0,0.008,5),minor=False)

        # Set x- and y-labels
        ax1.set_xlabel(r"Training Epoch")
        ax1.set_ylabel(r"Mean-Squared Error $(MSE)$")

        # Set the figure title
        # ax1.set_title(r"Convergence Progress: Mean-Squared Error vs. Training Epoch")
        
        # Set the figure legend
        ax1.legend(title=r"Batch Fraction ($\times 10^{-4})$",loc="upper right",ncol=3,labelcolor="black")

        if save:
            savename = os.path.join(plot_directory,"convergence_batchfraction_"+dataset+".png")
            plt.savefig(savename)
        else: pass

        plt.show()
    
    ##
        
    #==========================================================================
    # Matplotlib code
    
    #==========================================================================
            
    return None

#==============================================================================

def PlotContourSlices(dataset,axis,cut,save=False):
    
    from matplotlib.gridspec import GridSpec
    
    plt.style.use("plot.mplstyle")  
    
    params_plot = {'text.latex.preamble': [r'\usepackage{amsmath}',r'\usepackage{amssymb}'],
                   'axes.grid': False,
                   'xtick.major.top'         : False,
                   'xtick.major.bottom'      : False,
                   'ytick.major.left'        : False,
                   'ytick.major.right'       : False,
                   'figure.figsize'          : (6.3*1.5,4.5*1.5),}
    
    matplotlib.rcParams.update(params_plot) 

    #==========================================================================
    # Set filepaths and directories
    
    # Set base directory
    base_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"
    
    # Set current directory
    plot_directory = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/plots"
    
  
    #======================================================================
    # Make figure and plot axes
    fig = plt.figure(constrained_layout=True)
    
    # Define a gridspec
    gridspec = fig.add_gridspec(nrows=4,ncols=7,height_ratios=[1,1,1,1], width_ratios=[1,1,1,1,1,1,0.2] )
    
    #======================================================================
    
    # Set input filepath
    input_volume_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/" + dataset + ".npy"
    
    # Get a sorted list of all the experiements directories
    experiment_directories = []
    
    for folder_name in os.listdir(base_directory):
        if (dataset in folder_name) and ("compressratio" in folder_name):
            experiment_directories.append(folder_name)
    
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
        
        # Make 2D cuts from 3D data
        
        volx_true,voly_true,vals_true = MakeSliceFrom3D(data=input_data, cut=cut,axis=axis)
        volx_pred,voly_pred,vals_pred = MakeSliceFrom3D(data=output_data,cut=cut,axis=axis)
                   
        # Normalise values by the same maximum and minimum
        vals_true = Normalise(data=vals_true,maximum=vals_true.max(),minimum=vals_true.min())
        vals_pred = Normalise(data=vals_pred,maximum=vals_true.max(),minimum=vals_true.min())
        
        # Compute the errors between input and output vals
        val_error = (vals_true - vals_pred)
        
        if dataset == "passage": volx_true,voly_true = voly_true,volx_true
        
        #==================================================================
        # Matplotlib code
        
        row = (0 if index2 < 4 else 2)
        col = ((index2 + 2) % 6)
        
        vmin1,vmax1,layers = 0.0,1.0,25
        
        ax1 = fig.add_subplot(gridspec[row,  col]) 
        plt1 = ax1.contourf(volx_true,voly_true,vals_pred,np.linspace(vmin1,vmax1,layers),vmin=vmin1,vmax=vmax1,cmap=plt.get_cmap('Blues')  ,extend="both",alpha=1.0)        
        
        vmin2,vmax2,layers = -0.25,0.25,24
  
        ax2 = fig.add_subplot(gridspec[row+1,col])
        plt2 = ax2.contourf(volx_true,voly_true,val_error,np.linspace(vmin2,vmax2,layers),vmin=vmin2,vmax=vmax2,cmap=plt.get_cmap('seismic'),extend="both",alpha=1.0)
        # print(val_error.max(),val_error.min())
        
        if (index2==0):
            ax3 = fig.add_subplot(gridspec[0:2,0:2])
            ax3.contourf(volx_true,voly_true,vals_true,np.linspace(vmin1,vmax1,layers),vmin=vmin1,vmax=vmax1,cmap=plt.get_cmap('Blues'),extend="both",alpha=1.0)
            
            ax1.set_title(fr"$\gamma$ = {actual_compression_ratio[-1]:.2f}")
            ax3.set_title(fr"Ground Truth (Axis = {axis.upper()}, Index = {cut})")
        else: 
            ax1.set_title(fr"$\gamma$ = {actual_compression_ratio[-1]:.2f}")

        # if (index2 in [3,9]):
            
            # ax1.yaxis.set_label_position("right")
            # ax1.set_ylabel("Values")
            # ax2.yaxis.set_label_position("right")
            # ax2.set_ylabel("Errors")
            
    from matplotlib.colorbar import Colorbar
    from matplotlib import ticker
            
    cbax1 = fig.add_subplot(gridspec[0:2,6])
    cb1 = Colorbar(ax=cbax1,mappable=plt1,orientation="vertical",ticklocation="right",ticks=np.linspace(0,1,9))
    cb1.set_label(label=r"Scalar Values [-]",labelpad=8)
    cb1.ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.3f}'.format))
    
    cbax2 = fig.add_subplot(gridspec[2:4,6])
    cb2 = Colorbar(ax=cbax2,mappable=plt2,orientation="vertical",ticklocation="right",ticks=np.linspace(-0.25,0.25,9))
    cb2.set_label(label=r"Absolute Errors [-]",labelpad=8)
    cb2.ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.3f}'.format))

    #======================================================================
    # Specific data wrangling      
    
    #======================================================================
    # Matplotlib code
    
    savename = os.path.join(plot_directory,"contours_"+ dataset + "_" + axis + "_" + str(cut) + ".png")
    print(savename)
    
    if save:
        savename = os.path.join(plot_directory,"contours_"+ dataset + "_" + axis + "_" + str(cut) + ".png")
        plt.savefig(savename)
    else: pass
    
    plt.show()       

    #==========================================================================
    # Matplotlib code

    #==========================================================================
            
    return None

#==============================================================================
    
def MakeSliceFrom3D(data,cut,axis='x'):
    
    if axis == 'x': var1,var2,vals = data[cut,:,:,1],data[cut,:,:,2],data[cut,:,:,3]
    
    if axis == 'y' or axis == 'r': var1,var2,vals = data[:,cut,:,0],data[:,cut,:,2],data[:,cut,:,3]
    
    if axis == 'z' or axis == 't': var1,var2,vals = data[:,:,cut,0],data[:,:,cut,1],data[:,:,cut,3]
        
    return var1,var2,vals

#==============================================================================
    
def Normalise(data,maximum,minimum):
    
    data = (data-((data.max()+data.min())/2))/(abs(data.max()-data.min()))+0.5
        
    return data

#==============================================================================

save = True

PlotErrorDurationBatch(save=save)
PlotConvergenceBatch(save=save)


PlotContourSlices(dataset = "cube"   , axis = "x", cut = 0  , save=save)
PlotContourSlices(dataset = "cube"   , axis = "x", cut = 75 , save=save)
PlotContourSlices(dataset = "cube"   , axis = "x", cut = 149, save=save)
PlotContourSlices(dataset = "cube"   , axis = "y", cut = 0  , save=save)
PlotContourSlices(dataset = "cube"   , axis = "y", cut = 75 , save=save)
PlotContourSlices(dataset = "cube"   , axis = "y", cut = 149, save=save)
PlotContourSlices(dataset = "cube"   , axis = "z", cut = 0  , save=save)
PlotContourSlices(dataset = "cube"   , axis = "z", cut = 75 , save=save)
PlotContourSlices(dataset = "cube"   , axis = "z", cut = 149, save=save)

PlotContourSlices(dataset = "passage", axis = "x", cut = 0  , save=save)
PlotContourSlices(dataset = "passage", axis = "x", cut = 75 , save=save)
PlotContourSlices(dataset = "passage", axis = "x", cut = 130, save=save)
PlotContourSlices(dataset = "passage", axis = "r", cut = 0  , save=save)
PlotContourSlices(dataset = "passage", axis = "r", cut = 25 , save=save)
PlotContourSlices(dataset = "passage", axis = "r", cut = 48 , save=save)
PlotContourSlices(dataset = "passage", axis = "t", cut = 0  , save=save)
PlotContourSlices(dataset = "passage", axis = "t", cut = 25 , save=save)
PlotContourSlices(dataset = "passage", axis = "t", cut = 48 , save=save)

