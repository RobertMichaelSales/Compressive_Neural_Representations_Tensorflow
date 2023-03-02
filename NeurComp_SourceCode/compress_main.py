""" Created: 18.07.2022  \\  Updated: 01.03.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os, time, json, math, psutil, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

#==============================================================================
# Import user-defined libraries 

from data_management         import DataClass,MakeDataset,SaveData
from network_encoder         import EncodeParameters,EncodeArchitecture
from network_model           import ConstructNetwork
from configuration_classes   import NetworkConfigurationClass,GenericConfigurationClass
from compress_utilities      import TrainStep,SignalToNoise,GetLearningRate,CalculatePointCloudDensity,CalculateStandardDeviation,LRStudy,BFStudy

#==============================================================================

def compress(network_config,runtime_config,training_config,metadata_config,i_filepath,o_filepath):
    
    print("-"*80,"\nSQUASHNET: IMPLICIT NEURAL REPRESENTATIONS (by Rob Sales)")
    
    #==========================================================================
    # Check whether hardware acceleration is enabled
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if (len(gpus) != 0): 
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Enabled") )
        tf.config.experimental.set_memory_growth(gpus[0],True)
    else:        
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Disabled"))
        return None
    
    #==========================================================================
    # Check whether the input size exceeds available memory
    
    available_memory = psutil.virtual_memory().available
    print("\n{:30}{:.3f} GigaBytes".format("Available Memory:",(available_memory/1e9)))
    
    threshold_memory = int(4*1e9)
    print("\n{:30}{:.3f} GigaBytes".format("Threshold Memory:",(threshold_memory/1e9)))
    
    input_file_size = os.path.getsize(i_filepath)
    print("\n{:30}{:.3f} GigaBytes".format("Input File Size:",(input_file_size/1e9)))
    
    exceeds_memory = (input_file_size > min(available_memory,threshold_memory))
    
    if exceeds_memory: print("\n{:30}{}".format("Warning:","File Size > RAM Thresold"))
    
    #==========================================================================
    # Initialise i/o 
    
    print("-"*80,"\nINITIALISING DATA I/O:")
    
    # Create 'DataClass' objects to store i/o volume and values
    i_volume = DataClass(data_type="volume",is_tabular=metadata_config.is_tabular,exceeds_memory=exceeds_memory)
    i_values = DataClass(data_type="values",is_tabular=metadata_config.is_tabular,exceeds_memory=exceeds_memory)
    o_volume = DataClass(data_type="volume",is_tabular=metadata_config.is_tabular,exceeds_memory=exceeds_memory)
    o_values = DataClass(data_type="values",is_tabular=metadata_config.is_tabular,exceeds_memory=exceeds_memory)
    
    # Load and normalise input data
    i_volume.LoadData(input_data_path=i_filepath,columns=metadata_config.columns,shape=metadata_config.shape,dtype=metadata_config.dtype,normalise=metadata_config.normalise)
    i_values.LoadData(input_data_path=i_filepath,columns=metadata_config.columns,shape=metadata_config.shape,dtype=metadata_config.dtype,normalise=metadata_config.normalise)
       
    # Copy meta-data from the input
    o_volume.CopyData(DataClassObject=i_volume,exception_keys=[])
    o_values.CopyData(DataClassObject=i_values,exception_keys=[])    
    
    #==========================================================================    
    # Configure network 
    print("-"*80,"\nCONFIGURING NETWORK:")
    
    # Generate the network structure based on the input dimensions
    network_config.GenerateStructure(i_dimensions=i_volume.dimensions,o_dimensions=i_values.dimensions,size=i_values.size)
    
    # Build NeurComp from the config information
    SquashNet = ConstructNetwork(layer_dimensions=network_config.layer_dimensions,frequencies=network_config.frequencies)
    
    # Set a training optimiser
    optimiser = tf.keras.optimizers.Adam()
    
    # Set a performance metric
    metric = tf.keras.metrics.MeanSquaredError()
    
    #==========================================================================
    # Configure output folder
    print("-"*80,"\nCONFIGURING FOLDERS:")
    
    # Create an output directory for all future saved files
    output_directory = os.path.join(o_filepath,network_config.network_name)
    if not os.path.exists(output_directory):os.makedirs(output_directory)
    print("\n{:30}{}".format("Created output folder:",output_directory.split("/")[-1]))

    if runtime_config.graph_flag:
        tf.keras.utils.plot_model(model=SquashNet,to_file=os.path.join(output_directory,"network_graph.png"))
    else: pass
            
    #==========================================================================
    # Configure dataset
    print("-"*80,"\nCONFIGURING DATASET:")
    
    # Choose between batch fraction and size (for experiments only)
    if (training_config.batch_fraction):
        print("\n{:30}{}".format("User Input Batch Fraction:",training_config.batch_fraction))
        training_config.batch_size = math.floor(training_config.batch_fraction*i_values.size)
        training_config.batch_fraction = (training_config.batch_size/i_values.size) 
    else: pass  

    # Generate a TF dataset to supply volume and values batches during training 
    dataset = MakeDataset(volume=i_volume,values=i_values,batch_size=training_config.batch_size,repeat=False)
    
    #==========================================================================
    # Perform a-priori studies on the statistical metrics of volume and values
    
    if runtime_config.stats_flag: 
        
        print("-"*80,"\nCALCULATING INPUT STATISTICS:")
        
        values_standard_deviation = CalculateStandardDeviation(i_values.flat)
        print("\n{:30}{}".format("Values Standard Deviation:",values_standard_deviation))
        
        volume_pointcloud_density = CalculatePointCloudDensity(i_volume.flat)
        print("\n{:30}{}".format("Volume PointCloud Density:",volume_pointcloud_density))
        
    else: pass
    
    #==========================================================================
    # Perform a-priori studies on batch fraction and the initial learning rate
    
    if runtime_config.bf_study_flag or runtime_config.lr_study_flag: 
        
        print("-"*80,"\nPERFORMING A-PRIORI STUDIES:")  
        
        if runtime_config.bf_study_flag:
            print("\nRunning Batch Fraction Study:")
            BFStudy(model=SquashNet,volume=i_volume,values=i_values,lr_guess=1.0e-4,bf_bounds=(1.0e-4,2.5e-3),plot=True)  
        else: pass
    
        if runtime_config.lr_study_flag:
            print("\nRunning Learning Rate Study:")
            LRStudy(model=SquashNet,volume=i_volume,values=i_values,bf_guess=5.0e-4,lr_bounds=(-7.000,-1.000),plot=True)         
        else: pass
    
    else: pass
    
    #==========================================================================
    # Training loop
    
    print("-"*80,"\nCOMPRESSING DATA:")
        
    # Create a dictionary of lists to store training data
    training_data = {"epoch": [],"error": [],"time": [],"learning_rate": [], "psnr": []}

    # Start the overall training timer
    training_time_tick = time.time()
    
    # Iterate through each epoch
    for epoch in range(training_config.epochs):
        
        print("\n",end="")
        
        # Store and print the current epoch number
        training_data["epoch"].append(float(epoch))
        print("{:30}{:02}/{:02}".format("Epoch:",epoch,training_config.epochs))
        
        # Determine, update, store and print the learning rate 
        learning_rate = GetLearningRate(initial_lr=training_config.initial_lr,half_life=training_config.half_life,epoch=epoch)
        optimiser.lr.assign(learning_rate)
        training_data["learning_rate"].append(float(learning_rate))   
        print("{:30}{:.3E}".format("Learning Rate:",learning_rate))
        
        # Start timing current epoch
        epoch_time_tick = time.time()
        
        ## Iterate through each batch
        for batch, (volume_batch,values_batch) in enumerate(dataset):
            
            # Print the current batch number and run a training step
            print("\r{:30}{:04}/{:04}".format("Batch Number:",(batch+1),len(dataset)),end="") 
            TrainStep(model=SquashNet,optimiser=optimiser,metric=metric,volume_batch=volume_batch,values_batch=values_batch)
        ##
        
        print("\n",end="")
        
        # End the epoch time and store the elapsed time 
        epoch_time_tock = time.time() 
        epoch_time = float(epoch_time_tock-epoch_time_tick)
        training_data["time"].append(epoch_time)
        print("{:30}{:.2f} seconds".format("Epoch Time:",epoch_time))
        
        # Fetch, store and reset and the training error
        error = float(metric.result().numpy())
        metric.reset_states()
        training_data["error"].append(error)
        print("{:30}{:.7f}".format("Mean Squared Error:",error))
    ##   
 
    # End the overall training timer
    training_time_tock = time.time()
    training_time = float(training_time_tock-training_time_tick)
    print("\n{:30}{:.2f} seconds".format("Training Duration:",training_time))    
       
    #==========================================================================
    # Save network 
    
    if runtime_config.save_network_flag:
        
        print("-"*80,"\nSAVING NETWORK:")
        print("\n",end="")
        
        # Save the parameters
        parameters_path = os.path.join(output_directory,"parameters.bin")
        EncodeParameters(network=SquashNet,parameters_path=parameters_path,values_bounds=(i_values.max,i_values.min))
        print("{:30}{}".format("Saved parameters to:",parameters_path.split("/")[-1]))
        
        # Save the architecture
        architecture_path = os.path.join(output_directory,"architecture.bin")
        EncodeArchitecture(layer_dimensions=network_config.layer_dimensions,frequencies=network_config.frequencies,architecture_path=architecture_path)
        print("{:30}{}".format("Saved architecture to:",architecture_path.split("/")[-1]))
        
    else: pass
    
    #==========================================================================
    # Save outputs
    
    if runtime_config.save_outputs_flag:
        
        print("-"*80,"\nSAVING OUTPUTS:")
        print("\n",end="")
        
        # Generate the output volume and calculate the PSNR
        o_values.flat = SquashNet.predict(o_volume.flat,batch_size=training_config.batch_size,verbose="1")
        o_values.data = np.reshape(o_values.flat,(o_volume.data.shape[:-1]+(1,)),order="C")
        print("{:30}{:.3f}".format("Output volume PSNR:",SignalToNoise(true=i_values.data,pred=o_values.data)))
        training_data["psnr"].append(SignalToNoise(true=i_values.data,pred=o_values.data))
    
        # Save the output volume to ".npy" and ".vtk" files
        output_data_path = os.path.join(output_directory,"output_volume")
        SaveData(output_data_path=output_data_path,volume=o_volume,values=o_values,reverse_normalise=True)
        print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1]))
        
    else: pass    
    
    #==========================================================================
    # Save results
    
    if runtime_config.save_results_flag:
        
        print("-"*80,"\nSAVING RESULTS:")        
        print("\n",end="")
        
        # Save the training data
        training_data_path = os.path.join(output_directory,"training_data.json")
        with open(training_data_path,"w") as file: json.dump(training_data,file,indent=4,sort_keys=True)
        print("{:30}{}".format("Saved training data to:",training_data_path.split("/")[-1]))
    
        # Save the configuration
        combined_config_path = os.path.join(output_directory,"config.json")
        combined_config_dict = (network_config | training_config | runtime_config | metadata_config)
        with open(combined_config_path,"w") as file: json.dump(combined_config_dict,file,indent=4)
        print("{:30}{}".format("Saved configuration to:",combined_config_path.split("/")[-1]))
        
    else: pass
    
    #==========================================================================
    print("-"*80,"\n")
   
#==============================================================================
# Define the main function to run when file is invoked from within the terminal

if __name__=="__main__":
    
    if (len(sys.argv) == 1):
    
        network_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/network_config.json"
        
        with open(network_config_path) as network_config_file: 
            network_config_dictionary = json.load(network_config_file)
            network_config = NetworkConfigurationClass(network_config_dictionary)
        ##   
            
        runtime_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/runtime_config.json"
        
        with open(runtime_config_path) as runtime_config_file: 
            runtime_config_dictionary = json.load(runtime_config_file)
            runtime_config = GenericConfigurationClass(runtime_config_dictionary)
        ##    
            
        training_config_path = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/configs/training_config.json"
        
        with open(training_config_path) as training_config_file: 
            training_config_dictionary = json.load(training_config_file)
            training_config = GenericConfigurationClass(training_config_dictionary)
        ##
        
    else: 

        # sys.argv[1] = '{"frequencies": 0,"hidden_layers": 8,"network_name": "squashnet_test","target_compression_ratio":100.0}'
        network_config = NetworkConfigurationClass(json.loads(sys.argv[1]))
    
        # sys.argv[2] = '{"bf_study_flag": False,"graph_flag": False,"lr_study_flag": False,"stats_flag": False}'
        runtime_config = GenericConfigurationClass(json.loads(sys.argv[2]))
       
        # sys.argv[3] = '{"batch_fraction": 0,"batch_size": 1024,"epochs": 30,"half_life": 3,"initial_lr": 0.005}'
        training_config = GenericConfigurationClass(json.loads(sys.argv[3]))
    ##    
        
    metadata_config_dictionary = {"dtype": "float32","shape": (150,150,150,4),"columns": ([0,1,2],[3]),"is_tabular": False,"normalise": True}   
    metadata_config = GenericConfigurationClass(metadata_config_dictionary)
    
    i_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/inputs/volumes/cube.npy"
    o_filepath = "/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/NeurComp_AuxFiles/outputs"

    # Execute compression
    compress(network_config=network_config,runtime_config=runtime_config,training_config=training_config,metadata_config=metadata_config,i_filepath=i_filepath,o_filepath=o_filepath)   

else: pass

#==============================================================================
