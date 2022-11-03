""" Created: 18.07.2022  \\  Updated: 02.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np

from datetime import datetime

#==============================================================================
# Define a class for generating, managing and storing the network configuration

class NetworkConfigClass():
    
    #==========================================================================
    # Define the initialisation constructor function for 'NetworkConfigClass'
    # Note: These are the only user-configurable hyperparameters

    def __init__(self,name):
        
        # Initialise the name of the 'NetworkConfigClass' object
        self.name = name
        print("Creating NetworkConfigClass Object: {}".format(self.name))
   
        # Set network hyperparameters
        self.network_name                       = "neurcomp_v1"
        self.target_compression_ratio           = 50        
        self.hidden_layers                      = 8
        self.min_neurons_per_layer              = 10
        
        # Set training hyperparameters
        self.initial_learning_rate              = 5e-3
        self.batch_size                         = 16384
        self.num_epochs                         = 1
        self.decay_rate                         = 5
           
        return None
    
    #==========================================================================
    # Define a function to generate the network structure/dimensions from input
    # dimensions, output dimensions and input size. The 'layer_dimensions' list
    # has dimensions for each layer as its elements
    
    def NetworkStructure(self,input_data):
                
        # Extract the useful internal parameters from the 'input_data' object
        self.input_dimensions = input_data.input_dimensions
        self.output_dimensions = input_data.output_dimensions
        input_size = input_data.input_size
        
        # Compute the neurons per layer as well as the overall network capacity
        self.target_size = int(input_size / self.target_compression_ratio)
        self.neurons_per_layer = self.NeuronsPerLayer() 
        self.num_of_parameters = self.TotalParameters()
        
        # Specify the network architecture as a list of layer dimensions
        self.layer_dimensions       = []   
        self.layer_dimensions.extend([self.input_dimensions])  
        self.layer_dimensions.extend([self.neurons_per_layer]*self.hidden_layers)  
        self.layer_dimensions.extend([self.output_dimensions]) 
        
        print("Calculating Network Dimensions: {}\n".format(self.layer_dimensions))

        return None

    #==========================================================================
    # Define a function to compute the minimum number of neurons needed by each 
    # layer in to achieve (just exceed) the target compression ratio

    def NeuronsPerLayer(self):
      
        # Start searching from the minimum allowed number of neurons per layer
        self.neurons_per_layer = self.min_neurons_per_layer
          
        # Incriment neurons until the network capacity exceeds the target size
        while (self.TotalParameters() < self.target_size):
            self.neurons_per_layer = self.neurons_per_layer + 1
          
        # Determine the first neuron count that exceeds the target compression
        self.neurons_per_layer = self.neurons_per_layer - 1
        
        return self.neurons_per_layer
    
    #==========================================================================
    # Define a function to calculate the total number of network parameters for
    # a given network architecture (i.e. layer dimensions/neurons)
    
    def TotalParameters(self):
        
        # The network architecture can be built from the following blocks:
        # [input_layer -> dense_layer] + 
        # [dense_layer / residual_block -> residual_block] +
        # [residual_layer -> output_layer]    
     
        # Determine the number of inter-layer operations (i.e. total layers)
        self.total_layers = self.hidden_layers + 2     
                  
        # Set the total number of network parameters to zero
        self.num_of_parameters = 0                                                         
          
        #Iterate through each layer in the network (including input/output)
        for layer in np.arange(self.total_layers):
          
            # [input -> dense]
            if (layer==0):                             
                
                # Determine the input and output dimensions of each layer
                dim_input  = self.input_dimensions              
                dim_output = self.neurons_per_layer
                  
                # Add parameters from the weight matrix and bias vector
                self.num_of_parameters += (dim_input * dim_output) + dim_output
          
            # [residual -> output]
            elif (layer==self.total_layers-1):     
    
                # Determine the input and output dimensions of each layer
                dim_input  = self.neurons_per_layer
                dim_output = self.output_dimensions
                  
                # Add parameters from the weight matrix and bias vector
                self.num_of_parameters += (dim_input * dim_output) + dim_output 
          
            # [dense / residual -> residual]
            else:                         
                
                # Add parameters from the weight matrix and bias vector
                self.num_of_parameters += (self.neurons_per_layer * self.neurons_per_layer) + self.neurons_per_layer
                self.num_of_parameters += (self.neurons_per_layer * self.neurons_per_layer) + self.neurons_per_layer        
                  
        return self.num_of_parameters

    #==========================================================================
    # Define a function to save the network configuration to a '.json' filetype
    
    def SaveConfigToJson(self,configuration_filepath):
        
        print("Saving network configuration: {}".configuration_filepath)
        
        # Determine the file extension (type) from the provided file path
        extension = configuration_filepath.split(".")[-1].lower()
        
        # If the extension matches ".npy" then load it, else throw an error
        if extension == "json":  
            
            # Obtain a dictionary with the network configuration (i.e. config)
            configuration_dictionary = vars(self)
            
            # Obtain date and time data, then add it to the config dictionary
            now = datetime.now()
            date_and_time_string = now.strftime("%d %b %Y, %H:%M:%S").upper()
            configuration_dictionary["date_time"] = date_and_time_string
            
            # Write all entries in 'configuration_dictionary' to a .JSON file
            with open(configuration_filepath, 'w') as configuration_file:
                json.dump(configuration_dictionary,configuration_file,indent=4)
              
        else:
            print("Error: File Type Not Supported: '{}'. ".format(extension))    
    
        return None
    
#=============================================================================#