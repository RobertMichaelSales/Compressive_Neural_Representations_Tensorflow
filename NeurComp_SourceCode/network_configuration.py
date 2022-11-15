""" Created: 18.07.2022  \\  Updated: 10.11.2022  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np

#==============================================================================
# Define a class for generating, managing and storing the network configuration

class NetworkConfigClass():
    
    #==========================================================================
    # Define the initialisation constructor function for 'NetworkConfigClass'
    # Note: These are the only user-configurable hyperparameters

    def __init__(self,config_filepath):
        
        # If config file IS provided: 
        if os.path.exists(config_filepath):
            
            print("\n{:30}{}".format("Loaded network config:",config_filepath.split("/")[-1]))
            
            # Load config data from config file
            config_file = open(config_filepath)
            config_dict = json.load(config_file)
        
            # Set attributes from 'config_dict'
            for key in config_dict.keys():
                setattr(self,key,config_dict[key])
        
        # If config file IS NOT provided: 
        else:
            
            print("\n{:30}{}".format("Loaded network config:","default"))
               
            # Set network hyperparameters (default)
            self.network_name                       = "neurcomp_default"
            self.target_compression_ratio           = 50        
            self.hidden_layers                      = 8
            self.min_neurons_per_layer              = 10
            
            # Set training hyperparameters (default)
            self.initial_learning_rate              = 5e-3
            self.batch_size                         = 1024
            self.num_epochs                         = 1
            self.decay_rate                         = 3
            
        print("\n{:30}{}".format("Target compression ratio:",self.target_compression_ratio))
           
        return None
    
    #==========================================================================
    # Define a function to generate the network structure/dimensions from input
    # dimensions, output dimensions and input size. The 'layer_dimensions' list
    # has dimensions for each layer as its elements
    
    def NetworkStructure(self,input_data):
                
        # Extract the useful internal parameters from the 'input_data' object
        self.input_dimensions = input_data.input_dimensions
        self.output_dimensions = input_data.output_dimensions
        size = input_data.size
        
        # Compute the neurons per layer as well as the overall network capacity
        self.target_size = int(size/self.target_compression_ratio)
        self.neurons_per_layer = self.NeuronsPerLayer() 
        self.num_of_parameters = self.TotalParameters()
        self.actual_compression_ratio = size/self.num_of_parameters
        
        # Specify the network architecture as a list of layer dimensions
        self.layer_dimensions = []   
        self.layer_dimensions.extend([self.input_dimensions])  
        self.layer_dimensions.extend([self.neurons_per_layer]*self.hidden_layers)  
        self.layer_dimensions.extend([self.output_dimensions]) 
        
        print("\n{:30}{}".format("Network dimensions:",self.layer_dimensions))

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

#=============================================================================#
