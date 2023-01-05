""" Created: 18.07.2022  \\  Updated: 05.01.2023  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os, json
import numpy as np

#==============================================================================
# Define a class for generating, managing and storing the network configuration

class ConfigurationClass():
    
    #==========================================================================
    # Define the initialisation constructor function for 'NetworkConfigClass'
    # Note: These are the only user-configurable hyperparameters

    def __init__(self,config_path):
        
        # If config file IS provided: 
        if os.path.exists(config_path):
            
            print("\n{:30}{}".format("Loaded network config:",config_path.split("/")[-1]))
            
            # Load config data file
            with open(config_path) as config_file:
                config_dictionary = json.load(config_file)

            # Set config attributes 
            for attribute in config_dictionary.keys():
                setattr(self,attribute,config_dictionary[attribute])
        
        # If config file NOT provided: 
        else:
            
            print("\n{:30}{}".format("Loaded network config:","default"))
               
            # Set network hyperparameters (default)
            self.network_name                       = "neurcomp_default"
            self.target_compression_ratio           = 50        
            self.hidden_layers                      = 8
            self.min_neurons_per_layer              = 10
            
            # Set training hyperparameters (default)
            self.initial_lr                         = 5e-3
            self.batch_size                         = 1024
            self.epochs                             = 1
            self.half_life                          = 3
            
        print("\n{:30}{}".format("Target compression ratio:",self.target_compression_ratio))
           
        return None
    
    #==========================================================================
    # Define a function to generate the network structure/dimensions from input
    # dimensions, output dimensions and input size. The 'layer_dimensions' list
    # has dimensions for each layer as its elements
    
    def GenerateStructure(self,i_dimensions,o_dimensions,i_size):
                
        # Extract the useful internal parameters from the 'input_data' object
        self.i_dimensions = i_dimensions
        self.o_dimensions = o_dimensions
        self.i_size = i_size
        
        # Compute the neurons per layer as well as the overall network capacity
        self.target_size = int(self.i_size/self.target_compression_ratio)
        self.neurons_per_layer = self.NeuronsPerLayer() 
        self.num_of_parameters = self.TotalParameters()
        self.actual_compression_ratio = self.i_size/self.num_of_parameters
        
        print("\n{:30}{}".format("Actual compression ratio:",self.actual_compression_ratio))
        
        # Specify the network architecture as a list of layer dimensions
        self.layer_dimensions = []   
        self.layer_dimensions.extend([self.i_dimensions])  
        self.layer_dimensions.extend([self.neurons_per_layer]*self.hidden_layers)  
        self.layer_dimensions.extend([self.o_dimensions]) 
        
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
    
    # The network structure can be summarised as follows:
    # [input_layer      -> sine_layer] + 
    # [sine_layer/block -> sine_block] +
    # [sine_block       -> output_layer]  
    
    def TotalParameters(self):
    
        # Determine the number of inter-layer operations (i.e. total layers)
        self.total_layers = self.hidden_layers + 2     
                  
        # Set the total number of parameters to zero
        self.num_of_parameters = 0                                                         
          
        #Iterate through each layer in the network (including input/output)
        for layer in np.arange(self.total_layers):
          
            # [input_layer -> sine_layer]
            if (layer==0):                             
                
                # Determine the input and output dimensions of each layer
                i_dimensions = self.i_dimensions              
                o_dimensions = self.neurons_per_layer
                  
                # Add parameters from the weight matrix and bias vector
                self.num_of_parameters += (i_dimensions * o_dimensions) + o_dimensions
          
            # [sine_block -> output_layer]
            elif (layer==self.total_layers-1):     
    
                # Determine the input and output dimensions of each layer
                i_dimensions = self.neurons_per_layer
                o_dimensions = self.o_dimensions
                  
                # Add parameters from the weight matrix and bias vector
                self.num_of_parameters += (i_dimensions * o_dimensions) + o_dimensions 
          
            # [sine_layer/block -> sine_block]
            else:                         
                
                # Add parameters from the 1st weight matrix and bias vector
                self.num_of_parameters += (self.neurons_per_layer * self.neurons_per_layer) + self.neurons_per_layer
                
                # Add parameters from the 2nd weight matrix and bias vector
                self.num_of_parameters += (self.neurons_per_layer * self.neurons_per_layer) + self.neurons_per_layer        
                  
        return self.num_of_parameters

#=============================================================================#
