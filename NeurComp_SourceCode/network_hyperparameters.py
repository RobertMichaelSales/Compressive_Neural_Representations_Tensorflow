""" Created: 18.07.2022  \\  Updated: 18.07.2022  \\   Author: Robert Sales """

#=# IMPORT LIBRARIES #========================================================#

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from network_make import *

#=# DEFINE FUNCTIONS #========================================================#


#=# DEFINE CLASSES #==========================================================#

class HyperparameterClass():

    def __init__(self):
   
        # Set network hyperparameters (USER SPECIFIED)
        
        self.save_name              = "NeurComp_V6"
        self.compression_ratio      = 400        
        self.hidden_layers          = 10
        self.min_neurons_per_layer  = 10
        
        # Set training hyperparameters (USER SPECIFIED)
        
        self.learn_rate             = 1.0e-4
        self.batch_size             = 16000
        self.max_epochs             = 48
        self.error_function         = "Squared"
        self.error_power            = None
        self.normalise              = False
        self.annealing              = True
        
        return None
    
    def NetworkDetails(self,input_data):
        
        self.total_num_layers       = self.hidden_layers + 2
        
        # Get dataset hyperparameters
        
        self.input_dimension        = input_data.volume.shape[-1]
        self.yield_dimension        = 1
           
        # Compute network size
        
        self.volume_dims            = input_data.volume.shape   
        self.volume_size            = np.product([x for x in self.volume_dims])
        self.target_size            = self.volume_size/self.compression_ratio
        
        self.neurons_per_layer      = ComputeNeuronsPerLayer(self)
        self.num_of_parameters      = ComputeTotalParameters(self,self.neurons_per_layer)
        
        # Set other hyperparameters
    
        self.layer_dimensions       = []   
        
        self.layer_dimensions.extend([self.input_dimension])  
        self.layer_dimensions.extend([self.neurons_per_layer]*self.hidden_layers)  
        self.layer_dimensions.extend([self.yield_dimension]) 
                
        return None
        
    def SaveHyperparameters(self,filepaths):
        
        # Obtain current date and time data
    
        from datetime import datetime
        now = datetime.now()
        date_and_time = now.strftime("%d %B %Y, %H:%M:%S").upper()
    
        # Obtain a dictionary of hyperparameters
    
        dictionary_of_hyperparameters = vars(self)
    
        # Add the date & time 
    
        dictionary_of_hyperparameters["date_and_time"] = date_and_time
    
        # Write all entries to CSV
    
        with open(filepaths.hyperparameters_path,"w") as log:
            for key,value in dictionary_of_hyperparameters.items():
                string = "{:30}: {:30}\n".format(str(key),str(value))
                log.write(string)
    
        return None       
    
#=============================================================================#