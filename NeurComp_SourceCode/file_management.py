""" Created: 18.07.2022  \\  Updated: 18.07.2022  \\   Author: Robert Sales """

#=# IMPORT LIBRARIES #========================================================#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#=# DEFINE CLASSES #==========================================================#
     
class FilepathClass():
    
    def __init__(self,parent_folder,hyperparameters):
        
        # Get parent folder and network save name
        self.parent_folder = parent_folder
        save_name = hyperparameters.save_name
        
        # Make folders        
    
        # Make TensorBoard folder + path
        self.tensorboard_path = os.path.join(self.parent_folder,
                                             "Training",
                                             "Logs",
                                             "TensorBoard",
                                             save_name)  
        
        if not os.path.exists(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)
            print("Creating Folder: '{}'\n".format(self.tensorboard_path))
          
        # Make CSV summary folder + path
        self.csv_summary_path = os.path.join(self.parent_folder,
                                             "Training",
                                             "Logs",
                                             "CSV Summary",
                                             save_name)
        
        if not os.path.exists(self.csv_summary_path):
            os.makedirs(self.csv_summary_path)
            print("Creating Folder: '{}'\n".format(self.csv_summary_path))
    
        # Make checkpoints folder + path
        self.checkpoints_path = os.path.join(self.parent_folder,
                                             "Training",
                                             "Checkpoints",
                                             save_name)
        
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
            print("Creating Folder: '{}'\n".format(self.checkpoints_path))
            
        # Make trained model folder + path
        self.trained_model_path = os.path.join(self.parent_folder,
                                               "Training",
                                               "Trained Models",
                                               save_name)
        
        if not os.path.exists(self.trained_model_path):
            os.makedirs(self.trained_model_path)
            print("Creating Folder: '{}'\n".format(self.trained_model_path))

        # Make output volume folder + path
        self.output_volume_path = os.path.join(self.parent_folder,
                                               "Outputs",
                                               "Volumes",
                                               save_name)
        
        if not os.path.exists(self.output_volume_path):
            os.makedirs(self.output_volume_path)
            print("Creating Folder: '{}'\n".format(self.output_volume_path))
            
        # Make filepaths
        
        # Make normal model save filepath
        normal_model_name = hyperparameters.save_name        
        self.normal_model_path = os.path.join(self.trained_model_path,
                                              normal_model_name)
        
        # Make tflite model save filepath
        tflite_model_name = hyperparameters.save_name + "_normal" + ".tflite"        
        self.tflite_model_path = os.path.join(self.trained_model_path,
                                              tflite_model_name)
        
        # Make quantised model save filepath
        quantd_model_name = hyperparameters.save_name + "_quantd" + ".tflite"
        self.quantd_model_path = os.path.join(self.trained_model_path,
                                              quantd_model_name)
        
        # Make hyperparameters save filepath
        hyperparameters_name = hyperparameters.save_name + "_parameters.txt"
        self.hyperparameters_path = os.path.join(self.csv_summary_path,
                                                 hyperparameters_name)

        return None
#=============================================================================#