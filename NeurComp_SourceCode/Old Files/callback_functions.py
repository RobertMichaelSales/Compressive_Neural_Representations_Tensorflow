""" Created: 18.07.2022  \\  Updated: 18.07.2022  \\   Author: Robert Sales """

#=# IMPORT LIBRARIES #========================================================#

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

#=# DEFINE FUNCTIONS #========================================================#

def LearningScheduler(epoch,lr):
    
    # Temporarily import the ParameterArgs class
    from network_hyperparameters import HyperparameterClass
    
    # Temporarily declare a copy of hyperparameters
    args = HyperparameterClass()
    
    initial_lr, annealing = args.learn_rate, args.annealing

    # Compute the current learning rate
    if annealing:
        new_lr = initial_lr/(2**(int(epoch/5)))
    else:
        new_lr = initial_lr
    
    return new_lr


def TrainingCallbacks(hyperparameters,filepaths):
    
    # Define the learning rate callback
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(
        LearningScheduler,
        verbose=1
    )
    
    # Define the Tensorboard callback
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
    tensorboard_log_dir = filepaths.tensorboard_path
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = tensorboard_log_dir + "//" + hyperparameters.save_name,
        histogram_freq = 1,
        write_graph = False,
        write_images = False,
        #write_steps_per_second = False,
        update_freq = 'batch'
    )
    
    
    # Define the checkpoint save callback
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    checkpoint_filepath = filepaths.checkpoints_path
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath + "//" + hyperparameters.save_name +"_epoch{epoch:04d}_loss{loss:0.4f}",
        monitor = 'loss',
        verbose = 0,
        save_best_only = True,
        mode = 'min',
        save_weights_only = True,
        save_freq = 'epoch'
    )
    
    
    # Define the csv_logger callback
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger
    csv_logger_filepath = filepaths.csv_summary_path
    
    csv_logger_callback = tf.keras.callbacks.CSVLogger(
        filename = csv_logger_filepath + "//" + hyperparameters.save_name +"_Training_Log.csv",
        separator=",",
        append=True
    )
        
    
    # Group callbacks
    callbacks = [learning_rate_callback,tensorboard_callback,checkpoint_callback,csv_logger_callback]
    
    return callbacks

#=# DEFINE CLASSES #==========================================================#

#=============================================================================#