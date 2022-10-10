""" Created: 18.07.2022  \\  Updated: 18.07.2022  \\   Author: Robert Sales """

#=# IMPORT LIBRARIES #========================================================#

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

#=# DEFINE FUNCTIONS #========================================================#

def MeanAbsoluteError(y_true,y_pred):
    
    mae = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(y_pred,y_true)))
    
    return mae


def MeanSquaredError(y_true,y_pred):

    mse = tf.math.reduce_mean(tf.math.square(tf.math.subtract(y_pred,y_true)))
    
    return mse


def MeanSquareRootError(y_true,y_pred):

    msr = tf.math.reduce_mean(tf.math.sqrt(tf.math.abs(tf.math.subtract(y_pred,y_true))))
    
    return msr


def MeanAnyPowerError(y_true,y_pred):

    mpe = tf.math.reduce_mean(tf.math.pow(tf.math.abs(tf.math.subtract(y_pred,y_true)),POW))
    
    return mpe


def GetErrorFunction(hyperparameters):
    
    error_functions = {"Absolute":      MeanAbsoluteError,
                       "Squared":       MeanSquaredError,
                       "SquareRoot":    MeanSquareRootError,
                       "Power":         MeanAnyPowerError}
    
    return error_functions[hyperparameters.error_function]

#=============================================================================#