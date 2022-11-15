""" Created: 11.08.2022  \\  Updated: 27.10.2022  \\   Author: Robert Sales """

#=# IMPORT LIBRARIES #========================================================#

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

from datetime import datetime

#=# DEFINE FUNCTIONS #========================================================#


def LoadingBar(string,current_count,final_count,bar_length):
        
    # Calculate percentage complete
    percent = (100*(current_count+1)/final_count)
    
    # Calculate number of symbols for loading bar
    num_symbols = round(percent *(bar_length/100))
    
    # Compile string for animated progress bar
    bar = ("["+"="*(num_symbols-1)+">"+"."*(bar_length-num_symbols)+"]")
    
    # Compile the progress message 
    message = ("\r"+ string + ": "+ bar +" {:3.0f}%.".format(percent))
    
    print(message,end="")
    
    return None


def ElapsedTime(times):
    
    # Calculate the most recent delta_t
    delta_t = times[-1]-times[-2]
    
    # Convert delta_t to seconds
    seconds = delta_t.seconds
    
    # Convert seconds to minutes and hours
    m,s = divmod(seconds,60)
    h,m = divmod(m,60)
    
    # Convert to a printable string
    string = ('{:d} hours, {:02d} minutes, {:02d} seconds.'.format(h, m, s))
    
    # Compile the elapsed time message
    message = ("Process completed in: " + string + "\n")
    
    print(message)
    
    return None

#=# DEFINE CLASSES #==========================================================#

class LoggerClass(object):
    
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("console_logs.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        
    def flush(self):
        pass
        
#=============================================================================#