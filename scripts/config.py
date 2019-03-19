# config.py
import numpy as np


# Set filename and input and output paths.

PATH_IN = '../model/'
FILENAME_IN = 'model' # Antimony model file as input
PATH_OUT = '../cythonized_model/'
FILENAME_OUT = 'model.pyx'
PATH_PLOT = '../plots/'

# Set simulation parameters
TMAX = 100
NSTEPS = 100 

# Derived quantities; don't set these
time = np.linspace(0,TMAX,NSTEPS)
