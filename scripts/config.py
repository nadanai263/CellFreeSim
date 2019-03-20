# config.py
import numpy as np

# Set filename and input and output paths.
PATH_IN = '../model/'
FILENAME_IN = 'model' # Antimony model file as input
PATH_OUT = '../cythonized_model/'
FILENAME_OUT = 'model.pyx'
PATH_PLOT = '../plots/'

# Set simulation parameters
TMAX = 200
NSTEPS = 200 

# Set chemostat experiment parameters
DIL_FRAC = 0.2
INTERVAL_IMG = 0.2
INTERVAL_DIL = 3
INDEX_REFRESH = [3] # which species are refreshed?

