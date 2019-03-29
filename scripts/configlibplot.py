# configlibplot.py
# Config for plotting entire library
# as called from libaryplot.py
import numpy as np

# Set filename and input and output paths.
PATH_IN = '../model/'
FILENAME_IN = 'model_simpleTXTL' # Antimony model file as input
PATH_OUT = '../cythonized_model/'
FILENAME_OUT = 'model.pyx'
PATH_PLOT = '../plots/'

# Set simulation parameters
TMAX = 12*60 # minutes
NSTEPS = 200

# Set chemostat experiment parameters

DIL_FRAC = 0.0
INTERVAL_IMG = 5 # has to be factor of TMAX and INTERVAL_DIL
nstepschemos = np.int(TMAX/INTERVAL_IMG)+1

# Either periodic dilution:
INTERVAL_DIL = 20 # has to be factor of TMAX
ndilsteps=np.int((TMAX)/(INTERVAL_DIL)+1)
dilutiontimes=np.linspace(0,TMAX,ndilsteps)

# Set MCMC parameters
DATACHANNELS = [1,2] # Index of data channel to pass to MCMC (which species are we observing?)
PARAMCHANNELS = [1,3] # Index of param we are inferring

nwalkers,threads,iterations,tburn=20,2,3000,2000

# NUmber of model runs
NMODELRUNS = 20

# Experimental sigma
SIGMA = 1
