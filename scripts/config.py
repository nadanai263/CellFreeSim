# config.py
# Standard config file
# as called by cellfreesim.py

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
INTERVAL_IMG = 10 # has to be factor of TMAX and INTERVAL_DIL
INDEX_REFRESH = [0,1] # which species are refreshed?
CONC_REFRESH = [1,0.5] # by how much are the species refreshed?
nstepschemos = np.int(TMAX/INTERVAL_IMG)+1

# Either periodic dilution:
INTERVAL_DIL = 40 # has to be factor of TMAX
ndilsteps=np.int((TMAX)/(INTERVAL_DIL)+1)
#dilutiontimes=np.linspace(0,TMAX,ndilsteps)

# Or arbitrary dilution:
dilutiontimes = np.array([0,200,240,280,320,360,640,TMAX])

# Set MCMC parameters
DATACHANNELS = [4] # Index of data channel to pass to MCMC (which species are we observing?)
PARAMCHANNELS = [0,1,2,3] # Index of param we are inferring

nDimParams,nwalkers,threads,iterations,tburn=4,10,2,5000,4000
labels=["$lngamma$","$lnkmat$","$lnL1$","$lnk$"]
parametertruths=[np.log(0.1),np.log(0.1),np.log(0.01),np.log(0.05)]
pos=[np.array([initpos*(1+0.2*np.random.randn()) for initpos in parametertruths])
        for i in range(nwalkers)]

# Prior distribution set here: mus then sigmas
PMUSIGMA = [[mu for mu in parametertruths],[2 for i in range(len(parametertruths))]]

# NUmber of model runs
NMODELRUNS = 20

# Experimental sigma
SIGMA = 1
