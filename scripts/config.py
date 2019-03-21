# config.py
import numpy as np

# Set filename and input and output paths.
PATH_IN = '../model/'
FILENAME_IN = 'model_simpleTXTL' # Antimony model file as input
PATH_OUT = '../cythonized_model/'
FILENAME_OUT = 'model.pyx'
PATH_PLOT = '../plots/'

# Set simulation parameters
TMAX = 100
NSTEPS = 100

# Set chemostat experiment parameters
DIL_FRAC = 0.2
INTERVAL_IMG = 5 # has to be factor of TMAX and INTERVAL_DIL
INDEX_REFRESH = [3,4] # which species are refreshed?
nstepschemos = np.int(TMAX/INTERVAL_IMG)+1

# Either periodic dilution:
INTERVAL_DIL = 10 # has to be factor of TMAX
ndilsteps=np.int((TMAX)/(INTERVAL_DIL)+1)
#dilutiontimes=np.linspace(0,TMAX,ndilsteps)

# Or arbitrary dilution:
dilutiontimes = np.array([0.0,50.0,55.0,70.0,TMAX])


# Set MCMC parameters
DATACHANNEL = 2 # Index of data channel to pass to MCMC (which species are we observing?)
PARAMCHANNELS = [1,3] # Index of param we are inferring

nDimParams,nwalkers,threads,iterations,tburn=2,20,4,5000,2500
labels=["$lngamma$","$lnL1$"]
parametertruths=[np.log(0.1),np.log(0.1)]
pos=[np.array([
    np.log(0.1)*(1+0.1*np.random.randn()),
    np.log(0.1)*(1+0.1*np.random.randn())
    ]) for i in range(nwalkers)]

# Prior distribution set here: mus then sigmas
PMUSIGMA = [[np.log(0.1),np.log(0.1)],
            [3,3]]

# Experimental sigma
SIGMA = 1
