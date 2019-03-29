# libraryplot.py
# Plots entire model library

# Interactive, fast cythonized ODE solver which
# accepts Antimony/SBML model files as inputs
# N. Laohakunakorn, LBNC, EPFL
# V1.0, 2019

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tellurium as te # 2.1.5
import numpy as np
from scipy.integrate import odeint
import os
import pyfiglet as pyfiglet
import sys
import progressbar as progressbar
import emcee
import corner
import argparse

# 1. Define model using model file
# 2. Using roadrunner, load model and parse ODEs into cython
# 3. Generate cython file and compile
# 4. Solve ODE using compiled cython file

# Import functions and configuration parameters
try:
    import functions as cfs
except:
    print('Function file functions.py not found, or import errors: please check all dependencies are installed.')
    sys.exit()
try:
    from configlibplot import *
except:
    print('No config file! Please make one and re-run. Examples can be found on the GitHub repo.')
    sys.exit()

if not os.path.exists(PATH_OUT):
    os.mkdir('../cythonized_model/')


parser = argparse.ArgumentParser()
parser.add_argument('inputname', type=str)
args = parser.parse_args()
modelName = args.inputname
PATHMODEL = '../modellib/'


# Take model name and run MCMC:
print('Loading model: ', modelName)
r = te.loada(PATHMODEL+modelName)
odes = te.getODEsFromModel(r) # Standardised ODE parser from tellurium 2.1.5
speciesIds, speciesValues, parameterIds, parameterValues, derivatives = cfs.parseODEs(r,odes)
print('ODEs', odes)
print('Species', speciesIds)
print('Parameters', parameterIds)
print('paramvalues', parameterValues)
print('\n')
print('Generating cython file\n')
cfs.writeCython(speciesIds,speciesValues,parameterIds,parameterValues,derivatives,PATH_OUT,FILENAME_OUT)
print('Cythonizing\n')
os.chdir(PATH_OUT)
os.system('python3 setup.py build_ext --inplace')
print('\n\n\n')

y0 = np.array([float(value) for value in speciesValues])
params = np.array([float(value) for value in parameterValues])

INDEX_REFRESH = [0]
CONC_REFRESH = [1]

# Reimport updated model
sys.path.insert(0, PATH_OUT)
import model as model

cinputkeys = ['dilutiontimes', 'y0', 'params', 'interval_img', 'dil_frac', 'index_refresh', 'conc_refresh','cymodel']
cinputvalues = [dilutiontimes, y0, params, INTERVAL_IMG, DIL_FRAC, INDEX_REFRESH, CONC_REFRESH, model.model]
chemostatinputs = dict(zip(cinputkeys,cinputvalues))



print('Plotting ', modelName )
(timeTotal, dataout) = cfs.chemostatExperiment(chemostatinputs)

#-----------------------
# Plot model predictions
#-----------------------

cfs.plotInitialise(4,4)

for i in range(dataout.shape[1]-1):
    plt.plot(timeTotal/60,dataout[:,i+1],'-');

plt.savefig(PATHMODEL+'plotslibrary/modelplotbatch'+modelName+'.pdf')

print('\n\n\n')
