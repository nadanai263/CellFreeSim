# libraryMCMC.py
# Runs MCMC on entire model library

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
    from configMCMClib import *
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

DATACHANNELS = []
for j in range(len(speciesValues)):
    DATACHANNELS.append(int(j))

PARAMCHANNELS = []
for j in range(len(parameterValues)):
    PARAMCHANNELS.append(int(j))

INDEX_REFRESH = [0]
CONC_REFRESH = [1]

labels=[label for label in parameterIds]

parametertruths=[np.log(value) for value in parameterValues]
pos=[np.array([initpos*(1+0.1*np.random.randn()) for initpos in parametertruths])
        for i in range(nwalkers)]
PMUSIGMA = [[mu for mu in parametertruths],[2 for i in range(len(parametertruths))]]

nDimParams = len(parameterValues)

# Reimport updated model
sys.path.insert(0, PATH_OUT)
import model as model

print('Running MCMC for ', modelName )
(samplesnoburn, chemostatinputs, mcmc_inputs, timeTotal, dataout) = cfs.runMCMC(speciesValues,parameterValues,
                                                                    TMAX,INTERVAL_DIL,INTERVAL_IMG,
                                                                    dilutiontimes,DIL_FRAC,INDEX_REFRESH,CONC_REFRESH,model,
                                                                    DATACHANNELS,PARAMCHANNELS,PMUSIGMA,SIGMA,
                                                                    iterations,nwalkers,nDimParams,threads,pos,tburn)

#-----------------------
# Plot model predictions
#-----------------------

cfs.plotInitialise(4,4)

for i in range(len(mcmc_inputs['data'])-1):
    plt.plot(timeTotal/60,mcmc_inputs['data'][i+1],'o');


inds = np.random.randint(len(samplesnoburn), size=NMODELRUNS)
for ind in inds:
    lnparams = [param for param in samplesnoburn[ind]]
    # Unpack and repack params
    paramstmp = chemostatinputs['params']
    for i in range(len(mcmc_inputs['paramchannels'])):
        paramstmp[int(mcmc_inputs['paramchannels'][i])] = np.exp(lnparams[i])
    chemostatinputs['params'] = paramstmp
    sim_timeTotal,sim_dataout = cfs.chemostatExperiment(chemostatinputs)
    for j in range(len(mcmc_inputs['datachannels'])-1):
        plt.plot(sim_timeTotal/60,sim_dataout[:,int(mcmc_inputs['datachannels'][j+1])],'k-', alpha=0.1)
plt.savefig(PATHMODEL+'plotsMCMC/MCMC_preds'+modelName+'.pdf')


#-----------------------
# Plot corner
#-----------------------

plt.close('all')
fig=corner.corner(samplesnoburn, labels=labels,truths=parametertruths,quantiles=[0.16, 0.5, 0.84],show_titles=True, title_fmt='.2e', title_kwargs={"fontsize": 10},verbose=False)
plt.savefig(PATHMODEL+'plotsMCMC/MCMC_triangle'+modelName+'.pdf')

print('\n\n\n')
