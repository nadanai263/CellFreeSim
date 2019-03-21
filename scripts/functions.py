# cellfreesim functions.py

import matplotlib.pyplot as plt
import tellurium as te # 2.1.5
import numpy as np
import sys
from scipy.integrate import odeint

#-----------------------------------------------------------------------------
# Define parsing functions
#-----------------------------------------------------------------------------

def parseODEs(r,odes):

    # Parsing of ODEs into cython code

    # Split odes into channels and derivatives (normally these separated by two spaces)
    parts = odes.split('\n\n')
    channels = parts[0].lstrip('\n').split('\n')
    derivs = parts[1].rstrip('\n').split('\n')

    channeldict = {}
    for channel in channels:
        channeldict[channel.split(' = ')[0]] = channel.split(' = ')[1]

    derivdict = {}
    for deriv in derivs:
        derivdict[deriv.split(' = ')[0]] = deriv.split(' = ')[1]

    speciesIds = []
    derivatives = []
    for derivkey in derivdict.keys():
        speciesIds.append(derivkey[1:-3]) # Hardcoded d/dt

        channelkey = derivdict[derivkey]

        if channelkey[0]=='-':
            derivatives.append('-'+channeldict[channelkey[1:]])
        else:
            derivatives.append(channeldict[channelkey])

    speciesValues = r.getFloatingSpeciesConcentrations()
    parameterIds = r.getGlobalParameterIds()
    parameterValues = [value for value in r.getGlobalParameterValues()]

    return(speciesIds, speciesValues, parameterIds, parameterValues, derivatives)


def writeCython(speciesIds,speciesValues,parameterIds,parameterValues,derivatives,OUTPATH,FILENAME):
    # Generate cython script

    with open(OUTPATH+FILENAME,'w') as file:
        file.writelines('# Cythonized ODEs from antimony file\n\n')

        # Imports
        file.writelines('import numpy as np\n')
        file.writelines('cimport numpy as np\n')
        file.writelines('cimport cython\n')
        file.writelines('from libc.math cimport exp\n')
        file.writelines('from libc.math cimport sqrt\n')
        file.writelines('from libc.math cimport pow\n\n')

        # Model definition
        file.writelines('@cython.cdivision(True) # Zero-division checking turned off\n')
        file.writelines('@cython.boundscheck(False) # Bounds checking turned off for this function\n')
        file.writelines('@cython.wraparound(False)  # turn off negative index wrapping for entire function\n')
        file.writelines('def model(np.ndarray[np.float64_t,ndim=1] y, double t, np.ndarray[np.float64_t,ndim=1] params):\n\n')

        # Species
        for i in range(len(speciesIds)):
            file.write('\tcdef double '+speciesIds[i]+' = y['+str(i)+']\n')

        file.writelines('\n')
        for i in range(len(parameterIds)):
            file.write('\tcdef double '+parameterIds[i]+' = params['+str(i)+']\n')

        file.writelines('\n')
        file.writelines('\tcdef double derivs['+str(len(derivatives))+']\n')

        file.writelines('\n')
        file.writelines('\tderivs = [\n')
        for i in range(len(derivatives)-1):
            file.write('\t'+derivatives[i]+',\n')
        file.write('\t'+derivatives[len(derivatives)-1]+']\n')

        file.write('\treturn derivs\n')


    file.close()

#-----------------------------------------------------------------------------
# Define experiment functions
#-----------------------------------------------------------------------------

def chemostatExperiment(chemostatinputs):

    # Run chemostat experiment

    dilutiontimes = chemostatinputs['dilutiontimes']
    y0 = chemostatinputs['y0']
    params = chemostatinputs['params']
    INTERVAL_IMG = chemostatinputs['interval_img']
    DIL_FRAC = chemostatinputs['dil_frac']
    INDEX_REFRESH = chemostatinputs['index_refresh']
    cymodel = chemostatinputs['cymodel']

    ndim = y0.shape[0]

    # 1. From dilutiontimes, calculate time interval between dilution steps
    interval_dil=np.zeros(dilutiontimes.shape[0]-1)
    for i in range(dilutiontimes.shape[0]-1):
        interval_dil[i] = dilutiontimes[i+1]-dilutiontimes[i]

    # 2. Calulate number of steps in each time interval (depends on imaging frequency)
    nStepsPerRun = np.zeros(dilutiontimes.shape[0]-1)
    for i in range(dilutiontimes.shape[0]-1):
        nStepsPerRun[i]=int(interval_dil[i]/INTERVAL_IMG)+1

    # 3. Put time intervals together to make total time axis, and initialise output array
    timeProgram = {}
    timeTotal = np.zeros(1)
    for i in range(len(dilutiontimes)-1):
        timeProgram[i] = np.linspace(0,interval_dil[i],nStepsPerRun[i])
        timeTotal=np.concatenate([timeTotal,timeProgram[i][1:]+timeTotal[-1]],axis=0)
    dataout=np.zeros(len(timeTotal)*ndim).reshape(len(timeTotal),ndim)

    indStart = int(1)
    yTransfer = y0
    for i in range(len(dilutiontimes)-1):

        psoln = odeint(cymodel, yTransfer, timeProgram[i], args=(params,),mxstep=5000000) # scipy-Fortran RK4 solver
        indStop= indStart+int(nStepsPerRun[i]-1)
        dataout[indStart:indStop,:]=psoln[1:]
        indStart = indStop

        # Dilute everything and refresh appropriate species
        yTransfer = psoln[-1,:]*(1-DIL_FRAC)
        for ind in INDEX_REFRESH:
            yTransfer[ind] = yTransfer[ind]+DIL_FRAC*1

        dataout[0,:] = y0

    return(timeTotal, dataout)

#-----------------------------------------------------------------------------
# Define MCMC functions
#-----------------------------------------------------------------------------

def normalprior(param,mu,sigma):
    '''Log of the normal prior'''
    return np.log( 1.0 / (np.sqrt(2*np.pi)*sigma) ) - 0.5*(param - mu)**2/sigma**2

def lnlike(theta, chemostatinputs, mcmc_inputs):
    ''' Log likelihood, the function to maximise'''

    # We will infer experimental errors
    lnparam0, lnparam1 = theta

    # This has to be an array if more than one parameter, otherwise just a float
    paramstmp = chemostatinputs['params']
    paramstmp[int(mcmc_inputs['paramchannels'][0])] = np.exp(lnparam0)
    paramstmp[int(mcmc_inputs['paramchannels'][1])] = np.exp(lnparam1)
    chemostatinputs['params'] = paramstmp

    timeTotal,sim_dataout = chemostatExperiment(chemostatinputs)

    y_obs = mcmc_inputs['data']
    y_model = sim_dataout[:,int(mcmc_inputs['datachannel'])]
    INVS2 = 1/mcmc_inputs['yerr']**2

    X0=-0.5*(np.sum((y_obs-y_model)**2*INVS2+np.log(2*np.pi*1/INVS2)))

    return X0

def lnprior(theta, mcmc_inputs):
    lnparam0, lnparam1 = theta

    priorMus = mcmc_inputs['priorMuSigma'][0]
    priorSigmas = mcmc_inputs['priorMuSigma'][1]

    log_PRs=[
            normalprior(lnparam0,priorMus[0],priorSigmas[0]),
            normalprior(lnparam1,priorMus[1],priorSigmas[1])
            ]
    return np.sum(log_PRs)

def lnprob(theta,chemostatinputs, mcmc_inputs):
    lp = lnprior(theta, mcmc_inputs)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,chemostatinputs, mcmc_inputs)

def gelman_rubin(chain):
    ''' Gelman-Rubin diagnostic for one walker across all parameters. This value should tend to 1. '''
    ssq=np.var(chain,axis=1,ddof=1)
    W=np.mean(ssq,axis=0)
    Tb=np.mean(chain,axis=1)
    Tbb=np.mean(Tb,axis=0)
    m=chain.shape[0]*1.0
    n=chain.shape[1]*1.0
    B=n/(m-1)*np.sum((Tbb-Tb)**2,axis=0)
    varT=(n-1)/n*W+1/n*B
    Rhat=np.sqrt(varT/W)
    return Rhat
