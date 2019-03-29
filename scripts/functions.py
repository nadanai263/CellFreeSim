# cellfreesim functions.py

import matplotlib.pyplot as plt
import tellurium as te # 2.1.5
import numpy as np
import sys
import progressbar as progressbar
import emcee
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

    print(derivdict)
    print(channeldict)

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
    CONC_REFRESH = chemostatinputs['conc_refresh']
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

        j=0
        for ind in INDEX_REFRESH:
            yTransfer[ind] = yTransfer[ind]+DIL_FRAC*CONC_REFRESH[j]
            j+=1

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

    lnparams = [j for j in theta]

    # This has to be an array if more than one parameter, otherwise just a float
    paramstmp = chemostatinputs['params']
    for i in range(len(mcmc_inputs['paramchannels'])):
        paramstmp[int(mcmc_inputs['paramchannels'][i])] = np.exp(lnparams[i])
    chemostatinputs['params'] = paramstmp
    timeTotal,sim_dataout = chemostatExperiment(chemostatinputs)

    X0s = []
    for j in range(len(mcmc_inputs['datachannels'])):
        y_obs = mcmc_inputs['data'][j]
        y_model = sim_dataout[:,int(mcmc_inputs['datachannels'][j])]
        INVS2 = 1/mcmc_inputs['yerr']**2
        X0=-0.5*(np.sum((y_obs-y_model)**2*INVS2+np.log(2*np.pi*1/INVS2)))
        X0s.append(X0)

    return sum(X0s)

def lnprior(theta, mcmc_inputs):
    ''' Log priors'''
    lnparams = [j for j in theta]

    priorMus = mcmc_inputs['priorMuSigma'][0]
    priorSigmas = mcmc_inputs['priorMuSigma'][1]

    log_PRs = []
    for j in range(len(lnparams)):
        log_PRs.append(normalprior(lnparams[j],priorMus[j],priorSigmas[j]))
    return np.sum(log_PRs)

def lnprob(theta,chemostatinputs, mcmc_inputs):
    ''' Log posterior'''
    lp = lnprior(theta, mcmc_inputs)
    if not np.isfinite(lp):
        return -np.inf
    # How to properly account for NaNs turning up in lnlike?
    # This is NOT the way to do it:
    if np.isnan(lp + lnlike(theta,chemostatinputs, mcmc_inputs)):
        return -np.inf
    else:
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

def runMCMC(speciesValues,parameterValues,TMAX,INTERVAL_DIL,INTERVAL_IMG,
            dilutiontimes,DIL_FRAC,INDEX_REFRESH,CONC_REFRESH,model,
            DATACHANNELS,PARAMCHANNELS,PMUSIGMA,SIGMA,
            iterations,nwalkers,nDimParams,threads,pos,tburn):

    y0 = np.array([float(value) for value in speciesValues])
    params = np.array([float(value) for value in parameterValues])

    if TMAX%INTERVAL_DIL!=0:
        print('\n')
        print('TMAX is not divisible by INTERVAL_DIL!\n')
        print('Inaccurate results expected!\n')

    if INTERVAL_DIL%INTERVAL_IMG!=0:
        print('\n')
        print('INTERVAL_DIL is not divisible by INTERVAL_IMG!\n')
        print('Inaccurate results expected!\n')

    cinputkeys = ['dilutiontimes', 'y0', 'params', 'interval_img', 'dil_frac', 'index_refresh', 'conc_refresh','cymodel']
    cinputvalues = [dilutiontimes, y0, params, INTERVAL_IMG, DIL_FRAC, INDEX_REFRESH, CONC_REFRESH, model.model]
    chemostatinputs = dict(zip(cinputkeys,cinputvalues))

    # Generate silico data
    timeTotal,dataout = chemostatExperiment(chemostatinputs)
    mcmc_inputs = {}
    mcmc_inputs['data'] = [dataout[:,channel] for channel in DATACHANNELS]
    mcmc_inputs['priorMuSigma'] = PMUSIGMA
    mcmc_inputs['yerr'] = SIGMA
    mcmc_inputs['datachannels'] = DATACHANNELS
    mcmc_inputs['paramchannels'] = PARAMCHANNELS


    ##### The rest of the code is automatic #####
    sampler=emcee.EnsembleSampler(nwalkers,nDimParams,lnprob,a=2,args=([chemostatinputs, mcmc_inputs]),threads=threads)

    ### Start MCMC
    iter=iterations
    bar=progressbar.ProgressBar(max_value=iter)
    for i, result in enumerate(sampler.sample(pos, iterations=iter)):
        bar.update(i)
    ### Finish MCMC

    samples=sampler.chain[:,:,:].reshape((-1,nDimParams)) # shape = (nsteps, nDimParams)
    samplesnoburn=sampler.chain[:,tburn:,:].reshape((-1,nDimParams)) # shape = (nsteps, nDimParams)

    return(samplesnoburn, chemostatinputs, mcmc_inputs, timeTotal, dataout)


# Plotting

def plotInitialise(figW,figH):

    plt.close("all")
    figure_options={'figsize':(figW,figH)} # figure size in inches. A4=11.7x8.3, A5=8.3,5.8
    font_options={'size':'14','family':'sans-serif','sans-serif':'Arial'}
    plt.rc('figure', **figure_options)
    plt.rc('font', **font_options)

def plotFormat(ax,xlabel=False,
                    ylabel=False,
                    xlim=False,
                    ylim=False,
                    title=False,
                    xticks=False,
                    yticks=False,
                    logx=False,
                    logy=False,
                    logxy=False,
                    symlogx=False,
                    legend=False):

    # Set titles and labels
    if title!=False:
        ax.set_title(title)
    if xlabel!=False:
        ax.set_xlabel(xlabel, labelpad=12)
    if ylabel!=False:
        ax.set_ylabel(ylabel, labelpad=12)

    # Set axis limits
    if xlim!=False:
        ax.set_xlim(xlim)
    if ylim!=False:
        ax.set_ylim(ylim)

    # Set tick values
    if xticks!=False:
        ax.set_xticks(xticks)
    if yticks!=False:
        ax.set_yticks(yticks)

    # Set line thicknesses
    #ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%1.e"))
    #ax.axhline(linewidth=2, color='k')
    #ax.axvline(linewidth=2, color='k')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    # Set ticks
    if logx==True:
        ax.set_xscale("log")

    elif logy==True:
        ax.set_yscale("log")

    elif logxy==True:
        ax.set_xscale("log")
        ax.set_yscale("log")

    elif symlogx==True:
        ax.set_xscale("symlog",linthreshx=1e-4)
        ax.set_yscale("log")

    else:
        minorLocatorx=AutoMinorLocator(2) # Number of minor intervals per major interval
        minorLocatory=AutoMinorLocator(2)
        ax.xaxis.set_minor_locator(minorLocatorx)
        ax.yaxis.set_minor_locator(minorLocatory)

    ax.tick_params(which='major', width=2, length=8, pad=9,direction='in',top='on',right='on')
    ax.tick_params(which='minor', width=2, length=4, pad=9,direction='in',top='on',right='on')

    if legend==True:
        ax.legend(loc='upper right', fontsize=14,numpoints=1) ### Default 'best'
