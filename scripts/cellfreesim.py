# cellfreesim.py
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
    from config import *
except:
    print('No config file! Please make one and re-run. Examples can be found on the GitHub repo.')
    sys.exit()

if not os.path.exists(PATH_OUT):
    os.mkdir('../cythonized_model/')


def main():
    print()
    print(pyfiglet.figlet_format("LBNC - EPFL"))
    print('Welcome to the LBNC Cell Free Simulator V1.0.0, 2019')
    print('Make sure you have set required parameters in the config file')
    print()

    ### Interactive main menu

    while True:
        main_menu()

        commands_str = input('Input: ')

        if commands_str=='1':
            # 1. Load model from antimony file
            print('Loading model')
            r = te.loada(PATH_IN+FILENAME_IN)
            odes = te.getODEsFromModel(r) # Standardised ODE parser from tellurium 2.1.5
            speciesIds, speciesValues, parameterIds, parameterValues, derivatives = cfs.parseODEs(r,odes)
            print(odes)
            print('\n\n\n')

        elif commands_str=='2':
            # 2. Generate cython file
            print('Generating cython file')

            cfs.writeCython(speciesIds,speciesValues,parameterIds,parameterValues,derivatives,PATH_OUT,FILENAME_OUT)
            print('\n\n\n')

        elif commands_str=='3':
            # 3. Cythonize
            os.chdir(PATH_OUT)
            os.system('python3 setup.py build_ext --inplace')
            print('\n\n\n')

        elif commands_str=='4':
            # 4. Run batch experiment
            sys.path.insert(0, PATH_OUT)
            import model as model

            y0 = np.array([float(value) for value in speciesValues])
            params = np.array([float(value) for value in parameterValues])

            time = np.linspace(0,TMAX,NSTEPS)
            psoln = odeint(model.model, y0, time, args=(params,)) # scipy-Fortran RK4 solver

            for i in range(psoln.shape[1]):
                plt.plot(time,psoln[:,i]);
            plt.savefig(PATH_PLOT+'sim.pdf')
            plt.show();
            print('\n\n\n')

        elif commands_str=='5':
            # 5. Run chemostat experiment

            sys.path.insert(0, PATH_OUT)
            import model as model

            y0 = np.array([float(value) for value in speciesValues])
            params = np.array([float(value) for value in parameterValues])

            # TO BE UPDATED:
            if TMAX%INTERVAL_DIL!=0:
                print('\n')
                print('TMAX is not divisible by INTERVAL_DIL!\n')
                print('Inaccurate results expected!\n')

            if INTERVAL_DIL%INTERVAL_IMG!=0:
                print('\n')
                print('INTERVAL_DIL is not divisible by INTERVAL_IMG!\n')
                print('Inaccurate results expected!\n')

            cinputkeys = ['dilutiontimes', 'y0', 'params', 'interval_img', 'dil_frac', 'index_refresh', 'cymodel']
            cinputvalues = [dilutiontimes, y0, params, INTERVAL_IMG, DIL_FRAC, INDEX_REFRESH, model.model]
            chemostatinputs = dict(zip(cinputkeys,cinputvalues))

            timeTotal,dataout = cfs.chemostatExperiment(chemostatinputs)

            for i in range(dataout.shape[1]):
                plt.plot(timeTotal,dataout[:,i],'o-');

            # Compare with batch
            time = np.linspace(0,TMAX,nstepschemos)
            psoln = odeint(model.model, y0, time, args=(params,)) # scipy-Fortran RK4 solver

            for i in range(psoln.shape[1]):
                plt.plot(time,psoln[:,i]);

            plt.savefig(PATH_PLOT+'chemostat.pdf')
            plt.show();
            print('\n\n\n')

        elif commands_str=='6':
            # 6. Run MCMC experiment

            sys.path.insert(0, PATH_OUT)
            import model as model

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

            cinputkeys = ['dilutiontimes', 'y0', 'params', 'interval_img', 'dil_frac', 'index_refresh', 'cymodel']
            cinputvalues = [dilutiontimes, y0, params, INTERVAL_IMG, DIL_FRAC, INDEX_REFRESH, model.model]
            chemostatinputs = dict(zip(cinputkeys,cinputvalues))

            # Generate silico data
            timeTotal,dataout = cfs.chemostatExperiment(chemostatinputs)
            mcmc_inputs = {}
            mcmc_inputs['data'] = dataout[:,DATACHANNEL]
            mcmc_inputs['priorMuSigma'] = PMUSIGMA
            mcmc_inputs['yerr'] = SIGMA
            mcmc_inputs['datachannel'] = DATACHANNEL
            mcmc_inputs['paramchannels'] = PARAMCHANNELS

            ##### The rest of the code is automatic #####
            sampler=emcee.EnsembleSampler(nwalkers,nDimParams,cfs.lnprob,a=2,args=([chemostatinputs, mcmc_inputs]),threads=threads)

            ### Start MCMC
            iter=iterations
            bar=progressbar.ProgressBar(max_value=iter)
            for i, result in enumerate(sampler.sample(pos, iterations=iter)):
                bar.update(i)
            ### Finish MCMC

            samples=sampler.chain[:,:,:].reshape((-1,nDimParams)) # shape = (nsteps, nDimParams)
            samplesnoburn=sampler.chain[:,tburn:,:].reshape((-1,nDimParams)) # shape = (nsteps, nDimParams)

#            import pandas as pd
#            df=pd.DataFrame(samples)
#            df.to_csv(path_or_buf=plots_dir+'samplesout_'+filename+'.csv',sep=',') # UNCOMMENT TO SAVE OUTPUT

            #-----------------------
            # Plot model predictions
            #-----------------------

            for i in range(dataout.shape[1]):
                plt.plot(timeTotal,mcmc_inputs['data'],'o-');

            for lnparam0, lnparam1 in samplesnoburn[np.random.randint(len(samplesnoburn), size=20)]:
                # Unpack and repack params
                paramstmp = chemostatinputs['params']
                paramstmp[int(mcmc_inputs['paramchannels'][0])] = np.exp(lnparam0)
                paramstmp[int(mcmc_inputs['paramchannels'][1])] = np.exp(lnparam1)
                chemostatinputs['params'] = paramstmp
                sim_timeTotal,sim_dataout = cfs.chemostatExperiment(chemostatinputs)
                plt.plot(sim_timeTotal,sim_dataout[:,int(mcmc_inputs['datachannel'])],'k-', alpha=0.1)
            plt.savefig(PATH_PLOT+'MCMC_preds.pdf')
            plt.show();

            #-----------------------
            # Plot corner
            #-----------------------

            plt.close("all");
            fig=corner.corner(samplesnoburn, labels=labels,truths=parametertruths,quantiles=[0.16, 0.5, 0.84],show_titles=True, title_fmt='.2e', title_kwargs={"fontsize": 10},verbose=False)
            plt.savefig(PATH_PLOT+'MCMC_triangle.pdf')
            plt.show();
            print('\n\n\n')

        elif commands_str=='':
            # Press ENTER to exit without saving
            print()
            print('Goodbye!')
            print()
            break


def main_menu():
    print()
    print('Please select from the following options by typing in the number, followed by ENTER: ')
    print()
    print('1. Load model, parse ODEs and parameters')
    print('2. Generate cython file')
    print('3. Cythonize!')
    print('4. Run batch experiment')
    print('5. Run chemostat experiment')
    print('6. Run MCMC experiment')
    print('Press ENTER to exit')
    print()

# Call main function.

if __name__=='__main__':
    main()
