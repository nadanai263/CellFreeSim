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
            # 4. Run simulator
            sys.path.insert(0, PATH_OUT)
            import model as model

            y0 = np.array([float(value) for value in speciesValues])
            params = np.array([float(value) for value in parameterValues])

            psoln = odeint(model.model, y0, time, args=(params,)) # scipy-Fortran RK4 solver

            for i in range(psoln.shape[1]):
            	plt.plot(time,psoln[:,i]);
            plt.savefig(PATH_PLOT+'sim.pdf')
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
    print('4. Run simulator')
    print('Press ENTER to exit')
    print()

# Call main function.

if __name__=='__main__':
    main()


