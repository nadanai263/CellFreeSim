# cellfreesim functions.py

import matplotlib.pyplot as plt
import tellurium as te # 2.1.5
import numpy as np


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