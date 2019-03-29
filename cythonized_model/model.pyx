# Cythonized ODEs from antimony file

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
from libc.math cimport sqrt
from libc.math cimport pow

@cython.cdivision(True) # Zero-division checking turned off
@cython.boundscheck(False) # Bounds checking turned off for this function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def model(np.ndarray[np.float64_t,ndim=1] y, double t, np.ndarray[np.float64_t,ndim=1] params):

	cdef double alpha = y[0]
	cdef double S1 = y[1]
	cdef double S2 = y[2]
	cdef double S3 = y[3]

	cdef double L1 = params[0]
	cdef double k1 = params[1]
	cdef double K = params[2]
	cdef double n = params[3]

	cdef double derivs[4]

	derivs = [
	-alpha*L1,
	alpha*k1*pow(K,n)/(pow(K,n)+pow(S3,n)),
	alpha*k1*pow(K,n)/(pow(K,n)+pow(S1,n)),
	alpha*k1*pow(K,n)/(pow(K,n)+pow(S2,n))]
	return derivs
