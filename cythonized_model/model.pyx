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

	cdef double m = y[0]
	cdef double p = y[1]
	cdef double pmat = y[2]
	cdef double alpha = y[3]
	cdef double beta = y[4]

	cdef double gamma = params[0]
	cdef double delta = params[1]
	cdef double kmat = params[2]
	cdef double L1 = params[3]
	cdef double L2 = params[4]

	cdef double derivs[5]

	derivs = [
	alpha-gamma*m,
	beta*m-delta*p-kmat*p,
	kmat*p-delta*pmat,
	-L1*alpha,
	-L2*beta]
	return derivs
