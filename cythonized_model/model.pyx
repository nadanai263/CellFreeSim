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

	cdef double A = y[0]
	cdef double B = y[1]
	cdef double C = y[2]
	cdef double alpha = y[3]

	cdef double n = params[0]
	cdef double delta = params[1]
	cdef double L1 = params[2]

	cdef double derivs[4]

	derivs = [
	alpha/(0.1+pow(C,n))-delta*A,
	alpha/(0.1+pow(A,n))-delta*B,
	alpha/(0.1+pow(B,n))-delta*C,
	-L1*alpha]
	return derivs
