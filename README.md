# CellFreeSim
Interactive, fast cythonized ODE solver which accepts Antimony/SBML model files as inputs.

Used at the [LBNC](http://lbnc.epfl.ch/)-[EPFL](https://www.epfl.ch/) to model biochemical reactions for cell-free synthetic biology. These scripts convert input models (in [Antimony](https://github.com/sys-bio/antimony) or [SBML](http://sbml.org/Main_Page) format) into fast cythonized code, which is simulated using scipy-odeint. Model definitions and simulation parameters are easily set using a configuration file. The script outputs timeseries data and plots for all species. The compiled cython function can additionally be used in downstream processes such as MCMC which require fast simulations. Inspired by the Murray lab's simulators (in [python](https://github.com/murrayrm/txtlsim-python) and [matlab](https://github.com/murrayrm/txtlsim)), as well as the ease of use of [tellurium](https://github.com/sys-bio/tellurium)-[roadrunner](https://github.com/sys-bio/roadrunner). 
