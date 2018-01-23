# scipy_extra
Implementation of a maximum-likelihood fit framework based on scipy

This repository contains custom extensions to scipy.
I usually try to push parts of the code into the official scipy and afterwards remove them from this package.

## stats.py
Contains extensions to scipy.stats.
Additional distributions which are used in HEP (Crystalball, ARGUS, Chebyshev polynomials, Mixture models, ...)

## optimize.py
Contains an interface to the MINUIT minimizer used by a HEP data analysis framework called ROOT.

## fit.py
Contains a maximum-likelihood fit framework, which I used during my PhD thesis to extract the branching fraction of B to tau nu.
The fit framework does support simultaneous fits in multiple channels and constraints.
Multi-dimensional fits possible in principle, but this requires using multi-dimensional scipy distributions, which you probably have to write yourself.
The performance of the fit depends heavily on the used scipy distributions. Hence, if it is too slow, you may want to speed up the scipy distributions.
The framework does support the extraction of uncertainties from the log-likelihood and can do toy-experiments.

## plot.py
Contains some predefined plotting routines for common performance plots like stability tests, pull distributions or likelihood profiles.
