#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

import scipy_extra.fit
import scipy_extra.stats
import scipy_extra.plot


if __name__ == '__main__':

    # At first you have to define a scipy distribution which you want to fit to your data
    # A good starting point is the rv_mixture distribution which offers you the possibility to define a mixture of several other scipy.stats distributions
    continuum_hist = np.histogram(scipy.stats.norm.rvs(size=100000, loc=0, scale=1.5), bins=20)
    distribution = scipy_extra.stats.rv_mixture([('signal', scipy.stats.norm),
                                                 ('background', scipy.stats.norm),
                                                 ('continuum', scipy.stats.rv_histogram(continuum_hist))])

    # Secondly you need to define a mapping from the free parameters in the fit to the shape parameters of your distribution.
    # In this way you can incorporate arbitrary constraints in your model.
    def mapping(free_parameters):
        return dict(signal_loc=free_parameters[0],
                    signal_scale=0.5,
                    signal_norm=free_parameters[1],
                    background_loc=free_parameters[2],
                    background_scale=1.0,
                    background_norm=free_parameters[3],
                    continuum_loc=0.0,
                    continuum_scale=1.0,
                    continuum_norm=free_parameters[4])

    # Finally you can define a normalisation function which returns the overall norm of your distribution given the shape parameters. 
    # If you don't define this, one is assumed and the extended ML fit reduced to a usual ML fit.
    def normalisation(parameters):
        return parameters['signal_norm'] + parameters['background_norm'] + parameters['continuum_norm']

    fitter = scipy_extra.fit.Fitter(mapping, distribution, normalisation=normalisation)

    # The true parameters of our model (unknown in a real world example!)
    true_parameters = [0.4, 0.2, 0.6, 0.6, 0.2]
    # We generate some fake data using these true parameters.
    data = distribution.rvs(size=1000, **mapping(true_parameters))

    # And now we fit the fake data using a initial guess for the parameters (different from the true parameters!)
    initial_parameters = [0.1, 0.1, 0.6, 0.3, 0.6]
    r = fitter.fit(initial_parameters, data)
    print("Result")
    print(r)

    # After fitting we can get the uncertainty from the likelihood profile.
    # You have to pass the boundaries of all parameters you want the uncertainty for,
    # here we are only interested in the normalisations, so we pass None for the boundary of the location parameters.
    uncertainties = fitter.get_uncertainties([None, [0.0, 1.0], None, [0.0, 1.0], [0.0, 1.0]], data)
    print("Uncertainties")
    print(uncertainties)

    # In a similar way we can obtain the significance of the fit with respect to a null hypothesis.
    # Here we set the signal_norm to 0 and leave all other parameters free in the fit.
    significance = fitter.get_significance([None, 0.0, None, None, None, None], data)
    print("Significance")
    print(significance)

    # We can also directly calculate the likelihood profile
    profile_values = [None, np.linspace(0, 1, 21), None, None, None]
    likelihood_profile = fitter.likelihood_profile(profile_values, data)
    print("Likelihood Profile")
    print(likelihood_profile)

    # There are some pre-defined plotting routines available.
    # I recommend to copy the plotting code and adapt it to your requirements (naturally there are a lot of things you might want to change)
    plt.title("True parameters")
    scipy_extra.plot.fit(fitter, data, free_parameters=true_parameters)
    plt.show()

    plt.title("Initial parameters")
    scipy_extra.plot.fit(fitter, data, free_parameters=initial_parameters)
    plt.show()

    plt.title("Fitted parameters")
    scipy_extra.plot.fit(fitter, data)
    plt.show()

    plt.title("Likelihood Profile")
    scipy_extra.plot.likelihood_profile(fitter, profile_values, likelihood_profile, 1)
    plt.show()

    # Finally we can do some toy experiments
    # For example to estimate the stability of the fit we perform the fit multiple times with different signal norm values
    toy_stability_parameters = [[true_parameters[0], x, true_parameters[2], (1-x)*0.75, (1-x)*0.25] for x in np.repeat(np.linspace(0, 0.5, 11), 20)]
    toy_stability_results = fitter.toy(initial_parameters, toy_stability_parameters, 1000)
    scipy_extra.plot.stability(fitter, toy_stability_results, 1)
    plt.show()

    # Or we can calculate the pull distribution for our fit by performing the fit several times using the same parameters
    # This will take some time, because for the pull we have to calculate the uncertainty of each toy experiment.
    toy_pull_parameters = [true_parameters]*100
    toy_pull_results = fitter.toy(initial_parameters, toy_pull_parameters, 1000, [None, [0.0, 1.0], None, None, None])
    scipy_extra.plot.pull(fitter, toy_pull_results, 1)
    plt.show()

