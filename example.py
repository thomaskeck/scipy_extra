#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

import scipy_extra.stats
import scipy_extra.optimize
import scipy_extra.fit


def plot_data_and_model(data, fit_model):
    return
    plt.hist(data, bins=100, range=(-3, 3), normed=True)
    plt.plot(X, fit_model.frozen_distribution.pdf(X), label=fit_model.name)
    for name, distribution in fit_model.get_frozen_components():
        plt.plot(X, distribution.pdf(X), label=name)
    plt.show()


if __name__ == '__main__':
    X = np.linspace(-3, 3, 1000)

    fit_model = scipy_extra.fit.Model('MyFitModel')
    fit_model.add_component('signal', scipy.stats.norm, loc=0.2, scale=0.1, norm=0.1)
    fit_model.add_component('background', scipy.stats.norm, loc=0.8, scale=1.0, norm=0.1)

    continuum_binned = np.histogram(scipy.stats.norm.rvs(size=100000, loc=0, scale=2.0), bins=100)
    fit_model.add_component('continuum', scipy_extra.stats.template_gen(continuum_binned), norm=0.8)
    
    fit_model.fix_parameter('signal_scale')
    fit_model.fix_parameter('background_scale')
    fit_model.add_constraint('continuum_norm', lambda p: 1.0 - p['signal_norm'] - p['background_norm'])

    data = fit_model.frozen_distribution.rvs(size=10000)
    plot_data_and_model(data, fit_model)
    print(fit_model.parameters)

    fit_model.set_parameters(signal_loc=0.5, background_loc=0.5, signal_norm=0.4, background_norm=0.3, continuum_norm=0.3)
    fitter = scipy_extra.fit.Fitter(method=scipy_extra.optimize.Minuit)
    result, r = fitter.fit(fit_model, data)
    print("Raw scipy result", r)
    print("Result parameters", result)
    fit_model.set_parameters(**result)
    
    plot_data_and_model(data, fit_model)

