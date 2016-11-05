#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

import scipy_extra.stats
import scipy_extra.optimize
import scipy_extra.fit


def plot_data_and_model(title, data, fit_model, extended=False):
    plt.title(title)
    if extended:
        plt.hist(data, bins=100, range=(-3, 3), weights=np.ones(len(data)) * 100.0/6.0)
    else:
        plt.hist(data, bins=100, range=(-3, 3), normed=True)
    plt.plot(X, fit_model.norm * fit_model.frozen_distribution.pdf(X), label=fit_model.name)
    for name, norm, distribution in fit_model.get_frozen_components():
        plt.plot(X, norm * distribution.pdf(X), label=name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X = np.linspace(-3, 3, 1000)

    fit_model = scipy_extra.fit.Model('MyFitModel')
    fit_model.add_component('signal', scipy.stats.norm, loc=0.3, scale=0.5, norm=0.2)
    fit_model.add_component('background', scipy.stats.norm, loc=0.7, scale=1.0, norm=0.3)

    continuum_binned = np.histogram(scipy.stats.norm.rvs(size=100000, loc=0, scale=1.5), bins=100)
    fit_model.add_component('continuum', scipy_extra.stats.template_gen(continuum_binned), norm=0.5)
    
    fit_model.fix_parameter('signal_scale')
    fit_model.fix_parameter('background_scale')
    fit_model.add_constraint('continuum_norm', lambda p: max(1.0 - p['signal_norm'] - p['background_norm'], 0.0))

    data = fit_model.frozen_distribution.rvs(size=10000)
    plot_data_and_model("Model with true parameters", data, fit_model)
    print(fit_model.parameters)

    fit_model.set_parameters(signal_loc=0.5, background_loc=0.5, signal_norm=0.4, background_norm=0.3, continuum_norm=0.3)
    #fit_model.set_parameters(signal_loc=0.5, background_loc=0.5, signal_norm=4000, background_norm=3000, continuum_norm=3000)
    plot_data_and_model("Model Before Fit", data, fit_model)

    #fitter = scipy_extra.fit.Fitter(loss='extended-unbinned-maximum-likelihood')
    fitter = scipy_extra.fit.Fitter(loss='unbinned-maximum-likelihood')
    result, r = fitter.fit(fit_model, data)
    print("Raw scipy result", r)
    print("Result parameters", result)
    fit_model.set_parameters(**result)
    
    plot_data_and_model("Model After Fit", data, fit_model)
  
    signal_yield_generated = np.linspace(0.01, 0.5, 20)
    likelihood = fitter.likelihood_profile(fit_model, {'signal_norm': signal_yield_generated}, data)
    plt.title("Likelihood Profile")
    plt.xlabel("Generated Signal Yield")
    plt.ylabel("- Log Likelihood")
    plt.plot(signal_yield_generated, likelihood)
    plt.show()

    signal_yield_fitted = fitter.stability_test(fit_model, {'signal_norm': signal_yield_generated})['signal_norm']
    plt.title("Stability test")
    plt.xlabel("Generated Signal Yield")
    plt.ylabel("Fitted Signal Yield")
    plt.plot(signal_yield_generated, signal_yield_fitted)
    plt.plot([0.0, 1.0], [0.0, 1.0])
    plt.show()
    print(signal_yield_generated)
    print(signal_yield_fitted)

