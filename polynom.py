#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

import scipy_extra.stats
import scipy_extra.optimize
import scipy_extra.fit


def plot_data_and_model(title, data, fit_model):
    plt.title(title)
    plt.hist(data, bins=100, range=(0, 5), normed=True)
    plt.plot(X, fit_model.norm * fit_model.frozen_distribution.pdf(X), label=fit_model.name)
    for name, norm, distribution in fit_model.get_frozen_components():
        plt.plot(X, norm * distribution.pdf(X), label=name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X = np.linspace(0, 5, 1000)

    fit_model = scipy_extra.fit.Model('MyFitModel')
    fit_model.add_component('signal', scipy_extra.stats.polynom_1, loc=1.0, scale=2.0, a_0=1.0, a_1=1.0, norm=0.5)
    fit_model.add_component('background', scipy.stats.uniform, loc=2.0, scale=2.0, norm=0.5)

    fit_model.fix_parameter('signal_loc')
    fit_model.fix_parameter('background_loc')
    fit_model.fix_parameter('signal_scale')
    fit_model.fix_parameter('background_scale')
    #fit_model.add_constraint('background_norm', lambda p: 1.0 - p['signal_norm'])

    data = fit_model.frozen_distribution.rvs(size=1000)
    plot_data_and_model("Model with true parameters", data, fit_model)
    print(fit_model.parameters)

    fit_model.set_parameters(signal_loc=1.0, background_loc=2.0, signal_norm=0.4, signal_a_1=0.9)
    #fit_model.set_parameters(signal_norm=0.4)
    plot_data_and_model("Model Before Fit", data, fit_model)

    fitter = scipy_extra.fit.Fitter(loss='extended-unbinned-maximum-likelihood')
    #fitter = scipy_extra.fit.Fitter(loss='unbinned-maximum-likelihood')
    result, r = fitter.fit(fit_model, data)
    print("Raw scipy result", r)
    print("Result parameters", result)
    fit_model.set_parameters(**result)
    
    plot_data_and_model("Model After Fit", data, fit_model)
