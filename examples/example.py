#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

import scipy_extra.fit
import scipy_extra.stats


def plot_data_and_model(title, data, frozen_distribution, normalisation):
    plt.title(title)
    binning = dict(range=(-3, 3), bins=100)
    space = np.linspace(*binning['range'], binning['bins']*10)
    content, boundaries = np.histogram(data, **binning)
    plt.errorbar((boundaries[1:] + boundaries[:-1]) / 2, content, yerr=np.sqrt(content), color='black', fmt='s', markersize=8, label='Data', zorder=3)
    
    weight = len(data) / binning['bins'] * (binning['range'][1] - binning['range'][0])

    plt.fill_between(space, weight * normalisation(frozen_distribution.kwds) * frozen_distribution.pdf(space), label='Fit', color='gray')
    for name, distribution, norm_name, shape_names in zip(frozen_distribution.dist._components,
                                                          frozen_distribution.dist._distributions,
                                                          frozen_distribution.dist._distribution_norms,
                                                          frozen_distribution.dist._distribution_shapes):
        norm = frozen_distribution.kwds[norm_name]
        shapes = {'_'.join(k.split('_')[1:]) : frozen_distribution.kwds[k] for k in shape_names}
        plt.plot(space, weight * norm * distribution.pdf(space, **shapes), label=name)
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xlim(binning['range'])
    plt.show()


if __name__ == '__main__':

    continuum_hist = np.histogram(scipy.stats.norm.rvs(size=100000, loc=0, scale=1.5), bins=100)
    distribution = scipy_extra.stats.rv_mixture([('signal', scipy.stats.norm),
                                                 ('background', scipy.stats.norm),
                                                 ('continuum', scipy.stats.rv_histogram(continuum_hist))])

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

    def normalisation(parameters):
        return parameters['signal_norm'] + parameters['background_norm'] + parameters['continuum_norm']

    fitter = scipy_extra.fit.Fitter(mapping, distribution, normalisation=normalisation)

    true_parameters = [0.4, 0.2, 0.6, 0.6, 0.2]
    initial_parameters = [0.1, 0.1, 0.6, 0.3, 0.6]
    data = distribution.rvs(size=10000, **mapping(true_parameters))

    r = fitter.fit(initial_parameters, data)
    print("Result")
    print(r)

    uncertainties = fitter.get_uncertainties([0,3,4], [[0.0, 1.0]]*3, data)
    print("Uncertainties")
    print(uncertainties)

    significance = fitter.get_significance([0], [0.0], data)
    print("Significance")
    print(significance)

    def true_parameters_generator():
        for x in np.linspace(0, 1, 10):
            yield true_parameters[:1] + [x] + true_parameters[2:]
    result = fitter.stability_test(initial_parameters, true_parameters_generator(), 10000)
    print("Stability Test")
    for generated, fitted in zip(true_parameters_generator(), result):
        print(generated[1], fitted.x[1])
   
    print("Plotting")
    plot_data_and_model("Model with true parameters", data, distribution(**mapping(true_parameters)), normalisation)
    plot_data_and_model("Model with initial parameters", data, distribution(**mapping(initial_parameters)), normalisation)
    plot_data_and_model("Model with fitted parameters", data, distribution(**mapping(r.x)), normalisation)

