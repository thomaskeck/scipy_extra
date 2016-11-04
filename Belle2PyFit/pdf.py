#!/usr/bin/env python3

import copy
import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt


class ScipyDistributionGenerator(scipy.stats.rv_continuous):
    def __init__(self, fit_pdf):
        self.fit_pdf = fit_pdf
        shapes = ', '.join(self.fit_pdf.parameters.keys())
        super(ScipyDistributionGenerator, self).__init__(shapes=shapes)

    def _pdf(self, x, *parameters):
        parameters = {key: parameter for key, parameter in zip(self.fit_pdf.parameters.keys(), parameters)}
        return np.sum(parameters['{}_norm'.format(label)] *
                      distribution.pdf(x, **{key[len(label)+1:]: parameters[key] for key in parameters.keys() if key.startswith(label) and key != '{}_norm'.format(label)})
                      for label, distribution in zip(self.fit_pdf.labels, self.fit_pdf.distributions))
    
    def _cdf(self, x, *parameters):
        parameters = {key: parameter for key, parameter in zip(self.fit_pdf.parameters.keys(), parameters)}
        return np.sum(parameters['{}_norm'.format(label)] *
                      distribution.cdf(x, **{key[len(label)+1:]: parameters[key] for key in parameters.keys() if key.startswith(label) and key != '{}_norm'.format(label)})
                      for label, distribution in zip(self.fit_pdf.labels, self.fit_pdf.distributions))
    
    def _rvs(self, *parameters):
        parameters = {key: parameter for key, parameter in zip(self.fit_pdf.parameters.keys(), parameters)}
        probabilities = np.array([parameters['{}_norm'.format(label)] for label in self.fit_pdf.labels])
        probabilities /= probabilities.sum()
        choices = np.random.choice(len(self.fit_pdf.distributions), size=self._size, p=probabilities)
        result = np.zeros(self._size)
        for i, (label, distribution) in enumerate(zip(self.fit_pdf.labels, self.fit_pdf.distributions)):
            mask = choices == i
            component_parameters = {key[len(label)+1:]: parameters[key] for key in parameters.keys() if key.startswith(label) and key != '{}_norm'.format(label)}
            print(distribution, component_parameters)
            result[mask] = distribution.rvs(size=mask.sum(), **component_parameters)
        return result

    def _argcheck(self, *args):
        return 1


class ScipyTemplateDistribution(scipy.stats.rv_continuous):
    def __init__(self, data, **kwargs):
        pdf, bins = np.histogram(data, density=True, **kwargs)
        cdf = np.cumsum(pdf * (bins - np.roll(bins, 1))[1:])
        self.template_bins = bins
        self.template_pdf = np.hstack([0.0, pdf, 0.0])
        self.template_cdf = np.hstack([0.0, cdf, 1.0])
        self.template_bin_centers = (bins - (bins - np.roll(bins, 1)) / 2.0) [1:]
        super(ScipyTemplateDistribution, self).__init__()

    def _pdf(self, x):
        return self.template_pdf[np.digitize(x, bins=self.template_bins)]
    
    def _cdf(self, x):
        return self.template_cdf[np.digitize(x, bins=self.template_bins)]
    
    def _rvs(self):
        probabilities = self.template_pdf[1:-1]
        probabilities /= probabilities.sum()
        choices = np.random.choice(len(self.template_pdf) - 2, size=self._size, p=probabilities)
        return self.template_bin_centers[choices]


class PDF(object):
    @classmethod
    def from_scipy(cls, label, scipy_distribution, norm = 1.0, **parameters):
        parameters = {'{}_{}'.format(label, key): value for key, value in parameters.items()}
        parameters['{}_norm'.format(label)] = norm
        return cls([label], [scipy_distribution], parameters)

    @classmethod
    def from_data(cls, label, data, norm = 1.0, **kwargs):
        return cls([label], [ScipyTemplateDistribution(data, **kwargs)], {'{}_norm'.format(label): norm})

    def __init__(self, labels, distributions, parameters, constraints={}, sub_pdfs=[]):
        self.labels = labels
        self.distributions = distributions
        self.parameters = parameters
        self.scipy = ScipyDistributionGenerator(self)
        self.constraints = constraints
        self.sub_pdfs = sub_pdfs if sub_pdfs else [self]

    def __add__(self, other):
        parameters = {}
        parameters.update(self.parameters)
        parameters.update(other.parameters)
        constraints = {}
        constraints.update(self.constraints)
        constraints.update(other.constraints)
        return PDF(self.labels + other.labels, self.distributions + other.distributions, parameters, constraints, self.sub_pdfs + other.sub_pdfs)

    def __iter__(self):
        return iter(self.sub_pdfs)

    def add_constraint(self, parameter, function):
        self.constraints[parameter] = function

    def fix_parameter(self, parameter):
        self.add_constraint(parameter, lambda p: self.parameters[parameter])

    def get_free_parameters(self):
        return {parameter: value for parameter, value in self.parameters.items() if parameter not in self.constraints}

    def get_initial_parameters(self):
        return np.array(list(self.get_free_parameters().values()))

    def get_result_parameters(self, res):
        parameters = {k: v for k, v in zip(self.get_free_parameters().keys(), res.x)}
        for parameter, function in self.constraints.items():
            parameters[parameter] = function(parameters)
        return parameters 

    def get_maximumLikelihood_loss(self):
        def loss(free_parameters, data):
            parameters = {key: free_parameters[i] for i, key in enumerate(self.get_free_parameters().keys())}
            for parameter, function in self.constraints.items():
                parameters[parameter] = function(parameters)
            return -np.sum(np.log(self.scipy.pdf(data, **parameters)))
        return loss


if __name__ == '__main__':

    signal = PDF.from_scipy('signal', scipy.stats.norm, loc=0.5, scale=0.1)
    
    background = PDF.from_scipy('background', scipy.stats.norm, loc=0.5, scale=1.0)
    
    continuum_data = scipy.stats.norm.rvs(size=10000, loc=0, scale=2.0)
    continuum = PDF.from_data('continuum', continuum_data)

    fit_model = signal + background + continuum
    fit_model.fix_parameter('signal_scale')
    fit_model.fix_parameter('background_scale')
    fit_model.add_constraint('continuum_norm', lambda p: 1.0 - p['signal_norm'] - p['background_norm'])

    data = fit_model.scipy.rvs(size=1000, signal_loc=1.0, background_loc=0.0, signal_norm=0.3, background_norm=0.5, continuum_norm=0.2, signal_scale=0.1, background_scale=1.0)
    plt.hist(data, bins=100, range=(-3, 3), normed=True)
    plt.show()

    r = scipy.optimize.minimize(fit_model.get_maximumLikelihood_loss(), fit_model.get_initial_parameters(), args=(data,), method='nelder-mead')
    print("Raw scipy result", r)
    result = fit_model.get_result_parameters(r)
    print("Result parameters", result)
    
    plt.hist(data, bins=100, range=(-3, 3), normed=True)
    X = np.linspace(-3, 3, 100)
    plt.plot(X, fit_model.scipy.pdf(X, **result), label='Full fit_model')
    for component in fit_model:
        r = {key: value for key, value in result.items() if key.startswith(component.labels[0])}
        plt.plot(X, component.scipy.pdf(X, **r), label=component.labels[0])
    plt.show()

