#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
import functools

class PDF(object):
    def __init__(self, func, free_parameters={}, fixed_parameters={}, label=''):
        self.func = func
        self.free_parameters = free_parameters
        self.fixed_parameters = fixed_parameters
        self.label = label

    def __add__(self, other):
        def func(x, **free_parameters):
            return (self.func(x, **{p: free_parameters['{}_{}'.format(self.label, p)] for p in self.free_parameters}) +
                    other.func(x, **{p: free_parameters['{}_{}'.format(other.label, p)] for p in other.free_parameters}))

        free_parameters = {}
        for k, v in self.free_parameters.items():
            free_parameters['{}_{}'.format(self.label, k)] = v
        for k, v in other.free_parameters.items():
            free_parameters['{}_{}'.format(other.label, k)] = v
        
        fixed_parameters = {}
        for k, v in self.fixed_parameters.items():
            fixed_parameters['{}_{}'.format(self.label, k)] = v
        for k, v in other.fixed_parameters.items():
            fixed_parameters['{}_{}'.format(other.label, k)] = v

        return PDF(func, free_parameters, fixed_parameters, label='({}+{})'.format(self.label, other.label))

    def __call__(self, data, **free_parameters):
        return self.func(data, **free_parameters, **self.fixed_parameters)

    def GetInitialParameters(self):
        return np.array(list(self.free_parameters.values()))

    def GetFreeParametersFromFitResult(self, res):
        return {k: v for k, v in zip(self.free_parameters.keys(), res.x)}

    def GetLoss(self, method):
        if method == 'MaximumLikelihood':
            def loss(parameters, data):
                return -np.sum(np.log(self.func(data, **{k: parameters[i] for i, k in enumerate(self.free_parameters)})))
        else:
            return NotImplemented
        return loss

    def GetScipyDistribution(self, parameters):
        class ScipyDistributionGenerator(scipy.stats.rv_continuous):
            cdf_cache = {}
            def _pdf(other_self, data):
                return self.func(data, **self.free_parameters, **self.fixed_parameters)
            def _cdf(other_self, data):
                data = (data * 100).astype(int)
                result = []
                for d in data:
                    if d not in other_self.cdf_cache:
                        other_self.cdf_cache[d] = super(ScipyDistributionGenerator, other_self)._cdf(d / 100.0)
                    result.append(other_self.cdf_cache[d])
                return np.array(result)
                
            # TODO Implement _cdf for faster rvs generation
        return ScipyDistributionGenerator()


class Template(object):
    def __init__(self, histogram):
        self.bins, self.counts = histogram
        total_counts = np.sum(self.counts)
        self.pdf = self.counts / total_counts

    def __call__(self, data):
        return self.pdf[np.digitize(data, self.bins)]


if __name__ == '__main__':

    Signal =  PDF(scipy.stats.norm.pdf, free_parameters={'loc': 0.5}, fixed_parameters={'scale': 0.1}, label='Signal')
    Background =  PDF(scipy.stats.norm.pdf, free_parameters={'loc': 0.5}, fixed_parameters={'scale': 1.0}, label='Background')
    TemplateBackground =  PDF(Template(histogram), free_parameters={'loc': 0.5}, fixed_parameters={'scale': 1.0}, label='Background')
    Model = Signal + Background

    data = Model.GetScipyDistribution(parameters={'Signal_loc': 1.0, 'Background_loc': 0.0}).rvs(size=500)
    print(data)

    r = scipy.optimize.minimize(Model.GetLoss('MaximumLikelihood'), Model.GetInitialParameters(), args=(data,), method='nelder-mead')
    print(r)
    
    plt.hist(data, bins=100, range=(-3, 3), normed=True)
    X = np.linspace(-3, 3, 100)
    plt.plot(X, Model(X, **Model.GetFreeParametersFromFitResult(r)))
    plt.show()

    print(Model.GetFreeParametersFromFitResult(r))



